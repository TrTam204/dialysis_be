from django.utils import timezone as dj_timezone
from django.db import transaction
from django.db.models import ProtectedError
from django.shortcuts import get_object_or_404
from django_filters import rest_framework as django_filters
from rest_framework import filters, permissions, status, viewsets
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.views import APIView
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response

from .audit import AuditLogMixin, record_audit_log
from .models import (
    AuditLog,
    BloodSample,
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
    ScheduleAssignment,
    SchedulePlan,
    VitalSign,
    MedicalMessage,
)
from .permissions import (
    IsAdmin,
    IsAdminOrDoctor,
    IsNurseAssignedSessionStatusOnly,
    IsNurseAssignedSessionVitals,
    IsNursePatientStatusOnly,
    IsStaffMember,
)
from .serializers import (
    AuditLogSerializer,
    BloodSampleSerializer,
    CustomUserSerializer,
    DepartmentSerializer,
    DialysisMachineSerializer,
    DialysisSessionSerializer,
    PatientSerializer,
    ScheduleAssignmentSerializer,
    SchedulePlanSerializer,
    VitalSignSerializer,
    MedicalMessageSerializer,
)
from .kmeans_service import KMeansClusteringService
from .ai_bridge import predict_idh_for_session, get_lstm_model_info
from .services import (
    approve_schedule_plan,
    dashboard_dialysis_stats,
    dashboard_machine_stats,
    dashboard_summary,
    parse_date_range,
    reject_schedule_plan,
    report_machine_utilization,
    report_operational_summary,
)
from .reports import (
    BloodSampleReportView,
    MachineReportView,
    NurseWorkloadReportView,
    PatientReportView,
    SessionReportView,
)


class BaseRoleAwareViewSet(AuditLogMixin, viewsets.ModelViewSet):
    filter_backends = [
        django_filters.DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]


class DepartmentViewSet(BaseRoleAwareViewSet):
    queryset = Department.objects.all().order_by('id')
    serializer_class = DepartmentSerializer
    filterset_fields = ['is_active']
    search_fields = ['name', 'code', 'description']
    ordering_fields = ['id', 'name', 'code', 'is_active', 'created_at']
    ordering = ['id']

    def get_permissions(self):
        if self.action in {'create', 'update', 'partial_update', 'destroy'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()


class CustomUserViewSet(BaseRoleAwareViewSet):
    queryset = CustomUser.objects.all().order_by('id')
    serializer_class = CustomUserSerializer
    filterset_fields = ['role', 'department', 'is_active']
    search_fields = ['username', 'first_name', 'last_name', 'email', 'phone_number', 'department__name']
    ordering_fields = ['id', 'username', 'email', 'role', 'is_active', 'created_at']
    ordering = ['id']

    def get_permissions(self):
        if self.action in {'create', 'update', 'partial_update', 'destroy'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.id == request.user.id:
            return Response({'detail': 'You cannot delete your own account.'}, status=400)
        return super().destroy(request, *args, **kwargs)


class PatientViewSet(BaseRoleAwareViewSet):
    queryset = Patient.objects.select_related('department').all().order_by('patient_id')
    serializer_class = PatientSerializer
    filterset_fields = ['status', 'gender', 'date_of_birth', 'department']
    search_fields = ['patient_id', 'full_name', 'phone_number']
    ordering_fields = ['patient_id', 'full_name', 'status', 'date_of_birth', 'created_at']
    ordering = ['patient_id']

    def get_permissions(self):
        if self.action == 'create':
            self.permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]
        elif self.action in {'update', 'transfer'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]
        elif self.action == 'partial_update':
            # NURSE gets a workflow-limited update: PATCH status only.
            self.permission_classes = [
                permissions.IsAuthenticated,
                IsAdminOrDoctor | IsNursePatientStatusOnly,
            ]
        elif self.action == 'destroy':
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()

    @action(detail=True, methods=['post'], url_path='transfer')
    def transfer(self, request, pk=None):
        patient = self.get_object()
        target_dept_id = request.data.get('target_department_id') or request.data.get('department') or request.data.get('department_id')
        reason = request.data.get('reason', '')

        if not target_dept_id:
            return Response({'detail': 'Cần cung cấp target_department_id để chuyển khoa.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            target_dept = Department.objects.get(id=target_dept_id, is_active=True)
        except Department.DoesNotExist:
            return Response({'detail': 'Khoa đích không tồn tại hoặc đã bị vô hiệu hóa.'}, status=status.HTTP_404_NOT_FOUND)

        if patient.department_id == target_dept.id:
            return Response({'detail': f'Bệnh nhân hiện đã thuộc {target_dept.name}.'}, status=status.HTTP_400_BAD_REQUEST)

        old_dept_name = patient.department.name if patient.department else 'Chưa có'
        old_dept_id = patient.department_id

        with transaction.atomic():
            patient.department = target_dept
            patient.save(update_fields=['department', 'updated_at'])

            record_audit_log(
                actor=request.user,
                action=AuditLog.Action.UPDATE,
                instance=patient,
                changes={
                    'action': 'TRANSFER_DEPARTMENT',
                    'from_department_id': old_dept_id,
                    'to_department_id': target_dept.id,
                    'from_department_name': old_dept_name,
                    'to_department_name': target_dept.name,
                    'reason': reason,
                },
            )

        serializer = self.get_serializer(patient)
        return Response({
            'success': True,
            'detail': f'Đã chuyển bệnh nhân {patient.full_name} sang {target_dept.name} thành công.',
            'patient': serializer.data,
        }, status=status.HTTP_200_OK)


class DialysisMachineViewSet(BaseRoleAwareViewSet):
    queryset = DialysisMachine.objects.select_related('department').all().order_by('machine_id')
    serializer_class = DialysisMachineSerializer
    filterset_fields = ['status', 'department']
    search_fields = ['machine_id', 'name']
    ordering_fields = ['machine_id', 'name', 'status', 'department', 'created_at']
    ordering = ['machine_id']

    def get_permissions(self):
        if self.action in {'create', 'update', 'partial_update', 'destroy'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()


class DialysisSessionFilter(django_filters.FilterSet):
    date = django_filters.DateFilter(field_name='scheduled_start', lookup_expr='date')
    date_from = django_filters.DateTimeFilter(field_name='scheduled_start', lookup_expr='gte')
    date_to = django_filters.DateTimeFilter(field_name='scheduled_end', lookup_expr='lte')

    class Meta:
        model = DialysisSession
        fields = ['patient', 'machine', 'assigned_nurse', 'status']


class DialysisSessionViewSet(BaseRoleAwareViewSet):
    queryset = DialysisSession.objects.select_related('patient', 'machine', 'assigned_nurse').all().order_by('-scheduled_start')
    serializer_class = DialysisSessionSerializer
    filterset_class = DialysisSessionFilter
    search_fields = ['session_id', 'patient__full_name', 'patient__patient_id', 'machine__name', 'assigned_nurse__username']
    ordering_fields = ['session_id', 'scheduled_start', 'scheduled_end', 'status', 'created_at']
    ordering = ['-scheduled_start']

    def get_permissions(self):
        if self.action in {'create', 'update'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]
        elif self.action == 'partial_update':
            # NURSE gets a workflow-limited update: PATCH status of assigned sessions only.
            self.permission_classes = [
                permissions.IsAuthenticated,
                IsAdminOrDoctor | IsNurseAssignedSessionStatusOnly,
            ]
        elif self.action == 'destroy':
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()

    @action(detail=True, methods=['get', 'post'], url_path='vitals')
    def vitals(self, request, pk=None):
        session = get_object_or_404(DialysisSession, pk=pk)

        if request.method == 'GET':
            queryset = VitalSign.objects.filter(session=session).order_by('-recorded_at')
            serializer = VitalSignSerializer(queryset, many=True)
            return Response(serializer.data)

        if request.user.role == CustomUser.Role.NURSE and session.assigned_nurse_id != request.user.id:
            return Response(
                {'detail': 'NURSE can only record vitals for sessions assigned to them.'},
                status=403,
            )

        serializer = VitalSignSerializer(data=request.data, context={'session': session, 'request': request})
        serializer.is_valid(raise_exception=True)
        with transaction.atomic():
            vital = serializer.save(session=session, recorded_by=request.user)
            record_audit_log(actor=request.user, action=AuditLog.Action.CREATE, instance=vital)
        return Response(serializer.data, status=201)

    @action(detail=True, methods=['get'], url_path='idh-prediction')
    def idh_prediction(self, request, pk=None):
        session = self.get_object()
        # RBAC Cross-Department check (Mục V & X Master Plan)
        user = request.user
        sess_dept_id = session.machine.department_id if session.machine else (
            session.patient.department_id if session.patient else None
        )
        if user.role == CustomUser.Role.DOCTOR:
            if not user.department_id or (sess_dept_id and sess_dept_id != user.department_id):
                return Response(
                    {'detail': 'Bác sĩ không có quyền truy cập dự báo ca lọc của khoa khác.'},
                    status=status.HTTP_403_FORBIDDEN,
                )
        elif user.role == CustomUser.Role.NURSE:
            is_assigned = (session.assigned_nurse_id == user.id)
            is_same_dept = bool(user.department_id and sess_dept_id == user.department_id)
            if not (is_assigned or is_same_dept):
                return Response(
                    {'detail': 'Điều dưỡng không có quyền truy cập dự báo ca lọc ngoài khoa hoặc không được phân công.'},
                    status=status.HTTP_403_FORBIDDEN,
                )

        result = predict_idh_for_session(session)
        return Response(result, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'], url_path='sos')
    def emergency_sos(self, request, pk=None):
        session = self.get_object()
        user = request.user
        reason = request.data.get('reason', 'Khẩn cấp không xác định')
        notes = request.data.get('notes', '')

        now = dj_timezone.now()
        now_local = dj_timezone.localtime(now)
        sos_info = {
            'triggered_by': user.username,
            'triggered_by_name': f"{user.last_name} {user.first_name}".strip() or user.username,
            'triggered_by_role': user.role,
            'reason': reason,
            'notes': notes,
            'timestamp': now_local.isoformat(),
        }

        with transaction.atomic():
            session.is_emergency = True
            session.emergency_details = sos_info
            sos_prefix = f"\n[🚨 BÁO ĐỘNG SOS {now_local.strftime('%d/%m/%Y %H:%M:%S')}] {user.username} ({user.role}): {reason}"
            if notes:
                sos_prefix += f" - {notes}"
            session.clinical_notes = (session.clinical_notes or '') + sos_prefix
            session.save(update_fields=['is_emergency', 'emergency_details', 'clinical_notes', 'updated_at'])

            record_audit_log(
                actor=user,
                action=AuditLog.Action.UPDATE,
                instance=session,
                changes={'EMERGENCY_SOS': sos_info},
            )

            # Auto-broadcast emergency alert to department chat
            try:
                dept = session.patient.department or user.department
                if dept:
                    MedicalMessage.objects.create(
                        sender=user,
                        receiver=None,
                        department=dept,
                        content=f"🚨 [BÁO ĐỘNG SOS KHẨN CẤP] Ca lọc #{session.session_id} - Bệnh nhân {session.patient.full_name} gặp sự cố cấp tính: {reason}. Đề nghị Bác sĩ và kíp trực hỗ trợ ngay!",
                        session_reference=session,
                    )
            except Exception:
                pass

        protocol_guidelines = [
            {'step': 1, 'title': 'Ngừng rút dịch (UF rate = 0)', 'action': 'Lập tức chuyển tốc độ siêu lọc UF rate về 0 mL/h trên máy lọc.'},
            {'step': 2, 'title': 'Tư thế cấp cứu Trendelenburg', 'action': 'Hạ đầu giường thấp, nâng cao chân bệnh nhân 30-45 độ để duy trì tuần hoàn não.'},
            {'step': 3, 'title': 'Truyền bù dịch đẳng trương', 'action': 'Mở khóa đường truyền xả nhanh 100 - 200 mL dung dịch NaCl 0.9% qua cổng tĩnh mạch quả lọc.'},
            {'step': 4, 'title': 'Giảm tốc độ bơm máu Qb', 'action': 'Hạ lưu lượng dòng máu Qb xuống 150 - 200 mL/phút, hạ nhiệt độ dịch lọc xuống 35.5 - 36.0 °C.'},
            {'step': 5, 'title': 'Báo động Bác sĩ phụ trách kíp trực', 'action': f'Bác sĩ khoa tiếp nhận xử lý cấp cứu tại giường máy {session.machine.name if session.machine else session.session_id} ngay lập tức.'},
        ]

        return Response({
            'success': True,
            'session_id': session.session_id,
            'is_emergency': True,
            'emergency_details': sos_info,
            'protocol_guidelines': protocol_guidelines,
            'detail': f'Đã kích hoạt báo động khẩn cấp SOS cho phiên lọc {session.session_id}.',
        }, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'], url_path='clear-sos')
    def clear_sos(self, request, pk=None):
        session = self.get_object()
        user = request.user
        with transaction.atomic():
            session.is_emergency = False
            session.save(update_fields=['is_emergency', 'updated_at'])
            record_audit_log(
                actor=user,
                action=AuditLog.Action.UPDATE,
                instance=session,
                changes={'EMERGENCY_SOS_CLEARED': {'cleared_by': user.username, 'timestamp': dj_timezone.now().isoformat()}},
            )
        return Response({'success': True, 'is_emergency': False, 'detail': 'Đã hủy trạng thái báo động khẩn cấp SOS.'}, status=status.HTTP_200_OK)


class BloodSampleViewSet(BaseRoleAwareViewSet):
    queryset = BloodSample.objects.select_related('patient', 'created_by').all().order_by('-collection_date')
    serializer_class = BloodSampleSerializer
    filterset_fields = ['patient', 'collection_date']
    search_fields = ['sample_id', 'patient__full_name', 'patient__patient_id', 'notes']
    ordering_fields = ['sample_id', 'collection_date', 'created_at']
    ordering = ['-collection_date']

    def get_permissions(self):
        if self.action in {'create', 'update', 'partial_update'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]
        elif self.action == 'destroy':
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class VitalSignViewSet(BaseRoleAwareViewSet):
    queryset = VitalSign.objects.select_related('session', 'recorded_by').all().order_by('-recorded_at')
    serializer_class = VitalSignSerializer
    filterset_fields = ['session', 'recorded_by']
    search_fields = ['notes']
    ordering_fields = ['id', 'recorded_at', 'created_at']
    ordering = ['-recorded_at']

    def get_permissions(self):
        if self.action == 'create':
            self.permission_classes = [
                permissions.IsAuthenticated,
                IsAdminOrDoctor | IsNurseAssignedSessionVitals,
            ]
        elif self.action in {'update', 'partial_update'}:
            self.permission_classes = [
                permissions.IsAuthenticated,
                IsAdminOrDoctor | IsNurseAssignedSessionVitals,
            ]
        elif self.action == 'destroy':
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()

    def perform_create(self, serializer):
        serializer.save(recorded_by=self.request.user)


# ---------------------------------------------------------------------------
# Dashboard API (basic operational statistics, Phase 1 only)
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def dashboard_summary_view(request):
    return Response(dashboard_summary())


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def dashboard_dialysis_stats_view(request):
    date_from_str = request.query_params.get('date_from')
    date_to_str = request.query_params.get('date_to')
    if date_from_str or date_to_str:
        date_from, date_to, error_response = parse_date_range(request)
        if error_response:
            return error_response
        return Response(dashboard_dialysis_stats(date_from=date_from, date_to=date_to))

    days = request.query_params.get('days', 7)
    try:
        days = max(1, min(90, int(days)))
    except (TypeError, ValueError):
        days = 7
    return Response(dashboard_dialysis_stats(days=days))


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def dashboard_machine_stats_view(request):
    return Response(dashboard_machine_stats())


# ---------------------------------------------------------------------------
# Reports API (Milestone 7: Reports & Analytics)
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated, IsStaffMember])
def report_operational_summary_view(request):
    date_from, date_to, error_response = parse_date_range(request)
    if error_response:
        return error_response
    return Response(report_operational_summary(date_from=date_from, date_to=date_to))


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated, IsStaffMember])
def report_machine_utilization_view(request):
    date_from, date_to, error_response = parse_date_range(request)
    if error_response:
        return error_response
    return Response(report_machine_utilization(date_from=date_from, date_to=date_to))


class AuditLogFilter(django_filters.FilterSet):
    actor = django_filters.NumberFilter(field_name='actor__id')
    actor_username = django_filters.CharFilter(field_name='actor__username', lookup_expr='icontains')
    action = django_filters.ChoiceFilter(choices=AuditLog.Action.choices)
    entity_type = django_filters.CharFilter(lookup_expr='iexact')
    entity_id = django_filters.CharFilter(lookup_expr='iexact')
    date_from = django_filters.DateTimeFilter(field_name='timestamp', lookup_expr='gte')
    date_to = django_filters.DateTimeFilter(field_name='timestamp', lookup_expr='lte')

    class Meta:
        model = AuditLog
        fields = ['actor', 'actor_username', 'action', 'entity_type', 'entity_id', 'date_from', 'date_to']


class AuditLogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AuditLog.objects.select_related('actor').exclude(entity_type__iexact='MedicalMessage').order_by('-timestamp')
    serializer_class = AuditLogSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdmin]
    filter_backends = [
        django_filters.DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_class = AuditLogFilter
    search_fields = ['entity_type', 'entity_id', 'actor__username', 'actor__first_name', 'actor__last_name']
    ordering_fields = ['id', 'timestamp', 'entity_type', 'action']
    ordering = ['-timestamp']

    def get_queryset(self):
        return super().get_queryset().exclude(entity_type__iexact='MedicalMessage')


class SchedulePlanViewSet(BaseRoleAwareViewSet):
    queryset = (
        SchedulePlan.objects.select_related('department', 'created_by', 'approved_by')
        .prefetch_related('assignments')
        .all()
        .order_by('-week_start', '-created_at')
    )
    serializer_class = SchedulePlanSerializer
    filterset_fields = ['department', 'week_start', 'status']
    search_fields = ['name']
    ordering_fields = ['id', 'week_start', 'week_end', 'status', 'created_at']
    ordering = ['-week_start']

    def get_queryset(self):
        qs = super().get_queryset()
        user = self.request.user
        if not user or not user.is_authenticated:
            return qs.none()
        if user.role == 'ADMIN':
            return qs
        if user.department_id:
            return qs.filter(department_id=user.department_id)
        return qs.none()

    def get_permissions(self):
        if self.action in {'create', 'update', 'partial_update', 'approve', 'reject', 'generate'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]
        elif self.action == 'destroy':
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    @action(detail=False, methods=['post'], url_path='generate')
    def generate(self, request):
        department_id = request.data.get('department')
        week_start_str = request.data.get('week_start')

        if not department_id or not week_start_str:
            return Response({'detail': 'department and week_start are required.'}, status=status.HTTP_400_BAD_REQUEST)

        # RBAC: Doctor can only generate for their own department
        if request.user.role == 'DOCTOR':
            if not request.user.department_id:
                return Response({'detail': 'Bác sĩ chưa được phân công khoa phòng.'}, status=status.HTTP_403_FORBIDDEN)
            try:
                dept_id_int = int(department_id)
            except (ValueError, TypeError):
                return Response({'detail': 'department must be an integer ID.'}, status=status.HTTP_400_BAD_REQUEST)
            if dept_id_int != request.user.department_id:
                return Response({'detail': 'Bác sĩ chỉ có quyền tạo lịch cho khoa của mình.'}, status=status.HTTP_403_FORBIDDEN)

        try:
            from datetime import datetime
            week_start = datetime.strptime(week_start_str, '%Y-%m-%d').date()
            if week_start.weekday() != 0:
                return Response({'detail': 'Tuần bắt đầu phải là Thứ 2.'}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError:
            return Response({'detail': 'week_start must be YYYY-MM-DD format.'}, status=status.HTTP_400_BAD_REQUEST)

        seed = request.data.get('seed')
        if seed is not None and seed != '':
            try:
                seed = int(seed)
            except (ValueError, TypeError):
                seed = None
        else:
            seed = None

        from core.ga_service import GeneticAlgorithmService
        plan, err = GeneticAlgorithmService.generate_plan(department_id, week_start, request.user, seed=seed)
        if err:
            return Response({'detail': err}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(plan)
        return Response({'plan': serializer.data, 'detail': 'Khởi tạo kế hoạch bằng GA thành công.'}, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'], url_path='approve')
    def approve(self, request, pk=None):
        plan = self.get_object()
        if request.user.role == 'DOCTOR' and request.user.department_id and plan.department_id != request.user.department_id:
            return Response({'detail': 'Bác sĩ chỉ có quyền phê duyệt kế hoạch thuộc khoa của mình.'}, status=status.HTTP_403_FORBIDDEN)
        try:
            approved_plan, warnings, sessions = approve_schedule_plan(plan.pk, request.user)
            serializer = self.get_serializer(approved_plan)
            return Response(
                {
                    'plan': serializer.data,
                    'created_sessions_count': len(sessions),
                    'warnings': warnings,
                    'detail': 'Kế hoạch đã được phê duyệt thành công.',
                },
                status=status.HTTP_200_OK,
            )
        except ValidationError as e:
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'], url_path='reject')
    def reject(self, request, pk=None):
        plan = self.get_object()
        if request.user.role == 'DOCTOR' and request.user.department_id and plan.department_id != request.user.department_id:
            return Response({'detail': 'Bác sĩ chỉ có quyền từ chối kế hoạch thuộc khoa của mình.'}, status=status.HTTP_403_FORBIDDEN)
        reason = request.data.get('reason', '')
        try:
            rejected_plan = reject_schedule_plan(plan.pk, request.user, reason)
            serializer = self.get_serializer(rejected_plan)
            return Response(
                {
                    'plan': serializer.data,
                    'detail': 'Kế hoạch đã bị từ chối.',
                },
                status=status.HTTP_200_OK,
            )
        except ValidationError as e:
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)


class ScheduleAssignmentViewSet(BaseRoleAwareViewSet):
    queryset = (
        ScheduleAssignment.objects.select_related(
            'schedule_plan',
            'patient',
            'machine',
            'original_machine',
        )
        .all()
        .order_by('scheduled_date', 'shift')
    )
    serializer_class = ScheduleAssignmentSerializer
    filterset_fields = ['schedule_plan', 'patient', 'machine', 'scheduled_date', 'shift', 'source']
    search_fields = ['patient__full_name', 'patient__patient_id', 'machine__name']
    ordering_fields = ['id', 'scheduled_date', 'shift', 'created_at']
    ordering = ['scheduled_date', 'shift']

    def get_queryset(self):
        qs = super().get_queryset()
        user = self.request.user
        if not user or not user.is_authenticated:
            return qs.none()
        if user.role == 'ADMIN':
            return qs
        if user.department_id:
            return qs.filter(schedule_plan__department_id=user.department_id)
        return qs.none()

    def get_permissions(self):
        if self.action in {'create', 'update', 'partial_update'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]
        elif self.action == 'destroy':
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()


class KMeansReportView(APIView):
    """
    Unsupervised Machine Learning: Pure Python K-Means patient clustering report.
    Clusters hemodialysis patients based on 4 clinical/biochemical features:
    age, avg_hemoglobin, avg_potassium, and avg_idwg_pct.
    Accessible only by ADMIN and DOCTOR.
    """
    permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]

    def get(self, request):
        k_param = request.query_params.get('k', 3)
        try:
            k = int(k_param)
            if k <= 0:
                return Response(
                    {'detail': 'k must be a positive integer.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except (ValueError, TypeError):
            return Response(
                {'detail': 'Invalid k parameter. Must be an integer.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        seed_param = request.query_params.get('seed', 42)
        try:
            seed = int(seed_param)
        except (ValueError, TypeError):
            seed = 42

        inc_disc_param = request.query_params.get('include_discharged', 'false').lower()
        include_discharged = inc_disc_param in {'true', '1', 'yes'}

        department_id = request.query_params.get('department')
        if department_id:
            try:
                department_id = int(department_id)
            except (ValueError, TypeError):
                return Response(
                    {'detail': 'Invalid department ID.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            # If user is DOCTOR and department is not specified, scope to their department if it has patients
            if getattr(request.user, 'role', None) == CustomUser.Role.DOCTOR:
                doctor_dept_id = getattr(request.user, 'department_id', None)
                if doctor_dept_id and Patient.objects.filter(department_id=doctor_dept_id).exists():
                    department_id = doctor_dept_id

        # RBAC Cross-Department check (Mục V & IX Master Plan)
        user_role = getattr(request.user, 'role', None)
        user_dept_id = getattr(request.user, 'department_id', None)

        if user_role in (CustomUser.Role.DOCTOR, CustomUser.Role.NURSE):
            if not user_dept_id:
                return Response(
                    {'detail': 'Nhân viên y tế chưa được phân bổ khoa quản lý.'},
                    status=status.HTTP_403_FORBIDDEN,
                )
            if department_id is not None and int(department_id) != user_dept_id:
                return Response(
                    {'detail': 'Bạn không có quyền xem phân cụm bệnh nhân của khoa khác.'},
                    status=status.HTTP_403_FORBIDDEN,
                )
            department_id = user_dept_id

        service = KMeansClusteringService(
            department_id=department_id,
            include_discharged=include_discharged,
        )
        result = service.run_kmeans(k=k, seed=seed)

        return Response({
            'success': True,
            'k_value': result['k_value'],
            'seed': seed,
            'disclaimer': result.get(
                'disclaimer',
                'Kết quả phân cụm K-Means chỉ phản ánh độ tương đồng dữ liệu thống kê, dùng để tham khảo phân nhóm quản lý, không thay thế chẩn đoán lâm sàng của bác sĩ.'
            ),
            'wcss': result['wcss'],
            'silhouette_score': result['silhouette_score'],
            'k_evaluation': result.get('k_evaluation', []),
            'recommended_k': result.get('recommended_k', result['k_value']),
            'centroids': result['centroids'],
            'denormalized_centroids': result.get('denormalized_centroids', []),
            'cluster_summaries': result.get('cluster_summaries', []),
            'min_max': result.get('min_max', {}),
            'clusters': result['clusters'],
            'patients': result.get('patients', []),
            'iterations': result.get('iterations', 0),
            'converged': result.get('converged', True),
        }, status=status.HTTP_200_OK)


class LSTMModelInfoView(APIView):
    """
    Exposes LSTM IDH prediction model architecture, evaluation metrics
    (Accuracy, Precision, Recall, F1, Loss, Confusion Matrix), and training curves.
    """
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def get(self, request):
        info = get_lstm_model_info()
        return Response(info, status=status.HTTP_200_OK)


def format_status_text(last_seen, now=None):
    if not last_seen:
        return 'Không hoạt động'
    if now is None:
        now = dj_timezone.now()
    diff = (now - last_seen).total_seconds()
    if diff <= 300:
        return 'Đang hoạt động'
    minutes = int(diff // 60)
    if minutes < 60:
        return f'Hoạt động {minutes} phút trước'
    hours = int(minutes // 60)
    if hours < 24:
        return f'Hoạt động {hours} giờ trước'
    days = int(hours // 24)
    return f'Hoạt động {days} ngày trước'


class OnlineStatusView(APIView):
    """
    Returns list of online doctors, nurses, and admins across the hospital.
    """
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def get(self, request):
        user = request.user
        now = dj_timezone.now()

        from django.core.cache import cache
        cache.set(f'user_online_{user.id}', True, timeout=300)
        CustomUser.objects.filter(pk=user.pk).update(last_seen=now)

        qs = CustomUser.objects.filter(is_active=True).select_related('department')

        online_staff = []
        for staff in qs:
            is_cached = bool(cache.get(f'user_online_{staff.id}'))
            is_recent = bool(staff.last_seen and (now - staff.last_seen).total_seconds() <= 300)
            is_online = is_cached or is_recent or (staff.id == user.id)

            full_name = f"{staff.last_name} {staff.first_name}".strip() or staff.username
            status_text = 'Đang hoạt động' if is_online else format_status_text(staff.last_seen, now=now)
            dept_name = staff.department.name if staff.department else ('Toàn viện' if staff.role == CustomUser.Role.ADMIN else 'Chưa phân khoa')

            online_staff.append({
                'id': staff.id,
                'username': staff.username,
                'full_name': full_name,
                'role': staff.role,
                'department_id': staff.department_id,
                'department_name': dept_name,
                'is_online': is_online,
                'last_seen': staff.last_seen.isoformat() if staff.last_seen else None,
                'status_text': status_text,
            })

        online_staff.sort(key=lambda s: (not s['is_online'], s['role'], s['full_name']))

        return Response({
            'count_online': sum(1 for s in online_staff if s['is_online']),
            'total_staff': len(online_staff),
            'staff': online_staff,
        }, status=status.HTTP_200_OK)


class MedicalMessageViewSet(BaseRoleAwareViewSet):
    queryset = MedicalMessage.objects.select_related('sender', 'receiver', 'department', 'session_reference').all().order_by('created_at')
    serializer_class = MedicalMessageSerializer
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    search_fields = ['content', 'sender__username', 'receiver__username']

    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        if request.user and request.user.is_authenticated:
            now = dj_timezone.now()
            from django.core.cache import cache
            cache.set(f'user_online_{request.user.id}', True, timeout=300)
            CustomUser.objects.filter(pk=request.user.pk).update(last_seen=now)

    def get_queryset(self):
        user = self.request.user
        if not user or not user.is_authenticated:
            return MedicalMessage.objects.none()

        channel = self.request.query_params.get('channel')
        receiver_id = self.request.query_params.get('receiver') or self.request.query_params.get('recipient_id') or self.request.query_params.get('recipient')
        recipient_isnull = self.request.query_params.get('recipient__isnull') or self.request.query_params.get('receiver__isnull')

        from django.db.models import Q
        base_qs = MedicalMessage.objects.select_related('sender', 'receiver', 'department', 'session_reference')

        if receiver_id:
            # 1-1 Direct Messages:
            # TUYET DOI KHONG loc theo department!
            try:
                target_id = int(receiver_id)
                qs = base_qs.filter(
                    (Q(sender=user, receiver_id=target_id)) |
                    (Q(sender_id=target_id, receiver=user))
                )
                # Auto mark unread incoming messages as read
                MedicalMessage.objects.filter(sender_id=target_id, receiver=user, is_read=False).update(is_read=True)
            except (ValueError, TypeError):
                return base_qs.none()
        elif channel == 'department' or recipient_isnull in {'true', 'True', '1'}:
            # Kenh Chung Toan Khoa (broadcast: receiver is NULL)
            if user.role == CustomUser.Role.ADMIN:
                dept_filter = self.request.query_params.get('department')
                if dept_filter:
                    qs = base_qs.filter(receiver__isnull=True, department_id=dept_filter)
                else:
                    qs = base_qs.filter(receiver__isnull=True)
            else:
                qs = base_qs.filter(
                    Q(receiver__isnull=True) &
                    (Q(department=user.department) | Q(department__isnull=True) | Q(sender__role='ADMIN'))
                )
            # Auto mark broadcast messages as read for this user
            unread_broadcasts = qs.exclude(sender=user)
            for b_msg in unread_broadcasts:
                readers = list(b_msg.read_by_users or [])
                if user.id not in readers:
                    readers.append(user.id)
                    b_msg.read_by_users = readers
                    b_msg.save(update_fields=['read_by_users'])
        else:
            # Mac dinh: Broadcast hop le HOAC tin 1-1 ma user la sender/receiver
            if user.role == CustomUser.Role.ADMIN:
                qs = base_qs.filter(
                    Q(receiver__isnull=True) |
                    Q(sender=user) |
                    Q(receiver=user)
                )
            else:
                qs = base_qs.filter(
                    (Q(receiver__isnull=True) & (Q(department=user.department) | Q(department__isnull=True) | Q(sender__role='ADMIN'))) |
                    Q(sender=user) |
                    Q(receiver=user)
                )

        session_id = self.request.query_params.get('session')
        if session_id:
            qs = qs.filter(session_reference_id=session_id)

        return qs.order_by('created_at')

    def perform_create(self, serializer):
        user = self.request.user
        receiver = serializer.validated_data.get('receiver')
        if receiver:
            dept = receiver.department or user.department
        else:
            dept = serializer.validated_data.get('department') or user.department

        content = serializer.validated_data.get('content', '').strip()
        att_type = serializer.validated_data.get('attachment_type')
        att_file = serializer.validated_data.get('attachment_file')
        metadata = serializer.validated_data.get('metadata') or {}

        # Default content if empty
        if not content:
            if att_type == 'AUDIO':
                content = '[🎤 Tin nhắn thoại]'
            elif att_type == 'IMAGE':
                content = '[📷 Hình ảnh]'
            elif att_type == 'VIDEO':
                content = '[🎥 Video]'
            elif att_type == 'FILE':
                content = '[📎 Tệp đính kèm]'
            elif metadata.get('mentions'):
                content = '[📋 Thẻ dữ liệu y tế]'
            else:
                content = '[Tin nhắn]'
            serializer.validated_data['content'] = content

        if att_file and not serializer.validated_data.get('attachment_name'):
            serializer.validated_data['attachment_name'] = getattr(att_file, 'name', 'dinh_kem')

        serializer.save(sender=user, department=dept)

    @action(detail=False, methods=['post'], url_path='mark-read')
    def mark_read(self, request):
        user = request.user
        target_id = request.data.get('target_id') or request.data.get('receiver')
        channel = request.data.get('channel')

        marked_count = 0
        if target_id:
            try:
                tid = int(target_id)
                marked_count = MedicalMessage.objects.filter(
                    sender_id=tid, receiver=user, is_read=False
                ).update(is_read=True)
            except (ValueError, TypeError):
                pass
        elif channel == 'department':
            from django.db.models import Q
            if user.role == CustomUser.Role.ADMIN:
                msgs = MedicalMessage.objects.filter(receiver__isnull=True).exclude(sender=user)
            else:
                msgs = MedicalMessage.objects.filter(
                    Q(receiver__isnull=True) &
                    (Q(department=user.department) | Q(department__isnull=True) | Q(sender__role='ADMIN'))
                ).exclude(sender=user)
            for msg in msgs:
                current_readers = list(msg.read_by_users or [])
                if user.id not in current_readers:
                    current_readers.append(user.id)
                    msg.read_by_users = current_readers
                    msg.save(update_fields=['read_by_users'])
                    marked_count += 1

        return Response({'status': 'success', 'marked_count': marked_count}, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], url_path='mention-suggestions')
    def mention_suggestions(self, request):
        q = request.query_params.get('q', '').strip()
        from django.db.models import Q

        # 1. Patients
        patient_qs = Patient.objects.all().select_related('department')
        if q:
            patient_qs = patient_qs.filter(
                Q(full_name__icontains=q) |
                Q(patient_id__icontains=q) |
                Q(phone_number__icontains=q)
            )
        patient_items = []
        for p in patient_qs[:10]:
            dept_label = p.department.name if p.department else "Khoa A"
            patient_items.append({
                'ref_type': 'PATIENT',
                'id': p.pk,
                'code': p.patient_id,
                'title': f"Bệnh nhân: {p.full_name} ({p.patient_id})",
                'subtitle': f"{dept_label} • Cân khô: {p.dry_weight or '--'} kg • TT: {p.status}",
                'doc_summary': f"Hồ sơ bệnh án & Kết quả xét nghiệm mới nhất của {p.full_name}",
                'link': f"/patients/{p.patient_id}",
            })

        # 2. Machines
        machine_qs = DialysisMachine.objects.all().select_related('department')
        if q:
            machine_qs = machine_qs.filter(
                Q(machine_id__icontains=q) |
                Q(name__icontains=q)
            )
        machine_items = []
        for m in machine_qs[:10]:
            dept_label = m.department.name if m.department else "Chung"
            machine_items.append({
                'ref_type': 'MACHINE',
                'id': m.pk,
                'code': m.machine_id,
                'title': f"Máy lọc: {m.name} ({m.machine_id})",
                'subtitle': f"Khoa: {dept_label} • Trạng thái: {m.status}",
                'doc_summary': f"Thông số kỹ thuật & Lịch bảo trì máy {m.machine_id}",
                'link': "/machines",
            })

        # 3. Staff
        staff_qs = CustomUser.objects.filter(is_active=True).select_related('department')
        if q:
            staff_qs = staff_qs.filter(
                Q(username__icontains=q) |
                Q(first_name__icontains=q) |
                Q(last_name__icontains=q)
            )
        staff_items = []
        for s in staff_qs[:10]:
            full_name = f"{s.last_name} {s.first_name}".strip() or s.username
            dept_label = s.department.name if s.department else "Toàn viện"
            role_label = "Bác sĩ" if s.role == 'DOCTOR' else ("Điều dưỡng" if s.role == 'NURSE' else "Quản trị")
            staff_items.append({
                'ref_type': 'STAFF',
                'id': s.id,
                'code': s.username,
                'title': f"Cán bộ y tế: {full_name} (@{s.username})",
                'subtitle': f"Chức vụ: {role_label} • {dept_label}",
                'doc_summary': f"Hồ sơ chuyên môn & Ca trực của {full_name}",
                'link': "/staff",
            })

        return Response({
            'patients': patient_items,
            'machines': machine_items,
            'staff': staff_items,
            'all': patient_items + machine_items + staff_items,
        }, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'])
    def conversations(self, request):
        user = request.user
        now = dj_timezone.now()
        from django.core.cache import cache
        from django.db.models import Q

        cache.set(f'user_online_{user.id}', True, timeout=300)
        CustomUser.objects.filter(pk=user.pk).update(last_seen=now)

        # 1. Pinned Channel
        dept_name = user.department.name if user.department else "Toàn Viện"
        channel_name = f"Kênh Chung - {dept_name}"

        if user.role == CustomUser.Role.ADMIN:
            last_broadcast = MedicalMessage.objects.filter(receiver__isnull=True).order_by('-created_at').first()
            broadcast_qs = MedicalMessage.objects.filter(receiver__isnull=True).exclude(sender=user)
        else:
            last_broadcast = MedicalMessage.objects.filter(
                Q(receiver__isnull=True) &
                (Q(department=user.department) | Q(department__isnull=True) | Q(sender__role='ADMIN'))
            ).order_by('-created_at').first()
            broadcast_qs = MedicalMessage.objects.filter(
                Q(receiver__isnull=True) &
                (Q(department=user.department) | Q(department__isnull=True) | Q(sender__role='ADMIN'))
            ).exclude(sender=user)

        # Calculate unread for pinned channel
        pinned_unread = 0
        for b_msg in broadcast_qs.order_by('-created_at')[:50]:
            if user.id not in (b_msg.read_by_users or []):
                pinned_unread += 1

        pinned_channel = {
            'id': 'department',
            'name': channel_name,
            'pinned': True,
            'unread_count': pinned_unread,
            'last_message': f"{last_broadcast.sender.get_full_name() or last_broadcast.sender.username}: {last_broadcast.content}" if last_broadcast else "Chưa có thông báo nào",
            'last_message_at': last_broadcast.created_at.isoformat() if last_broadcast else None,
        }

        # 2. Contacts / Conversations
        contacts = []
        all_users = CustomUser.objects.filter(is_active=True).exclude(pk=user.pk).select_related('department')

        for u in all_users:
            is_cached = bool(cache.get(f'user_online_{u.id}'))
            is_recent = bool(u.last_seen and (now - u.last_seen).total_seconds() <= 300)
            u_online = is_cached or is_recent

            status_text = 'Đang hoạt động' if u_online else format_status_text(u.last_seen, now=now)
            full_name = f"{u.last_name} {u.first_name}".strip() or u.username
            u_dept_name = u.department.name if u.department else ('Toàn viện' if u.role == CustomUser.Role.ADMIN else 'Chưa phân khoa')

            last_msg = MedicalMessage.objects.filter(
                (Q(sender=user, receiver=u)) |
                (Q(sender=u, receiver=user))
            ).order_by('-created_at').first()

            last_msg_content = last_msg.content if last_msg else None
            is_last_from_me = (last_msg.sender_id == user.id) if last_msg else False
            last_msg_at = last_msg.created_at.isoformat() if last_msg else None
            last_seen_iso = u.last_seen.isoformat() if u.last_seen else None
            last_seen_epoch = u.last_seen.timestamp() if u.last_seen else 0

            # Count unread direct messages from u to current user
            unread_count = MedicalMessage.objects.filter(sender=u, receiver=user, is_read=False).count()

            contacts.append({
                'id': u.id,
                'username': u.username,
                'full_name': full_name,
                'role': u.role,
                'department_id': u.department_id,
                'department_name': u_dept_name,
                'is_online': u_online,
                'last_seen': last_seen_iso,
                'last_seen_epoch': last_seen_epoch,
                'status_text': status_text,
                'unread_count': unread_count,
                'last_message': last_msg_content,
                'is_last_from_me': is_last_from_me,
                'last_message_at': last_msg_at,
            })

        # Dynamic Sorting: Priority 1 = last_message_at desc; Priority 2 = is_online, last_seen, name
        has_msg = sorted([c for c in contacts if c['last_message_at']], key=lambda x: x['last_message_at'], reverse=True)
        no_msg = sorted([c for c in contacts if not c['last_message_at']], key=lambda x: (not x['is_online'], -x['last_seen_epoch'], x['full_name'].lower()))

        ordered_conversations = has_msg + no_msg
        total_unread = sum(c['unread_count'] for c in ordered_conversations) + pinned_unread

        return Response({
            'pinned_channel': pinned_channel,
            'conversations': ordered_conversations,
            'contacts': ordered_conversations,
            'count_online': sum(1 for c in ordered_conversations if c['is_online']),
            'total_unread': total_unread,
        }, status=status.HTTP_200_OK)
