from django.db import transaction
from django.db.models import ProtectedError
from django.shortcuts import get_object_or_404
from django_filters import rest_framework as django_filters
from rest_framework import filters, permissions, viewsets
from rest_framework.decorators import action, api_view, permission_classes
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
)
from .services import (
    dashboard_dialysis_stats,
    dashboard_machine_stats,
    dashboard_summary,
    parse_date_range,
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
    queryset = Patient.objects.all().order_by('patient_id')
    serializer_class = PatientSerializer
    filterset_fields = ['status', 'gender', 'date_of_birth']
    search_fields = ['patient_id', 'full_name', 'phone_number']
    ordering_fields = ['patient_id', 'full_name', 'status', 'date_of_birth', 'created_at']
    ordering = ['patient_id']

    def get_permissions(self):
        if self.action == 'create':
            self.permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]
        elif self.action == 'update':
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
    queryset = AuditLog.objects.select_related('actor').all().order_by('-timestamp')
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

    def get_permissions(self):
        if self.action in {'create', 'update', 'partial_update'}:
            self.permission_classes = [permissions.IsAuthenticated, IsAdminOrDoctor]
        elif self.action == 'destroy':
            self.permission_classes = [permissions.IsAuthenticated, IsAdmin]
        else:
            self.permission_classes = [permissions.IsAuthenticated, IsStaffMember]
        return super().get_permissions()
