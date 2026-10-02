from django.db.models import ProtectedError
from django.shortcuts import get_object_or_404
from django_filters import rest_framework as django_filters
from rest_framework import filters, permissions, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response

from .models import (
    BloodSample,
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
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
    BloodSampleSerializer,
    CustomUserSerializer,
    DepartmentSerializer,
    DialysisMachineSerializer,
    DialysisSessionSerializer,
    PatientSerializer,
    VitalSignSerializer,
)
from .services import dashboard_dialysis_stats, dashboard_machine_stats, dashboard_summary
from .reports import (
    BloodSampleReportView,
    MachineReportView,
    NurseWorkloadReportView,
    PatientReportView,
    SessionReportView,
)


class BaseRoleAwareViewSet(viewsets.ModelViewSet):
    filter_backends = [
        django_filters.DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        try:
            self.perform_destroy(instance)
        except ProtectedError:
            return Response(
                {'detail': 'Cannot delete this record because it is still referenced by other data.'},
                status=400,
            )
        return Response(status=204)


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
        serializer.save(session=session, recorded_by=request.user)
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
