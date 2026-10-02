from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    BloodSampleViewSet,
    CustomUserViewSet,
    DepartmentViewSet,
    DialysisMachineViewSet,
    DialysisSessionViewSet,
    PatientViewSet,
    VitalSignViewSet,
    dashboard_dialysis_stats_view,
    dashboard_machine_stats_view,
    dashboard_summary_view,
    BloodSampleReportView,
    MachineReportView,
    NurseWorkloadReportView,
    PatientReportView,
    SessionReportView,
)

router = DefaultRouter()
router.register(r'departments', DepartmentViewSet, basename='department')
router.register(r'users', CustomUserViewSet, basename='user')
router.register(r'patients', PatientViewSet, basename='patient')
router.register(r'machines', DialysisMachineViewSet, basename='machine')
router.register(r'sessions', DialysisSessionViewSet, basename='session')
router.register(r'blood-samples', BloodSampleViewSet, basename='bloodsample')
router.register(r'vital-signs', VitalSignViewSet, basename='vitalsign')

urlpatterns = [
    path('dashboard/summary/', dashboard_summary_view, name='dashboard-summary'),
    path('dashboard/dialysis-stats/', dashboard_dialysis_stats_view, name='dashboard-dialysis-stats'),
    path('dashboard/machine-stats/', dashboard_machine_stats_view, name='dashboard-machine-stats'),
    path('reports/sessions/', SessionReportView.as_view(), name='reports-sessions'),
    path('reports/patients/', PatientReportView.as_view(), name='reports-patients'),
    path('reports/machines/', MachineReportView.as_view(), name='reports-machines'),
    path('reports/nurse-workload/', NurseWorkloadReportView.as_view(), name='reports-nurse-workload'),
    path('reports/blood-samples/', BloodSampleReportView.as_view(), name='reports-blood-samples'),
    path('', include(router.urls)),
]
