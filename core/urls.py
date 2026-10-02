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
    report_machine_utilization_view,
    report_operational_summary_view,
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
    path('reports/operational-summary/', report_operational_summary_view, name='report-operational-summary'),
    path('reports/machine-utilization/', report_machine_utilization_view, name='report-machine-utilization'),
    path('', include(router.urls)),
]
