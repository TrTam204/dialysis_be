from datetime import date, timedelta
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase
from core.models import CustomUser, Department, DialysisMachine, DialysisSession, Patient


class ReportsAggregationTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.dept = Department.objects.create(
            name='Test Department RPT',
            code='DEPT-RPT',
            description='Test department',
        )
        cls.admin_user = CustomUser.objects.create_user(
            username='test_rpt_admin',
            email='test_rpt_admin@example.com',
            password='password123',
            role=CustomUser.Role.ADMIN,
            department=cls.dept,
        )
        cls.nurse_user = CustomUser.objects.create_user(
            username='test_rpt_nurse',
            email='test_rpt_nurse@example.com',
            password='password123',
            role=CustomUser.Role.NURSE,
            department=cls.dept,
        )
        cls.doctor_user = CustomUser.objects.create_user(
            username='test_rpt_doctor',
            email='test_rpt_doctor@example.com',
            password='password123',
            role=CustomUser.Role.DOCTOR,
            department=cls.dept,
        )
        cls.non_staff_user = CustomUser.objects.create_user(
            username='test_rpt_guest',
            email='test_rpt_guest@example.com',
            password='password123',
            role='GUEST',
            department=cls.dept,
        )
        cls.machine1 = DialysisMachine.objects.create(
            machine_id='M-RPT-01',
            name='Machine RPT 01',
            status='AVAILABLE',
            department=cls.dept,
        )
        cls.machine2 = DialysisMachine.objects.create(
            machine_id='M-RPT-02',
            name='Machine RPT 02',
            status='IN_USE',
            department=cls.dept,
        )
        cls.patient = Patient.objects.create(
            patient_id='PT-RPT-01',
            full_name='Test Patient RPT',
            date_of_birth=date(1980, 1, 1),
            gender='MALE',
            dry_weight=60.0,
            status='IN_TREATMENT',
        )

        cls.today = timezone.localdate()
        dt_today = timezone.datetime(cls.today.year, cls.today.month, cls.today.day)

        # Completed session: 4 hours actual duration, uf_target 2.5, uf_actual 2.4
        cls.s1 = DialysisSession.objects.create(
            session_id='SS-RPT-01',
            patient=cls.patient,
            machine=cls.machine1,
            assigned_nurse=cls.nurse_user,
            scheduled_start=timezone.make_aware(dt_today.replace(hour=8, minute=0)),
            scheduled_end=timezone.make_aware(dt_today.replace(hour=12, minute=0)),
            status=DialysisSession.Status.COMPLETED,
            actual_start=timezone.make_aware(dt_today.replace(hour=8, minute=0)),
            actual_end=timezone.make_aware(dt_today.replace(hour=12, minute=0)),
            uf_target=2.5,
            uf_actual=2.4,
        )

        # Cancelled session: uf_target 2.0, uf_actual 0.0
        cls.s2 = DialysisSession.objects.create(
            session_id='SS-RPT-02',
            patient=cls.patient,
            machine=cls.machine1,
            assigned_nurse=cls.nurse_user,
            scheduled_start=timezone.make_aware(dt_today.replace(hour=13, minute=0)),
            scheduled_end=timezone.make_aware(dt_today.replace(hour=17, minute=0)),
            status=DialysisSession.Status.CANCELLED,
            uf_target=2.0,
            uf_actual=0.0,
        )

    def test_operational_summary_unauthenticated(self):
        """Unauthenticated requests must be rejected with 401."""
        response = self.client.get('/api/reports/operational-summary/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_operational_summary_authenticated_default(self):
        """Authenticated staff can fetch summary with default date range."""
        self.client.force_authenticate(user=self.admin_user)
        response = self.client.get('/api/reports/operational-summary/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn('date_from', data)
        self.assertIn('date_to', data)
        self.assertIn('total_sessions', data)
        self.assertIn('completed_sessions', data)
        self.assertIn('cancelled_sessions', data)
        self.assertIn('completion_rate', data)
        self.assertIn('total_uf_target', data)
        self.assertIn('total_uf_actual', data)
        self.assertIn('daily_trends', data)

    def test_operational_summary_metrics_accuracy(self):
        """Verify metric calculation accuracy within the date range of the test sessions."""
        self.client.force_authenticate(user=self.admin_user)
        today_str = self.today.isoformat()
        response = self.client.get(f'/api/reports/operational-summary/?date_from={today_str}&date_to={today_str}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        # In this date (today), there are at least our 2 test sessions
        self.assertGreaterEqual(data['total_sessions'], 2)
        self.assertGreaterEqual(data['completed_sessions'], 1)
        self.assertGreaterEqual(data['cancelled_sessions'], 1)
        # Completion rate must be between 0 and 100
        self.assertGreater(data['completion_rate'], 0)
        self.assertLessEqual(data['completion_rate'], 100)
        # UF sum check
        self.assertGreaterEqual(data['total_uf_target'], 4.5)
        self.assertGreaterEqual(data['total_uf_actual'], 2.4)
        # Daily trends length is exactly 1 day
        self.assertEqual(len(data['daily_trends']), 1)
        self.assertEqual(data['daily_trends'][0]['date'], today_str)

    def test_operational_summary_invalid_date_from_greater_than_date_to(self):
        """date_from > date_to must return HTTP 400."""
        self.client.force_authenticate(user=self.admin_user)
        response = self.client.get('/api/reports/operational-summary/?date_from=2026-10-10&date_to=2026-10-01')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('date_from cannot be greater than date_to', response.json().get('detail', ''))

    def test_operational_summary_invalid_date_format(self):
        """Invalid date format must return HTTP 400."""
        self.client.force_authenticate(user=self.admin_user)
        response = self.client.get('/api/reports/operational-summary/?date_from=invalid-date')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Invalid date format', response.json().get('detail', ''))

        response2 = self.client.get('/api/reports/operational-summary/?date_to=2026-99-99')
        self.assertEqual(response2.status_code, status.HTTP_400_BAD_REQUEST)

    def test_operational_summary_empty_dataset(self):
        """Date range with no sessions returns 0/empty values without error."""
        self.client.force_authenticate(user=self.admin_user)
        response = self.client.get('/api/reports/operational-summary/?date_from=2015-01-01&date_to=2015-01-03')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertEqual(data['total_sessions'], 0)
        self.assertEqual(data['completed_sessions'], 0)
        self.assertEqual(data['cancelled_sessions'], 0)
        self.assertEqual(data['completion_rate'], 0.0)
        self.assertEqual(data['total_uf_target'], 0.0)
        self.assertEqual(data['total_uf_actual'], 0.0)
        self.assertEqual(len(data['daily_trends']), 3)

    def test_machine_utilization_unauthenticated(self):
        """Unauthenticated requests must be rejected with 401."""
        response = self.client.get('/api/reports/machine-utilization/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_machine_utilization_metrics_accuracy(self):
        """Verify machine metrics: session count, completed count, runtime hours."""
        self.client.force_authenticate(user=self.nurse_user)
        today_str = self.today.isoformat()
        response = self.client.get(f'/api/reports/machine-utilization/?date_from={today_str}&date_to={today_str}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn('machines', data)
        self.assertIn('limitation_note', data)
        self.assertIn('total_machines', data)
        self.assertIn('total_completed_sessions', data)
        self.assertIn('total_runtime_hours', data)

        # Find M-RPT-01 in list
        m1 = next((m for m in data['machines'] if m['machine_id'] == 'M-RPT-01'), None)
        self.assertIsNotNone(m1)
        self.assertEqual(m1['session_count'], 2)
        self.assertEqual(m1['completed_count'], 1)
        self.assertEqual(m1['cancelled_count'], 1)
        self.assertEqual(m1['actual_runtime_hours'], 4.0)
        self.assertEqual(m1['actual_runtime_minutes'], 240.0)

        # Machine 2 has 0 sessions today
        m2 = next((m for m in data['machines'] if m['machine_id'] == 'M-RPT-02'), None)
        self.assertIsNotNone(m2)
        self.assertEqual(m2['session_count'], 0)
        self.assertEqual(m2['completed_count'], 0)
        self.assertEqual(m2['actual_runtime_hours'], 0.0)

    def test_machine_utilization_invalid_date_range(self):
        """date_from > date_to returns 400."""
        self.client.force_authenticate(user=self.admin_user)
        response = self.client.get('/api/reports/machine-utilization/?date_from=2026-12-31&date_to=2026-01-01')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_dashboard_dialysis_stats_backward_compatibility(self):
        """Existing /api/dashboard/dialysis-stats/ keeps exact contract and supports days=N."""
        self.client.force_authenticate(user=self.nurse_user)

        # Default 7 days
        res7 = self.client.get('/api/dashboard/dialysis-stats/')
        self.assertEqual(res7.status_code, status.HTTP_200_OK)
        data7 = res7.json()
        self.assertEqual(len(data7), 7)
        self.assertIn('date', data7[0])
        self.assertIn('count', data7[0])

        # Custom days=14
        res14 = self.client.get('/api/dashboard/dialysis-stats/?days=14')
        self.assertEqual(res14.status_code, status.HTTP_200_OK)
        data14 = res14.json()
        self.assertEqual(len(data14), 14)

        # Date range support
        today_str = self.today.isoformat()
        res_range = self.client.get(f'/api/dashboard/dialysis-stats/?date_from={today_str}&date_to={today_str}')
        self.assertEqual(res_range.status_code, status.HTTP_200_OK)
        data_range = res_range.json()
        self.assertEqual(len(data_range), 1)
        self.assertEqual(data_range[0]['date'], today_str)

    def test_rbac_doctor_access_allowed(self):
        """DOCTOR role can access operational summary and machine utilization reports."""
        self.client.force_authenticate(user=self.doctor_user)
        res_op = self.client.get('/api/reports/operational-summary/')
        self.assertEqual(res_op.status_code, status.HTTP_200_OK)

        res_mach = self.client.get('/api/reports/machine-utilization/')
        self.assertEqual(res_mach.status_code, status.HTTP_200_OK)

    def test_rbac_non_staff_access_forbidden(self):
        """Non-staff role is rejected with 403 Forbidden."""
        self.client.force_authenticate(user=self.non_staff_user)
        res_op = self.client.get('/api/reports/operational-summary/')
        self.assertEqual(res_op.status_code, status.HTTP_403_FORBIDDEN)

        res_mach = self.client.get('/api/reports/machine-utilization/')
        self.assertEqual(res_mach.status_code, status.HTTP_403_FORBIDDEN)
