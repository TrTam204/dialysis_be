"""
Unit and Integration Tests for AI Bridge (Django <-> FastAPI / Local PyTorch LSTM Engine).
Verifies:
- Insufficient data handling (0 vital signs)
- Short sequence padding (1-2 vital signs)
- Stable vital signs -> LOW risk prediction
- Dropping SBP vital signs -> HIGH risk alert prediction
- GET /api/ai/lstm-model-info/ endpoint structure and 5 required metrics
"""
from datetime import date, timedelta
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from core.ai_bridge import build_vitals_sequence, predict_idh_for_session
from core.models import (
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
    VitalSign,
)


class AIBridgeTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.dept = Department.objects.create(
            name='Khoa Loc Mau AI Bridge',
            code='K-AIB-01',
        )
        cls.doctor = CustomUser.objects.create_user(
            username='doctor_aib',
            email='doctor_aib@example.com',
            password='Password123!',
            role=CustomUser.Role.DOCTOR,
            department=cls.dept,
        )
        cls.nurse = CustomUser.objects.create_user(
            username='nurse_aib',
            email='nurse_aib@example.com',
            password='Password123!',
            role=CustomUser.Role.NURSE,
            department=cls.dept,
        )
        cls.machine = DialysisMachine.objects.create(
            machine_id='M-AIB-01',
            name='May Loc Mau AI 01',
            status=DialysisMachine.Status.AVAILABLE,
            department=cls.dept,
        )
        cls.patient = Patient.objects.create(
            patient_id='PT-AIB-01',
            full_name='Bệnh nhân Thử nghiệm AI',
            date_of_birth=date(1970, 1, 1),
            status=Patient.Status.ACTIVE,
            department=cls.dept,
        )

    def setUp(self):
        self.now = timezone.now()

    def test_01_session_without_vitals_returns_insufficient_data(self):
        """When session has 0 vital signs, API must return INSUFFICIENT_DATA without crashing."""
        session = DialysisSession.objects.create(
            session_id='SS-AIB-NO-VITALS',
            patient=self.patient,
            machine=self.machine,
            scheduled_start=self.now,
            scheduled_end=self.now + timedelta(hours=4),
            status=DialysisSession.Status.SCHEDULED,
        )

        self.client.force_authenticate(user=self.doctor)
        response = self.client.get(f'/api/sessions/{session.session_id}/idh-prediction/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['has_prediction'])
        self.assertEqual(response.data['status'], 'INSUFFICIENT_DATA')
        self.assertIn('Cần ít nhất 1 lần đo sinh hiệu', response.data['message'])

    def test_02_session_with_one_or_two_vitals_applies_left_padding(self):
        """When session has 1 or 2 vital signs, bridge applies left-padding to satisfy LSTM length."""
        session = DialysisSession.objects.create(
            session_id='SS-AIB-PADDED',
            patient=self.patient,
            machine=self.machine,
            scheduled_start=self.now,
            scheduled_end=self.now + timedelta(hours=4),
            status=DialysisSession.Status.IN_PROGRESS,
        )
        VitalSign.objects.create(
            session=session,
            recorded_by=self.doctor,
            recorded_at=self.now,
            systolic_bp=130,
            diastolic_bp=80,
            heart_rate=75,
        )

        self.client.force_authenticate(user=self.doctor)
        response = self.client.get(f'/api/sessions/{session.session_id}/idh-prediction/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['has_prediction'])
        self.assertTrue(response.data['is_padded'])
        self.assertEqual(response.data['recorded_vitals_count'], 1)
        self.assertIn(response.data['risk_level'], {'LOW', 'MEDIUM', 'HIGH'})

    def test_03_stable_vitals_trajectory_predicts_low_risk(self):
        """Stable vital signs throughout session must yield LOW risk probability."""
        session = DialysisSession.objects.create(
            session_id='SS-AIB-STABLE',
            patient=self.patient,
            machine=self.machine,
            scheduled_start=self.now - timedelta(hours=2),
            scheduled_end=self.now + timedelta(hours=2),
            actual_start=self.now - timedelta(hours=2),
            status=DialysisSession.Status.IN_PROGRESS,
            uf_target=2.0,
        )
        # Create 5 stable vital signs every 30 mins
        stable_readings = [
            (135, 80, 72),
            (132, 78, 74),
            (130, 78, 72),
            (128, 76, 75),
            (126, 75, 74),
        ]
        for idx, (sbp, dbp, hr) in enumerate(stable_readings):
            VitalSign.objects.create(
                session=session,
                recorded_by=self.doctor,
                recorded_at=self.now - timedelta(hours=2) + timedelta(minutes=idx * 30),
                systolic_bp=sbp,
                diastolic_bp=dbp,
                heart_rate=hr,
            )

        self.client.force_authenticate(user=self.doctor)
        response = self.client.get(f'/api/sessions/{session.session_id}/idh-prediction/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['has_prediction'])
        self.assertEqual(response.data['risk_level'], 'LOW')
        self.assertFalse(response.data['is_idh_predicted'])
        self.assertFalse(response.data['is_padded'])
        self.assertEqual(response.data['recorded_vitals_count'], 5)
        self.assertIn(response.data['inference_source'], {'FASTAPI_MICROSERVICE', 'LOCAL_LSTM_ENGINE'})

    def test_04_dropping_sbp_trajectory_triggers_high_risk_alert(self):
        """Sharply dropping SBP (e.g. 130 -> 110 -> 92 -> 84) must trigger HIGH risk alert."""
        session = DialysisSession.objects.create(
            session_id='SS-AIB-ALERT',
            patient=self.patient,
            machine=self.machine,
            scheduled_start=self.now - timedelta(hours=2),
            scheduled_end=self.now + timedelta(hours=2),
            actual_start=self.now - timedelta(hours=2),
            status=DialysisSession.Status.IN_PROGRESS,
            uf_target=4.5,
        )
        # Create dropping SBP readings
        dropping_readings = [
            (130, 80, 70),
            (118, 72, 74),
            (102, 65, 84),
            (90, 58, 92),
            (82, 50, 98),
        ]
        for idx, (sbp, dbp, hr) in enumerate(dropping_readings):
            VitalSign.objects.create(
                session=session,
                recorded_by=self.doctor,
                recorded_at=self.now - timedelta(hours=2) + timedelta(minutes=idx * 30),
                systolic_bp=sbp,
                diastolic_bp=dbp,
                heart_rate=hr,
            )

        self.client.force_authenticate(user=self.nurse)
        response = self.client.get(f'/api/sessions/{session.session_id}/idh-prediction/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['has_prediction'])
        self.assertEqual(response.data['risk_level'], 'HIGH')
        self.assertEqual(response.data.get('model_version'), '1.0.0')
        self.assertIn('predicted_at', response.data)
        self.assertIn('disclaimer', response.data)
        self.assertTrue(response.data['is_idh_predicted'])
        self.assertGreaterEqual(response.data['risk_probability'], 0.65)
        self.assertTrue(any('CẢNH BÁO CAO' in rec for rec in response.data['clinical_recommendations']))

    def test_05_lstm_model_info_endpoint(self):
        """GET /api/ai/lstm-model-info/ must return HTTP 200 with all 5 evaluation metrics."""
        self.client.force_authenticate(user=self.doctor)
        response = self.client.get('/api/ai/lstm-model-info/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.data

        self.assertEqual(data['model_name'], 'IDHLSTMNet')
        self.assertIn('metrics', data)
        self.assertIn('test', data['metrics'])

        test_metrics = data['metrics']['test']
        self.assertIn('accuracy', test_metrics)
        self.assertIn('precision', test_metrics)
        self.assertIn('recall', test_metrics)
        self.assertIn('f1_score', test_metrics)
        self.assertIn('loss', test_metrics)

        # Check values match saved evaluation metrics
        self.assertGreater(test_metrics['accuracy'], 0.90)
        self.assertGreater(test_metrics['f1_score'], 0.85)

        # Check training curves exist
        self.assertIn('training_history', data)
        self.assertGreater(len(data['training_history']), 0)
        first_ep = data['training_history'][0]
        self.assertIn('train_loss', first_ep)
        self.assertIn('train_accuracy', first_ep)

    def test_06_idh_prediction_doctor_cross_department_forbidden(self):
        other_dept = Department.objects.create(name='Khoa Ngoai B', code='K-NG-B')
        doctor_other = CustomUser.objects.create_user(
            username='doctor_other_dept',
            email='doc_other@example.com',
            password='Password123!',
            role=CustomUser.Role.DOCTOR,
            department=other_dept,
        )
        session = DialysisSession.objects.create(
            session_id='SS-AIB-RBAC-01',
            patient=self.patient,
            machine=self.machine,
            scheduled_start=self.now,
            scheduled_end=self.now + timedelta(hours=4),
            status=DialysisSession.Status.IN_PROGRESS,
        )
        self.client.force_authenticate(user=doctor_other)
        response = self.client.get(f'/api/sessions/{session.session_id}/idh-prediction/')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_07_idh_prediction_nurse_cross_department_forbidden(self):
        other_dept = Department.objects.create(name='Khoa Ngoai C', code='K-NG-C')
        nurse_other = CustomUser.objects.create_user(
            username='nurse_other_dept',
            email='nurse_other@example.com',
            password='Password123!',
            role=CustomUser.Role.NURSE,
            department=other_dept,
        )
        session = DialysisSession.objects.create(
            session_id='SS-AIB-RBAC-02',
            patient=self.patient,
            machine=self.machine,
            scheduled_start=self.now,
            scheduled_end=self.now + timedelta(hours=4),
            status=DialysisSession.Status.IN_PROGRESS,
        )
        self.client.force_authenticate(user=nurse_other)
        response = self.client.get(f'/api/sessions/{session.session_id}/idh-prediction/')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
