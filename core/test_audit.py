from datetime import date, datetime
from django.db import transaction
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase
from core.models import AuditLog, CustomUser, Department, DialysisMachine, DialysisSession, Patient, VitalSign


class AuditLogTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.dept = Department.objects.create(
            name='Khoa Loc Mau Audit',
            code='K-AUDIT',
            description='Khoa test audit',
        )
        cls.admin_user = CustomUser.objects.create_user(
            username='admin_audit',
            email='admin_audit@example.com',
            password='Password123!',
            role=CustomUser.Role.ADMIN,
            department=cls.dept,
        )
        cls.doctor_user = CustomUser.objects.create_user(
            username='doctor_audit',
            email='doctor_audit@example.com',
            password='Password123!',
            role=CustomUser.Role.DOCTOR,
            department=cls.dept,
        )
        cls.nurse_user = CustomUser.objects.create_user(
            username='nurse_audit',
            email='nurse_audit@example.com',
            password='Password123!',
            role=CustomUser.Role.NURSE,
            department=cls.dept,
        )

    def setUp(self):
        AuditLog.objects.all().delete()

    def test_01_admin_access_audit_logs_200(self):
        self.client.force_authenticate(user=self.admin_user)
        response = self.client.get('/api/audit-logs/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)

    def test_02_doctor_access_audit_logs_403(self):
        self.client.force_authenticate(user=self.doctor_user)
        response = self.client.get('/api/audit-logs/')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_03_nurse_access_audit_logs_403(self):
        self.client.force_authenticate(user=self.nurse_user)
        response = self.client.get('/api/audit-logs/')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_04_create_entity_creates_audit_log(self):
        self.client.force_authenticate(user=self.admin_user)
        data = {
            'patient_id': 'PT-AUD-01',
            'full_name': 'Nguyen Van Audit',
            'date_of_birth': '1985-05-15',
            'gender': 'MALE',
            'phone_number': '0901234567',
            'address': 'Hanoi',
            'status': 'ACTIVE',
        }
        response = self.client.post('/api/patients/', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        log = AuditLog.objects.filter(entity_type='Patient', entity_id='PT-AUD-01').first()
        self.assertIsNotNone(log)
        self.assertEqual(log.action, AuditLog.Action.CREATE)
        self.assertEqual(log.actor, self.admin_user)
        self.assertEqual(log.changes.get('full_name'), 'Nguyen Van Audit')

    def test_05_update_entity_captures_diff(self):
        self.client.force_authenticate(user=self.admin_user)
        patient = Patient.objects.create(
            patient_id='PT-AUD-02',
            full_name='Tran Van Cu',
            date_of_birth=date(1990, 1, 1),
            gender='MALE',
            status='ACTIVE',
        )

        response = self.client.patch(
            f'/api/patients/{patient.patient_id}/',
            {'full_name': 'Tran Van Moi', 'phone_number': '0988776655'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        log = AuditLog.objects.filter(
            entity_type='Patient',
            entity_id='PT-AUD-02',
            action=AuditLog.Action.UPDATE,
        ).first()
        self.assertIsNotNone(log)
        self.assertIn('full_name', log.changes)
        self.assertEqual(log.changes['full_name']['before'], 'Tran Van Cu')
        self.assertEqual(log.changes['full_name']['after'], 'Tran Van Moi')

    def test_06_delete_entity_captures_snapshot(self):
        self.client.force_authenticate(user=self.admin_user)
        patient = Patient.objects.create(
            patient_id='PT-AUD-03',
            full_name='Le Van Xoa',
            date_of_birth=date(1992, 2, 2),
            gender='MALE',
            status='ACTIVE',
        )

        response = self.client.delete(f'/api/patients/{patient.patient_id}/')
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        log = AuditLog.objects.filter(
            entity_type='Patient',
            entity_id='PT-AUD-03',
            action=AuditLog.Action.DELETE,
        ).first()
        self.assertIsNotNone(log)
        self.assertEqual(log.changes.get('full_name'), 'Le Van Xoa')

    def test_07_sensitive_fields_not_in_changes(self):
        self.client.force_authenticate(user=self.admin_user)
        user_data = {
            'username': 'user_secret_test',
            'email': 'secret@example.com',
            'password': 'SuperSecretPassword123!',
            'role': 'NURSE',
            'department': self.dept.id,
            'is_active': True,
        }
        response = self.client.post('/api/users/', user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        new_user_id = str(response.data['id'])

        log = AuditLog.objects.filter(entity_type='CustomUser', entity_id=new_user_id).first()
        self.assertIsNotNone(log)
        self.assertNotIn('password', log.changes)
        for key in log.changes.keys():
            self.assertNotIn('token', key.lower())
            self.assertNotIn('secret', key.lower())

    def test_08_transaction_failure_rolls_back_audit_log(self):
        self.client.force_authenticate(user=self.admin_user)
        dept_with_ref = Department.objects.create(name='Dept Protected', code='DEPT-PROT')
        CustomUser.objects.create_user(
            username='user_under_dept',
            email='dept@example.com',
            password='Password123!',
            role=CustomUser.Role.NURSE,
            department=dept_with_ref,
        )

        initial_log_count = AuditLog.objects.count()
        # Deleting dept_with_ref should raise ProtectedError (400)
        response = self.client.delete(f'/api/departments/{dept_with_ref.id}/')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        # AuditLog count should NOT increase because transaction rolled back
        self.assertEqual(AuditLog.objects.count(), initial_log_count)

    def test_09_pagination_works(self):
        self.client.force_authenticate(user=self.admin_user)
        # Create 15 audit logs
        logs = [
            AuditLog(
                actor=self.admin_user,
                action=AuditLog.Action.CREATE,
                entity_type='Patient',
                entity_id=f'PT-PAGE-{i:02d}',
                changes={'index': i},
            )
            for i in range(15)
        ]
        AuditLog.objects.bulk_create(logs)

        response = self.client.get('/api/audit-logs/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 15)
        self.assertEqual(len(response.data['results']), 10)
        self.assertIsNotNone(response.data['next'])

    def test_10_filters_work(self):
        self.client.force_authenticate(user=self.admin_user)
        AuditLog.objects.create(
            actor=self.admin_user,
            action=AuditLog.Action.CREATE,
            entity_type='Patient',
            entity_id='PT-FLT-01',
            changes={},
        )
        AuditLog.objects.create(
            actor=self.admin_user,
            action=AuditLog.Action.DELETE,
            entity_type='DialysisMachine',
            entity_id='M-FLT-01',
            changes={},
        )

        # Filter by action
        res_act = self.client.get('/api/audit-logs/?action=CREATE')
        self.assertEqual(res_act.status_code, status.HTTP_200_OK)
        self.assertEqual(res_act.data['count'], 1)
        self.assertEqual(res_act.data['results'][0]['action'], 'CREATE')

        # Filter by entity_type
        res_ent = self.client.get('/api/audit-logs/?entity_type=DialysisMachine')
        self.assertEqual(res_ent.status_code, status.HTTP_200_OK)
        self.assertEqual(res_ent.data['count'], 1)
        self.assertEqual(res_ent.data['results'][0]['entity_type'], 'DialysisMachine')
