from datetime import date
from django.db.models import ProtectedError
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import CustomUser, Department, Patient


class PatientDepartmentLinkTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.dept1 = Department.objects.create(
            name='Khoa Loc Mau 1',
            code='KLM1',
            description='Khoa 1 description',
        )
        cls.dept2 = Department.objects.create(
            name='Khoa Loc Mau 2',
            code='KLM2',
            description='Khoa 2 description',
        )
        cls.admin_user = CustomUser.objects.create_user(
            username='admin_dept_test',
            email='admin_dept_test@example.com',
            password='Password123!',
            role=CustomUser.Role.ADMIN,
            department=cls.dept1,
        )
        cls.doctor_user = CustomUser.objects.create_user(
            username='doctor_dept_test',
            email='doctor_dept_test@example.com',
            password='Password123!',
            role=CustomUser.Role.DOCTOR,
            department=cls.dept1,
        )
        cls.nurse_user = CustomUser.objects.create_user(
            username='nurse_dept_test',
            email='nurse_dept_test@example.com',
            password='Password123!',
            role=CustomUser.Role.NURSE,
            department=cls.dept1,
        )

    def test_patient_model_department_nullable(self):
        """
        Patient can be created without a department (null=True).
        """
        patient = Patient.objects.create(
            patient_id='P_NODPT_01',
            full_name='Nguyen Van Khong Khoa',
            date_of_birth=date(1990, 1, 1),
        )
        self.assertIsNone(patient.department)
        patient_refreshed = Patient.objects.get(patient_id='P_NODPT_01')
        self.assertIsNone(patient_refreshed.department)

    def test_patient_model_department_assigned(self):
        """
        Patient can be assigned to a department and accessed via reverse relation.
        """
        patient = Patient.objects.create(
            patient_id='P_DPT_01',
            full_name='Tran Thi Co Khoa',
            date_of_birth=date(1985, 5, 15),
            department=self.dept1,
        )
        self.assertEqual(patient.department, self.dept1)
        self.assertTrue(self.dept1.patients.filter(patient_id='P_DPT_01').exists())

    def test_department_protected_from_deletion_when_referenced_by_patient(self):
        """
        Department cannot be deleted if a patient is linked to it (on_delete=models.PROTECT).
        """
        Patient.objects.create(
            patient_id='P_PROTECT_01',
            full_name='Le Van Bao Ve',
            date_of_birth=date(1975, 10, 20),
            department=self.dept1,
        )
        with self.assertRaises(ProtectedError):
            self.dept1.delete()

    def test_api_create_patient_without_department_backward_compatibility(self):
        """
        Legacy API requests without department field continue to work smoothly.
        """
        self.client.force_authenticate(user=self.admin_user)
        url = reverse('patient-list')
        payload = {
            'patient_id': 'P_API_LEGACY',
            'full_name': 'Benh Nhan Cu Khong Khoa',
            'date_of_birth': '1992-03-10',
            'gender': 'MALE',
            'status': 'ACTIVE',
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertIsNone(res.data.get('department'))
        self.assertIsNone(res.data.get('department_name'))

        p = Patient.objects.get(patient_id='P_API_LEGACY')
        self.assertIsNone(p.department)

    def test_api_create_patient_with_department(self):
        """
        API allows assigning a department when creating a patient.
        """
        self.client.force_authenticate(user=self.doctor_user)
        url = reverse('patient-list')
        payload = {
            'patient_id': 'P_API_DEPT',
            'full_name': 'Benh Nhan Khoa 1',
            'date_of_birth': '1988-07-22',
            'gender': 'FEMALE',
            'status': 'ACTIVE',
            'department': self.dept1.id,
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertEqual(res.data.get('department'), self.dept1.id)
        self.assertEqual(res.data.get('department_name'), self.dept1.name)

        p = Patient.objects.get(patient_id='P_API_DEPT')
        self.assertEqual(p.department, self.dept1)

    def test_nurse_cannot_create_patient(self):
        """
        NURSE is NOT allowed to create patients via POST (forbidden by IsAdminOrDoctor).
        """
        self.client.force_authenticate(user=self.nurse_user)
        url = reverse('patient-list')
        payload = {
            'patient_id': 'P_NURSE_CREATE',
            'full_name': 'Benh Nhan Nurse Tao',
            'date_of_birth': '1990-01-01',
            'gender': 'MALE',
            'status': 'ACTIVE',
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_403_FORBIDDEN)
        self.assertFalse(Patient.objects.filter(patient_id='P_NURSE_CREATE').exists())

    def test_doctor_cannot_create_patient_in_another_department(self):
        """
        Reproduction test:
        DOCTOR in Dept 1 must NOT be allowed to create a patient in Dept 2.
        """
        self.client.force_authenticate(user=self.doctor_user)
        url = reverse('patient-list')
        payload = {
            'patient_id': 'P_CROSS_DEPT',
            'full_name': 'Benh Nhan Tao Cheo Khoa',
            'date_of_birth': '1990-01-01',
            'gender': 'MALE',
            'status': 'ACTIVE',
            'department': self.dept2.id,
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Bác sĩ chỉ được tạo bệnh nhân thuộc khoa của mình', str(res.data.get('department', '')))
        self.assertFalse(Patient.objects.filter(patient_id='P_CROSS_DEPT').exists())

    def test_doctor_creates_patient_without_department_auto_assigns_doctor_department(self):
        """
        Reproduction test:
        When DOCTOR creates a patient without specifying a department,
        system must automatically assign DOCTOR's department.
        """
        self.client.force_authenticate(user=self.doctor_user)
        url = reverse('patient-list')
        payload = {
            'patient_id': 'P_DOC_AUTO',
            'full_name': 'Benh Nhan Tu Dong Gan Khoa',
            'date_of_birth': '1995-06-15',
            'gender': 'MALE',
            'status': 'ACTIVE',
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertEqual(res.data.get('department'), self.dept1.id)
        self.assertEqual(res.data.get('department_name'), self.dept1.name)

        p = Patient.objects.get(patient_id='P_DOC_AUTO')
        self.assertEqual(p.department, self.dept1)

    def test_admin_cannot_create_patient_in_inactive_department(self):
        """
        Reproduction test:
        ADMIN cannot assign an inactive department when creating a patient.
        """
        dept_inactive = Department.objects.create(
            name='Khoa Ngung Hoat Dong',
            code='KNHD',
            is_active=False,
        )
        self.client.force_authenticate(user=self.admin_user)
        url = reverse('patient-list')
        payload = {
            'patient_id': 'P_INACTIVE_DEPT',
            'full_name': 'Benh Nhan Khoa Dong Cua',
            'date_of_birth': '1985-02-20',
            'gender': 'FEMALE',
            'status': 'ACTIVE',
            'department': dept_inactive.id,
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Khoa đã ngừng hoạt động', str(res.data.get('department', '')))
        self.assertFalse(Patient.objects.filter(patient_id='P_INACTIVE_DEPT').exists())


    def test_api_admin_can_assign_initial_department_when_null(self):
        """
        ADMIN is allowed to assign initial department to a patient that currently has department=None.
        """
        p = Patient.objects.create(
            patient_id='P_ADMIN_INIT',
            full_name='Benh Nhan Chua Co Khoa',
            date_of_birth=date(1982, 11, 5),
            department=None,
        )
        self.client.force_authenticate(user=self.admin_user)
        url = reverse('patient-detail', kwargs={'pk': p.patient_id})
        res = self.client.patch(url, {'department': self.dept1.id})
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data.get('department'), self.dept1.id)
        self.assertEqual(res.data.get('department_name'), self.dept1.name)

        p.refresh_from_db()
        self.assertEqual(p.department, self.dept1)

    def test_api_doctor_cannot_update_department(self):
        """
        DOCTOR is NOT allowed to assign or change patient department via PATCH.
        """
        p = Patient.objects.create(
            patient_id='P_DOC_BLOCKED',
            full_name='Benh Nhan Thu Nghiem Doctor',
            date_of_birth=date(1982, 11, 5),
            department=None,
        )
        self.client.force_authenticate(user=self.doctor_user)
        url = reverse('patient-detail', kwargs={'pk': p.patient_id})
        res = self.client.patch(url, {'department': self.dept1.id})
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Chỉ ADMIN mới có quyền gán khoa cho bệnh nhân', str(res.data.get('department', '')))

        p.refresh_from_db()
        self.assertIsNone(p.department)

    def test_api_nurse_cannot_update_department(self):
        """
        NURSE is NOT allowed to change patient department via PATCH (forbidden by IsNursePatientStatusOnly).
        """
        p = Patient.objects.create(
            patient_id='P_NURSE_BLOCKED',
            full_name='Benh Nhan Thu Nghiem Nurse',
            date_of_birth=date(1982, 11, 5),
            department=None,
        )
        self.client.force_authenticate(user=self.nurse_user)
        url = reverse('patient-detail', kwargs={'pk': p.patient_id})
        res = self.client.patch(url, {'department': self.dept1.id})
        self.assertEqual(res.status_code, status.HTTP_403_FORBIDDEN)

        p.refresh_from_db()
        self.assertIsNone(p.department)

    def test_api_admin_cannot_change_assigned_department_via_patch(self):
        """
        ADMIN cannot change an already assigned department via regular PATCH.
        Formal department transfer must go through the dedicated transfer workflow.
        """
        p = Patient.objects.create(
            patient_id='P_ADM_NO_TRF',
            full_name='Benh Nhan Da Co Khoa 1',
            date_of_birth=date(1982, 11, 5),
            department=self.dept1,
        )
        self.client.force_authenticate(user=self.admin_user)
        url = reverse('patient-detail', kwargs={'pk': p.patient_id})
        res = self.client.patch(url, {'department': self.dept2.id})
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Không thể thay đổi khoa đã gán qua cập nhật thông thường', str(res.data.get('department', '')))

        p.refresh_from_db()
        self.assertEqual(p.department, self.dept1)

    def test_api_doctor_can_patch_other_patient_fields(self):
        """
        DOCTOR can still normally update non-department fields (e.g. phone, dry_weight, medical_history).
        """
        p = Patient.objects.create(
            patient_id='P_DOC_NORMAL_PATCH',
            full_name='Ten Cu',
            phone_number='0123456789',
            date_of_birth=date(1982, 11, 5),
            dry_weight=55.0,
            department=self.dept1,
        )
        self.client.force_authenticate(user=self.doctor_user)
        url = reverse('patient-detail', kwargs={'pk': p.patient_id})
        res = self.client.patch(url, {
            'phone_number': '0988776655',
            'dry_weight': 56.5,
        })
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        p.refresh_from_db()
        self.assertEqual(p.phone_number, '0988776655')
        self.assertEqual(p.dry_weight, 56.5)
        self.assertEqual(p.department, self.dept1)

    def test_api_filter_patients_by_department(self):
        """
        Patients can be filtered by department query parameter.
        """
        Patient.objects.create(patient_id='P_F1', full_name='Patient D1', date_of_birth='1990-01-01', department=self.dept1)
        Patient.objects.create(patient_id='P_F2', full_name='Patient D2', date_of_birth='1990-01-01', department=self.dept2)
        Patient.objects.create(patient_id='P_F0', full_name='Patient NoDept', date_of_birth='1990-01-01', department=None)

        self.client.force_authenticate(user=self.doctor_user)
        url = reverse('patient-list')

        # Filter dept1
        res1 = self.client.get(url, {'department': self.dept1.id})
        self.assertEqual(res1.status_code, status.HTTP_200_OK)
        patient_ids_1 = [item['patient_id'] for item in res1.data.get('results', res1.data)]
        self.assertIn('P_F1', patient_ids_1)
        self.assertNotIn('P_F2', patient_ids_1)
        self.assertNotIn('P_F0', patient_ids_1)

        # Filter dept2
        res2 = self.client.get(url, {'department': self.dept2.id})
        self.assertEqual(res2.status_code, status.HTTP_200_OK)
        patient_ids_2 = [item['patient_id'] for item in res2.data.get('results', res2.data)]
        self.assertIn('P_F2', patient_ids_2)
        self.assertNotIn('P_F1', patient_ids_2)
        self.assertNotIn('P_F0', patient_ids_2)

    def test_api_delete_department_protected_when_has_patient(self):
        """
        Attempting to delete a department with patients via API returns 400 Bad Request.
        """
        Patient.objects.create(
            patient_id='P_DEL_PROT',
            full_name='Patient Linked to Dept',
            date_of_birth=date(1990, 1, 1),
            department=self.dept1,
        )
        self.client.force_authenticate(user=self.admin_user)
        url = reverse('department-detail', kwargs={'pk': self.dept1.id})
        res = self.client.delete(url)
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Cannot delete this record', res.data.get('detail', ''))
        self.assertTrue(Department.objects.filter(id=self.dept1.id).exists())

    def test_doctor_with_inactive_department_cannot_create_patient(self):
        """
        DOCTOR whose department is inactive cannot create patients.
        """
        dept_inactive = Department.objects.create(
            name='Khoa Inactive Doctor',
            code='KID_DOC',
            is_active=False,
        )
        doc_inactive = CustomUser.objects.create_user(
            username='doc_inactive_user',
            email='doc_inactive@example.com',
            password='Password123!',
            role=CustomUser.Role.DOCTOR,
            department=dept_inactive,
        )
        self.client.force_authenticate(user=doc_inactive)
        url = reverse('patient-list')
        payload = {
            'patient_id': 'P_DOC_INACT_NEW',
            'full_name': 'Benh Nhan Bac Si Khoa Dong',
            'date_of_birth': '1990-01-01',
            'gender': 'MALE',
            'status': 'ACTIVE',
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('không thuộc khoa nào đang hoạt động', str(res.data.get('department', '')))
        self.assertFalse(Patient.objects.filter(patient_id='P_DOC_INACT_NEW').exists())

