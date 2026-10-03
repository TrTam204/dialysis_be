from datetime import date, datetime, timedelta
from django.db import IntegrityError, transaction
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase
from core.models import (
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
    ScheduleAssignment,
    SchedulePlan,
    Shift,
    TreatmentPattern,
)


class SchedulingDomainTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.dept = Department.objects.create(
            name='Khoa Than Loc Mau M9',
            code='K-M9',
            description='Khoa test M9 domain',
        )
        cls.admin_user = CustomUser.objects.create_user(
            username='admin_m9',
            email='admin_m9@example.com',
            password='Password123!',
            role=CustomUser.Role.ADMIN,
            department=cls.dept,
        )
        cls.doctor_user = CustomUser.objects.create_user(
            username='doctor_m9',
            email='doctor_m9@example.com',
            password='Password123!',
            role=CustomUser.Role.DOCTOR,
            department=cls.dept,
        )
        cls.nurse_user = CustomUser.objects.create_user(
            username='nurse_m9',
            email='nurse_m9@example.com',
            password='Password123!',
            role=CustomUser.Role.NURSE,
            department=cls.dept,
        )
        cls.machine1 = DialysisMachine.objects.create(
            machine_id='M-M9-01',
            name='Machine M9 01',
            status='AVAILABLE',
            department=cls.dept,
        )
        cls.machine2 = DialysisMachine.objects.create(
            machine_id='M-M9-02',
            name='Machine M9 02',
            status='AVAILABLE',
            department=cls.dept,
        )

    def test_01_create_patient_with_preferred_shift_and_treatment_pattern(self):
        self.client.force_authenticate(user=self.doctor_user)
        data = {
            'patient_id': 'PT-M9-01',
            'full_name': 'Nguyen Van M9',
            'date_of_birth': '1980-01-01',
            'gender': 'MALE',
            'status': 'ACTIVE',
            'preferred_shift': Shift.SHIFT_1,
            'treatment_pattern': TreatmentPattern.T2_T4_T6,
        }
        response = self.client.post('/api/patients/', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['preferred_shift'], 'SHIFT_1')
        self.assertEqual(response.data['treatment_pattern'], 'T2_T4_T6')

        patient = Patient.objects.get(patient_id='PT-M9-01')
        self.assertEqual(patient.preferred_shift, Shift.SHIFT_1)
        self.assertEqual(patient.treatment_pattern, TreatmentPattern.T2_T4_T6)

    def test_02_create_patient_with_null_treatment_pattern_and_shift(self):
        self.client.force_authenticate(user=self.doctor_user)
        data = {
            'patient_id': 'PT-M9-02',
            'full_name': 'Tran Van Null Pattern',
            'date_of_birth': '1985-05-05',
            'gender': 'FEMALE',
            'status': 'ACTIVE',
        }
        response = self.client.post('/api/patients/', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIsNone(response.data['preferred_shift'])
        self.assertIsNone(response.data['treatment_pattern'])

    def test_03_dialysis_session_assigned_nurse_can_be_null(self):
        self.client.force_authenticate(user=self.doctor_user)
        patient = Patient.objects.create(
            patient_id='PT-M9-03',
            full_name='Le Van NurseNull',
            date_of_birth=date(1990, 1, 1),
            gender='MALE',
            status='ACTIVE',
        )
        now = timezone.now()
        data = {
            'session_id': 'SS-M9-NULLNURSE',
            'patient': patient.patient_id,
            'machine': self.machine1.machine_id,
            'assigned_nurse': None,
            'scheduled_start': now.isoformat(),
            'scheduled_end': (now + timedelta(hours=4)).isoformat(),
            'status': 'SCHEDULED',
        }
        response = self.client.post('/api/sessions/', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIsNone(response.data['assigned_nurse'])
        self.assertIsNone(response.data['nurse_name'])

        session = DialysisSession.objects.get(session_id='SS-M9-NULLNURSE')
        self.assertIsNone(session.assigned_nurse)

    def test_04_create_schedule_plan(self):
        self.client.force_authenticate(user=self.admin_user)
        week_start = date(2026, 10, 12)
        week_end = date(2026, 10, 18)
        data = {
            'name': 'Lịch Tuần 42/2026',
            'department': self.dept.id,
            'week_start': week_start.isoformat(),
            'week_end': week_end.isoformat(),
            'status': 'PROPOSED',
            'fitness_score': 95.5,
            'algorithm_metadata': {'generations': 150, 'runtime_ms': 250},
        }
        response = self.client.post('/api/schedule-plans/', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['status'], 'PROPOSED')
        self.assertEqual(response.data['department'], self.dept.id)

        plan = SchedulePlan.objects.get(id=response.data['id'])
        self.assertEqual(plan.created_by, self.admin_user)

    def test_05_create_schedule_assignment_with_ga_source(self):
        self.client.force_authenticate(user=self.admin_user)
        plan = SchedulePlan.objects.create(
            name='Plan Test 05',
            department=self.dept,
            week_start=date(2026, 10, 12),
            week_end=date(2026, 10, 18),
            status=SchedulePlan.Status.PROPOSED,
        )
        patient = Patient.objects.create(
            patient_id='PT-M9-05',
            full_name='Pham Van Assignment',
            date_of_birth=date(1992, 2, 2),
            gender='MALE',
            status='ACTIVE',
            treatment_pattern=TreatmentPattern.T2_T4_T6,
        )
        scheduled_date = date(2026, 10, 12)
        start_dt = timezone.make_aware(datetime(2026, 10, 12, 7, 0, 0))
        end_dt = timezone.make_aware(datetime(2026, 10, 12, 11, 0, 0))

        data = {
            'schedule_plan': plan.id,
            'patient': patient.patient_id,
            'machine': self.machine1.machine_id,
            'scheduled_date': scheduled_date.isoformat(),
            'shift': Shift.SHIFT_1,
            'start_datetime': start_dt.isoformat(),
            'end_datetime': end_dt.isoformat(),
            'source': 'GA',
        }
        response = self.client.post('/api/schedule-assignments/', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['source'], 'GA')
        self.assertEqual(response.data['original_machine'], self.machine1.machine_id)
        self.assertEqual(response.data['original_shift'], 'SHIFT_1')

        asgn = ScheduleAssignment.objects.get(id=response.data['id'])
        self.assertEqual(asgn.original_machine, self.machine1)
        self.assertEqual(asgn.original_shift, Shift.SHIFT_1)

    def test_06_manual_override_switches_source_and_preserves_original(self):
        self.client.force_authenticate(user=self.admin_user)
        plan = SchedulePlan.objects.create(
            name='Plan Test 06',
            department=self.dept,
            week_start=date(2026, 10, 12),
            week_end=date(2026, 10, 18),
        )
        patient = Patient.objects.create(
            patient_id='PT-M9-06',
            full_name='Vu Van Override',
            date_of_birth=date(1993, 3, 3),
            status='ACTIVE',
        )
        scheduled_date = date(2026, 10, 12)
        asgn = ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=patient,
            machine=self.machine1,
            scheduled_date=scheduled_date,
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 12, 7, 0, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 12, 11, 0, 0)),
            source=ScheduleAssignment.Source.GA,
            original_machine=self.machine1,
            original_shift=Shift.SHIFT_1,
        )

        # Admin overrides machine from machine1 to machine2
        patch_data = {
            'machine': self.machine2.machine_id,
            'shift': Shift.SHIFT_2,
        }
        response = self.client.patch(f'/api/schedule-assignments/{asgn.id}/', patch_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['source'], 'MANUAL')
        self.assertEqual(response.data['machine'], self.machine2.machine_id)
        self.assertEqual(response.data['shift'], 'SHIFT_2')
        # original fields remain IMMUTABLE
        self.assertEqual(response.data['original_machine'], self.machine1.machine_id)
        self.assertEqual(response.data['original_shift'], 'SHIFT_1')

    def test_07_duplicate_machine_date_shift_in_same_plan_rejected(self):
        plan = SchedulePlan.objects.create(
            name='Plan Dup Machine',
            department=self.dept,
            week_start=date(2026, 10, 12),
            week_end=date(2026, 10, 18),
        )
        p1 = Patient.objects.create(patient_id='PT-M9-D1', full_name='Patient D1', date_of_birth=date(1980, 1, 1))
        p2 = Patient.objects.create(patient_id='PT-M9-D2', full_name='Patient D2', date_of_birth=date(1982, 2, 2))
        s_date = date(2026, 10, 12)
        start_dt = timezone.make_aware(datetime(2026, 10, 12, 7, 0, 0))
        end_dt = timezone.make_aware(datetime(2026, 10, 12, 11, 0, 0))

        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=p1,
            machine=self.machine1,
            scheduled_date=s_date,
            shift=Shift.SHIFT_1,
            start_datetime=start_dt,
            end_datetime=end_dt,
        )

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ScheduleAssignment.objects.create(
                    schedule_plan=plan,
                    patient=p2,
                    machine=self.machine1,  # Duplicate machine on same plan, date, shift
                    scheduled_date=s_date,
                    shift=Shift.SHIFT_1,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                )

    def test_08_duplicate_patient_date_in_same_plan_rejected(self):
        plan = SchedulePlan.objects.create(
            name='Plan Dup Patient',
            department=self.dept,
            week_start=date(2026, 10, 12),
            week_end=date(2026, 10, 18),
        )
        p = Patient.objects.create(patient_id='PT-M9-D3', full_name='Patient D3', date_of_birth=date(1980, 1, 1))
        s_date = date(2026, 10, 12)

        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=p,
            machine=self.machine1,
            scheduled_date=s_date,
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 12, 7, 0, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 12, 11, 0, 0)),
        )

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ScheduleAssignment.objects.create(
                    schedule_plan=plan,
                    patient=p,  # Duplicate patient on same date
                    machine=self.machine2,
                    scheduled_date=s_date,
                    shift=Shift.SHIFT_2,
                    start_datetime=timezone.make_aware(datetime(2026, 10, 12, 12, 30, 0)),
                    end_datetime=timezone.make_aware(datetime(2026, 10, 12, 16, 30, 0)),
                )

    def test_09_unique_approved_plan_per_department_week_rejected(self):
        week_start = date(2026, 10, 19)
        week_end = date(2026, 10, 25)

        # First approved plan
        SchedulePlan.objects.create(
            name='Plan Approved 1',
            department=self.dept,
            week_start=week_start,
            week_end=week_end,
            status=SchedulePlan.Status.APPROVED,
        )

        # Second approved plan on same dept and week_start should violate UniqueConstraint
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SchedulePlan.objects.create(
                    name='Plan Approved 2',
                    department=self.dept,
                    week_start=week_start,
                    week_end=week_end,
                    status=SchedulePlan.Status.APPROVED,
                )

    def test_10_assignment_start_before_end_constraint(self):
        plan = SchedulePlan.objects.create(
            name='Plan Invalid Times',
            department=self.dept,
            week_start=date(2026, 10, 12),
            week_end=date(2026, 10, 18),
        )
        p = Patient.objects.create(patient_id='PT-M9-TIME', full_name='Patient Time', date_of_birth=date(1980, 1, 1))
        start_dt = timezone.make_aware(datetime(2026, 10, 12, 11, 0, 0))
        end_dt = timezone.make_aware(datetime(2026, 10, 12, 7, 0, 0))  # end <= start

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ScheduleAssignment.objects.create(
                    schedule_plan=plan,
                    patient=p,
                    machine=self.machine1,
                    scheduled_date=date(2026, 10, 12),
                    shift=Shift.SHIFT_1,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                )
