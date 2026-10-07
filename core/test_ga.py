from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from datetime import date, timedelta, datetime
from core.models import CustomUser, Department, Patient, DialysisMachine, SchedulePlan, ScheduleAssignment, DialysisSession, Shift

class GeneticAlgorithmTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.dept = Department.objects.create(code="D1", name="Khoa Lọc máu")
        self.admin = CustomUser.objects.create_user(username="admin", email="a@a.com", password="pw", role=CustomUser.Role.ADMIN, department=self.dept)
        self.nurse = CustomUser.objects.create_user(username="nurse", email="n@n.com", password="pw", role=CustomUser.Role.NURSE, department=self.dept)
        
        self.week_start = date(2026, 10, 5) # Monday
        
        self.machine1 = DialysisMachine.objects.create(machine_id="M1", name="Máy 1", department=self.dept, status=DialysisMachine.Status.AVAILABLE)
        self.machine2 = DialysisMachine.objects.create(machine_id="M2", name="Máy 2", department=self.dept, status=DialysisMachine.Status.AVAILABLE)
        self.machine_broken = DialysisMachine.objects.create(machine_id="M3", name="Máy 3", department=self.dept, status=DialysisMachine.Status.BROKEN)
        
        self.p1 = Patient.objects.create(patient_id="P1", full_name="Nguyễn Văn A", date_of_birth="1980-01-01", status=Patient.Status.ACTIVE, treatment_pattern='T2_T4_T6')
        self.p2 = Patient.objects.create(patient_id="P2", full_name="Trần Thị B", date_of_birth="1985-01-01", status=Patient.Status.ACTIVE, treatment_pattern='T3_T5_T7')
        self.p_no_pattern = Patient.objects.create(patient_id="P3", full_name="Lê Văn C", date_of_birth="1990-01-01", status=Patient.Status.ACTIVE)
        self.p_discharged = Patient.objects.create(patient_id="P4", full_name="Phạm D", date_of_birth="1995-01-01", status=Patient.Status.DISCHARGED, treatment_pattern='T2_T4_T6')

    def test_permission_generate(self):
        url = reverse('scheduleplan-generate')
        # Unauth
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)
        
        # Nurse
        self.client.force_authenticate(user=self.nurse)
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_403_FORBIDDEN)
        
        # Admin
        self.client.force_authenticate(user=self.admin)
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        
    def test_week_start_must_be_monday(self):
        url = reverse('scheduleplan-generate')
        self.client.force_authenticate(user=self.admin)
        
        # Sunday (2026-10-11)
        sunday = date(2026, 10, 11)
        res = self.client.post(url, {'department': self.dept.id, 'week_start': sunday.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Tuần bắt đầu phải là Thứ 2', res.data['detail'])
        
        # Monday (2026-10-12)
        monday = date(2026, 10, 12)
        res = self.client.post(url, {'department': self.dept.id, 'week_start': monday.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        
    def test_ga_constraints_and_results(self):
        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        
        plan_id = res.data['plan']['id']
        plan = SchedulePlan.objects.get(id=plan_id)
        
        # SchedulePlan is PROPOSED
        self.assertEqual(plan.status, SchedulePlan.Status.PROPOSED)
        
        # Fitness is saved
        self.assertIsNotNone(plan.fitness_score)
        self.assertIn('population_size', plan.algorithm_metadata)
        
        assignments = plan.assignments.all()
        self.assertTrue(len(assignments) > 0)
        
        # Check source
        for a in assignments:
            self.assertEqual(a.source, ScheduleAssignment.Source.GA)
            # Machine not broken
            self.assertNotEqual(a.machine, self.machine_broken)
            # Start < End
            self.assertLess(a.start_datetime, a.end_datetime)
            # Within week
            self.assertGreaterEqual(a.scheduled_date, plan.week_start)
            self.assertLessEqual(a.scheduled_date, plan.week_end)
            
        # P1 has 3 assignments, P2 has 3 assignments. Total 6. P3, P4 have 0.
        p1_assignments = assignments.filter(patient=self.p1)
        self.assertEqual(p1_assignments.count(), 3)
        p2_assignments = assignments.filter(patient=self.p2)
        self.assertEqual(p2_assignments.count(), 3)
        self.assertEqual(assignments.filter(patient=self.p_no_pattern).count(), 0)
        self.assertEqual(assignments.filter(patient=self.p_discharged).count(), 0)
        
        # Check specific days for T2_T4_T6
        dates_p1 = [a.scheduled_date for a in p1_assignments]
        self.assertIn(self.week_start, dates_p1) # Mon
        self.assertIn(self.week_start + timedelta(days=2), dates_p1) # Wed
        self.assertIn(self.week_start + timedelta(days=4), dates_p1) # Fri
        
        # Check no duplicate machine/date/shift
        unique_slots = set((a.machine_id, a.scheduled_date, a.shift) for a in assignments)
        self.assertEqual(len(unique_slots), len(assignments))
        
        # Check no DialysisSession created
        sessions = DialysisSession.objects.filter(schedule_assignment__schedule_plan=plan)
        self.assertEqual(sessions.count(), 0)

    def test_no_valid_patients(self):
        # Delete patients
        Patient.objects.all().delete()
        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Không tìm thấy bệnh nhân", res.data['detail'])

    def test_ga_impossible_schedule(self):
        # Create 10 more patients on T2_T4_T6, but we only have 2 machines
        # 12 patients * 3 days = 36 slots.
        # 2 machines * 3 shifts = 6 slots per day, total 18 slots per week.
        # It's impossible to schedule 12 patients on a single day.
        for i in range(10):
            Patient.objects.create(
                patient_id=f"PX{i}", 
                full_name=f"Bệnh nhân X{i}", 
                date_of_birth="1980-01-01", 
                status=Patient.Status.ACTIVE, 
                treatment_pattern='T2_T4_T6'
            )
        
        initial_plan_count = SchedulePlan.objects.count()
        initial_assignment_count = ScheduleAssignment.objects.count()

        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Không thể tạo lịch hợp lệ", res.data['detail'])
        
        self.assertEqual(SchedulePlan.objects.count(), initial_plan_count)
        self.assertEqual(ScheduleAssignment.objects.count(), initial_assignment_count)

    def test_ga_overlap_existing_session(self):
        # Create an existing session for Machine 1 on Monday Shift 1
        DialysisSession.objects.create(
            session_id="S_EXIST",
            patient=self.p1,
            machine=self.machine1,
            scheduled_start=datetime.combine(self.week_start, datetime.strptime('07:00', '%H:%M').time()),
            scheduled_end=datetime.combine(self.week_start, datetime.strptime('11:00', '%H:%M').time()),
            status=DialysisSession.Status.SCHEDULED
        )
        
        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        plan_id = res.data['plan']['id']
        plan = SchedulePlan.objects.get(id=plan_id)
        assignments = plan.assignments.all()
        
        # Check no overlap
        overlap = assignments.filter(
            machine=self.machine1, 
            scheduled_date=self.week_start, 
            shift=Shift.SHIFT_1
        ).exists()
        self.assertFalse(overlap, "GA created an assignment that overlaps with an existing session")

    def test_treatment_pattern_t3_t5_t7(self):
        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        
        plan_id = res.data['plan']['id']
        plan = SchedulePlan.objects.get(id=plan_id)
        
        p2_assignments = plan.assignments.filter(patient=self.p2)
        self.assertEqual(p2_assignments.count(), 3)
        
        dates_p2 = [a.scheduled_date for a in p2_assignments]
        # T3 (Tue), T5 (Thu), T7 (Sat)
        self.assertIn(self.week_start + timedelta(days=1), dates_p2) # Tue
        self.assertIn(self.week_start + timedelta(days=3), dates_p2) # Thu
        self.assertIn(self.week_start + timedelta(days=5), dates_p2) # Sat
        
        for d in dates_p2:
            self.assertIn(d.weekday(), [1, 3, 5], f"Date {d} is not Tue, Thu, or Sat")
