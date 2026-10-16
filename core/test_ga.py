from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from datetime import date, timedelta, datetime, time
from django.utils import timezone
from core.models import CustomUser, Department, Patient, DialysisMachine, SchedulePlan, ScheduleAssignment, DialysisSession, Shift
from core.services import validate_schedule_plan_for_approval

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
        
        self.p1 = Patient.objects.create(patient_id="P1", full_name="Nguyễn Văn A", date_of_birth="1980-01-01", status=Patient.Status.ACTIVE, treatment_pattern='T2_T4_T6', department=self.dept)
        self.p2 = Patient.objects.create(patient_id="P2", full_name="Trần Thị B", date_of_birth="1985-01-01", status=Patient.Status.ACTIVE, treatment_pattern='T3_T5_T7', department=self.dept)
        self.p_no_pattern = Patient.objects.create(patient_id="P3", full_name="Lê Văn C", date_of_birth="1990-01-01", status=Patient.Status.ACTIVE, department=self.dept)
        self.p_discharged = Patient.objects.create(patient_id="P4", full_name="Phạm D", date_of_birth="1995-01-01", status=Patient.Status.DISCHARGED, treatment_pattern='T2_T4_T6', department=self.dept)

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
                treatment_pattern='T2_T4_T6',
                department=self.dept
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
        # Create an existing session for Machine 1 on Monday Shift 1 (full overlap)
        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(self.week_start, time(7, 0)), tz)
        end_dt = timezone.make_aware(datetime.combine(self.week_start, time(11, 0)), tz)
        DialysisSession.objects.create(
            session_id="S_EXIST",
            patient=self.p1,
            machine=self.machine1,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
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

    def test_ga_overlap_existing_session_partial(self):
        # Create an existing session partially overlapping Shift 1 (08:30 to 12:30 on Machine 1)
        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(self.week_start, time(8, 30)), tz)
        end_dt = timezone.make_aware(datetime.combine(self.week_start, time(12, 30)), tz)
        DialysisSession.objects.create(
            session_id="S_PARTIAL",
            patient=self.p1,
            machine=self.machine1,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            status=DialysisSession.Status.SCHEDULED
        )

        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})

        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        plan = SchedulePlan.objects.get(id=res.data['plan']['id'])
        assignments = plan.assignments.all()

        # S_PARTIAL overlaps with Shift 1 (07:00-11:00) and Shift 2 (12:00-16:00) on M1
        overlap_shift1 = assignments.filter(machine=self.machine1, scheduled_date=self.week_start, shift=Shift.SHIFT_1).exists()
        overlap_shift2 = assignments.filter(machine=self.machine1, scheduled_date=self.week_start, shift=Shift.SHIFT_2).exists()
        self.assertFalse(overlap_shift1, "GA created assignment overlapping partial session in Shift 1")
        self.assertFalse(overlap_shift2, "GA created assignment overlapping partial session in Shift 2")

    def test_ga_non_overlapping_existing_session_allowed(self):
        # Existing session completes before Shift 1 starts (05:00 to 06:30 on Monday)
        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(self.week_start, time(5, 0)), tz)
        end_dt = timezone.make_aware(datetime.combine(self.week_start, time(6, 30)), tz)
        DialysisSession.objects.create(
            session_id="S_EARLY",
            patient=self.p1,
            machine=self.machine1,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            status=DialysisSession.Status.COMPLETED
        )

        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})

        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        plan = SchedulePlan.objects.get(id=res.data['plan']['id'])
        self.assertIsNotNone(plan.fitness_score)
        self.assertLess(plan.fitness_score, 10000)

    def test_ga_stored_datetimes_are_timezone_aware(self):
        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)

        plan = SchedulePlan.objects.get(id=res.data['plan']['id'])
        assignments = plan.assignments.all()
        self.assertTrue(assignments.exists())

        tz = timezone.get_current_timezone()
        for a in assignments:
            self.assertTrue(timezone.is_aware(a.start_datetime), "start_datetime must be timezone-aware")
            self.assertTrue(timezone.is_aware(a.end_datetime), "end_datetime must be timezone-aware")
            self.assertLess(a.start_datetime, a.end_datetime)

            local_start = timezone.localtime(a.start_datetime, tz)
            local_end = timezone.localtime(a.end_datetime, tz)
            if a.shift == Shift.SHIFT_1:
                self.assertEqual(local_start.time(), time(7, 0))
                self.assertEqual(local_end.time(), time(11, 0))
            elif a.shift == Shift.SHIFT_2:
                self.assertEqual(local_start.time(), time(12, 0))
                self.assertEqual(local_end.time(), time(16, 0))
            elif a.shift == Shift.SHIFT_3:
                self.assertEqual(local_start.time(), time(17, 0))
                self.assertEqual(local_end.time(), time(21, 0))

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

    def test_ga_patient_overlap_different_machine_reproduction(self):
        """
        Reproduction test:
        Patient P1 has an existing DialysisSession on Machine 2 at Monday Shift 1 (07:00-11:00).
        Patient P1 prefers Shift 1.
        GA runs for Department 1 (which includes Machine 1 and Machine 2).
        Machine 1 is completely free at Shift 1.

        Before fix: GA only checks machine overlap for Machine 1 (free), so it schedules P1 on Machine 1 at Shift 1.
        This creates a patient conflict: P1 has two simultaneous sessions on Monday Shift 1 (Machine 1 and Machine 2).
        When the plan is evaluated for approval, it fails rule 10 (patient overlap).

        After fix: GA checks patient overlap against existing sessions of candidate patients across all departments.
        GA penalizes placing P1 at Shift 1 with HARD_PENALTY, forcing GA to schedule P1 at Shift 2 or Shift 3 instead.
        """
        self.p1.preferred_shift = Shift.SHIFT_1
        self.p1.save()

        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(self.week_start, time(7, 0)), tz)
        end_dt = timezone.make_aware(datetime.combine(self.week_start, time(11, 0)), tz)
        DialysisSession.objects.create(
            session_id="S_P1_EXISTING_M2",
            patient=self.p1,
            machine=self.machine2,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            status=DialysisSession.Status.SCHEDULED
        )

        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)

        plan = SchedulePlan.objects.get(id=res.data['plan']['id'])
        assignments = plan.assignments.all()

        # Check that P1 is NOT scheduled on Monday Shift 1 (which overlaps with S_P1_EXISTING_M2)
        p1_monday_shift1 = assignments.filter(
            patient=self.p1,
            scheduled_date=self.week_start,
            shift=Shift.SHIFT_1
        )
        self.assertFalse(
            p1_monday_shift1.exists(),
            "GA scheduled patient P1 at Monday Shift 1 despite P1 already having an active session on Machine 2"
        )

        # Check approval validation succeeds
        val = validate_schedule_plan_for_approval(plan)
        self.assertTrue(val['is_valid'], f"Plan validation failed: {val.get('errors')}")

    def test_ga_patient_overlap_other_department(self):
        """
        Patient P1 has an existing active session in another department (Dept 2).
        GA runs for Dept 1.
        GA must recognize P1's session in Dept 2 and not schedule P1 in the overlapping shift.
        """
        dept2 = Department.objects.create(code="D2", name="Khoa Lọc Máu 2")
        machine_other = DialysisMachine.objects.create(
            machine_id="M_OTHER",
            name="Máy Khoa 2",
            department=dept2,
            status=DialysisMachine.Status.AVAILABLE
        )

        self.p1.preferred_shift = Shift.SHIFT_1
        self.p1.save()

        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(self.week_start, time(7, 0)), tz)
        end_dt = timezone.make_aware(datetime.combine(self.week_start, time(11, 0)), tz)
        DialysisSession.objects.create(
            session_id="S_P1_DEPT2",
            patient=self.p1,
            machine=machine_other,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            status=DialysisSession.Status.SCHEDULED
        )

        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)

        plan = SchedulePlan.objects.get(id=res.data['plan']['id'])
        assignments = plan.assignments.all()

        p1_monday_shift1 = assignments.filter(
            patient=self.p1,
            scheduled_date=self.week_start,
            shift=Shift.SHIFT_1
        )
        self.assertFalse(
            p1_monday_shift1.exists(),
            "GA scheduled P1 at Monday Shift 1 despite active session in another department"
        )
        val = validate_schedule_plan_for_approval(plan)
        self.assertTrue(val['is_valid'], f"Plan validation failed: {val.get('errors')}")

    def test_ga_patient_overlap_partial(self):
        """
        Patient P1 has an existing session on Machine 2 partially overlapping Shift 1 and Shift 2 (09:00 - 13:00).
        GA must avoid assigning P1 to both Shift 1 (07:00-11:00) and Shift 2 (12:00-16:00) on Monday.
        GA must assign P1 to Shift 3.
        """
        self.p1.preferred_shift = Shift.SHIFT_1
        self.p1.save()

        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(self.week_start, time(9, 0)), tz)
        end_dt = timezone.make_aware(datetime.combine(self.week_start, time(13, 0)), tz)
        DialysisSession.objects.create(
            session_id="S_P1_PARTIAL",
            patient=self.p1,
            machine=self.machine2,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            status=DialysisSession.Status.SCHEDULED
        )

        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)

        plan = SchedulePlan.objects.get(id=res.data['plan']['id'])
        assignments = plan.assignments.all()

        overlap_shift1 = assignments.filter(patient=self.p1, scheduled_date=self.week_start, shift=Shift.SHIFT_1).exists()
        overlap_shift2 = assignments.filter(patient=self.p1, scheduled_date=self.week_start, shift=Shift.SHIFT_2).exists()
        self.assertFalse(overlap_shift1, "GA assigned P1 to Shift 1 which partially overlaps 09:00-13:00")
        self.assertFalse(overlap_shift2, "GA assigned P1 to Shift 2 which partially overlaps 09:00-13:00")

        p1_monday_shift3 = assignments.filter(patient=self.p1, scheduled_date=self.week_start, shift=Shift.SHIFT_3).exists()
        self.assertTrue(p1_monday_shift3, "GA should place P1 in non-overlapping Shift 3")
        val = validate_schedule_plan_for_approval(plan)
        self.assertTrue(val['is_valid'], f"Plan validation failed: {val.get('errors')}")

    def test_ga_patient_cancelled_session_allowed(self):
        """
        Patient P1 has a CANCELLED session on Monday Shift 1.
        GA should treat this slot as available for P1 and allow scheduling P1 at Shift 1.
        """
        self.p1.preferred_shift = Shift.SHIFT_1
        self.p1.save()

        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(self.week_start, time(7, 0)), tz)
        end_dt = timezone.make_aware(datetime.combine(self.week_start, time(11, 0)), tz)
        DialysisSession.objects.create(
            session_id="S_P1_CANCELLED",
            patient=self.p1,
            machine=self.machine2,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            status=DialysisSession.Status.CANCELLED
        )

        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)

        plan = SchedulePlan.objects.get(id=res.data['plan']['id'])
        assignments = plan.assignments.all()

        # Since preferred_shift is SHIFT_1 and CANCELLED sessions are ignored,
        # GA should happily schedule P1 at Shift 1 on Machine 1
        p1_monday_shift1 = assignments.filter(
            patient=self.p1,
            scheduled_date=self.week_start,
            shift=Shift.SHIFT_1
        )
        self.assertTrue(p1_monday_shift1.exists(), "GA should allow scheduling P1 at Shift 1 when existing session is CANCELLED")
        val = validate_schedule_plan_for_approval(plan)
        self.assertTrue(val['is_valid'], f"Plan validation failed: {val.get('errors')}")

    def test_ga_patient_non_overlapping_session_allowed(self):
        """
        Patient P1 has a session ending before Shift 1 starts (05:00 to 06:30 on Monday).
        GA should allow scheduling P1 at Shift 1 without penalty.
        """
        self.p1.preferred_shift = Shift.SHIFT_1
        self.p1.save()

        tz = timezone.get_current_timezone()
        start_dt = timezone.make_aware(datetime.combine(self.week_start, time(5, 0)), tz)
        end_dt = timezone.make_aware(datetime.combine(self.week_start, time(6, 30)), tz)
        DialysisSession.objects.create(
            session_id="S_P1_EARLY",
            patient=self.p1,
            machine=self.machine2,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            status=DialysisSession.Status.COMPLETED
        )

        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {'department': self.dept.id, 'week_start': self.week_start.isoformat()})
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)

        plan = SchedulePlan.objects.get(id=res.data['plan']['id'])
        assignments = plan.assignments.all()
        p1_monday_shift1 = assignments.filter(
            patient=self.p1,
            scheduled_date=self.week_start,
            shift=Shift.SHIFT_1
        )
        self.assertTrue(p1_monday_shift1.exists(), "GA should schedule P1 at Shift 1 since early session does not overlap")
        self.assertLess(plan.fitness_score, 10000)
        val = validate_schedule_plan_for_approval(plan)
        self.assertTrue(val['is_valid'], f"Plan validation failed: {val.get('errors')}")

    def test_ga_patient_all_shifts_blocked_fails_cleanly(self):
        """
        Patient P1 has existing sessions covering all 3 shifts on Monday.
        GA cannot schedule P1 on Monday without violating hard constraint.
        GA must return HTTP 400 Bad Request with clear detail, and not create any Plan or Assignments.
        """
        tz = timezone.get_current_timezone()
        shifts_times = [
            (time(7, 0), time(11, 0), "S_BLOCKED_1"),
            (time(12, 0), time(16, 0), "S_BLOCKED_2"),
            (time(17, 0), time(21, 0), "S_BLOCKED_3"),
        ]
        for s_time, e_time, sid in shifts_times:
            DialysisSession.objects.create(
                session_id=sid,
                patient=self.p1,
                machine=self.machine2,
                scheduled_start=timezone.make_aware(datetime.combine(self.week_start, s_time), tz),
                scheduled_end=timezone.make_aware(datetime.combine(self.week_start, e_time), tz),
                status=DialysisSession.Status.SCHEDULED
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
