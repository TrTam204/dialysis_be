from datetime import date, timedelta, datetime, time
from django.db import transaction
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient
from rest_framework.serializers import ValidationError

from core.models import CustomUser, Department, Patient, DialysisMachine, SchedulePlan, ScheduleAssignment, DialysisSession, Shift, TreatmentPattern
from core.ga_service import GeneticAlgorithmService, GAConfig
from core.services import approve_schedule_plan, validate_schedule_plan_for_approval


class GeneticAlgorithmV1Tests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.week_start = date(2026, 10, 19) # Monday

        # 1. Khoa A
        self.dept_a = Department.objects.create(code="DEPT_A", name="Khoa Loc Mau A")
        self.admin = CustomUser.objects.create_user(
            username="admin_v1", email="admin_v1@example.com", password="password",
            role=CustomUser.Role.ADMIN, department=self.dept_a
        )
        self.doctor_a = CustomUser.objects.create_user(
            username="doctor_a", email="doctor_a@example.com", password="password",
            role=CustomUser.Role.DOCTOR, department=self.dept_a
        )
        self.nurse_a = CustomUser.objects.create_user(
            username="nurse_a", email="nurse_a@example.com", password="password",
            role=CustomUser.Role.NURSE, department=self.dept_a
        )
        self.machine_a1 = DialysisMachine.objects.create(
            machine_id="M_A1", name="May A1", department=self.dept_a, status=DialysisMachine.Status.AVAILABLE
        )
        self.machine_a2 = DialysisMachine.objects.create(
            machine_id="M_A2", name="May A2", department=self.dept_a, status=DialysisMachine.Status.AVAILABLE
        )
        self.p_a1 = Patient.objects.create(
            patient_id="PA1", full_name="Benh Nhan A1", date_of_birth="1980-01-01",
            status=Patient.Status.ACTIVE, treatment_pattern='T2_T4_T6', department=self.dept_a,
            preferred_shift=Shift.SHIFT_1
        )
        self.p_a2 = Patient.objects.create(
            patient_id="PA2", full_name="Benh Nhan A2", date_of_birth="1985-01-01",
            status=Patient.Status.ACTIVE, treatment_pattern='T2_T4_T6', department=self.dept_a,
            preferred_shift=Shift.SHIFT_2
        )

        # 2. Khoa B
        self.dept_b = Department.objects.create(code="DEPT_B", name="Khoa Loc Mau B")
        self.doctor_b = CustomUser.objects.create_user(
            username="doctor_b", email="doctor_b@example.com", password="password",
            role=CustomUser.Role.DOCTOR, department=self.dept_b
        )
        self.machine_b1 = DialysisMachine.objects.create(
            machine_id="M_B1", name="May B1", department=self.dept_b, status=DialysisMachine.Status.AVAILABLE
        )
        self.machine_b2 = DialysisMachine.objects.create(
            machine_id="M_B2", name="May B2", department=self.dept_b, status=DialysisMachine.Status.AVAILABLE
        )
        self.p_b1 = Patient.objects.create(
            patient_id="PB1", full_name="Benh Nhan B1", date_of_birth="1990-01-01",
            status=Patient.Status.ACTIVE, treatment_pattern='T3_T5_T7', department=self.dept_b,
            preferred_shift=Shift.SHIFT_1
        )
        self.p_b2 = Patient.objects.create(
            patient_id="PB2", full_name="Benh Nhan B2", date_of_birth="1992-01-01",
            status=Patient.Status.ACTIVE, treatment_pattern='T3_T5_T7', department=self.dept_b,
            preferred_shift=Shift.SHIFT_2
        )

    def test_two_independent_departments_no_leakage(self):
        """
        C.1: GA chỉ lập lịch cho bệnh nhân và máy của đúng khoa.
        Không có bệnh nhân hoặc máy của Khoa B lọt vào kế hoạch Khoa A và ngược lại.
        """
        self.client.force_authenticate(user=self.admin)
        url = reverse('scheduleplan-generate')

        # 1. Sinh lịch cho Khoa A
        res_a = self.client.post(url, {
            'department': self.dept_a.id,
            'week_start': self.week_start.isoformat(),
            'seed': 42
        })
        self.assertEqual(res_a.status_code, status.HTTP_201_CREATED)
        plan_a = SchedulePlan.objects.get(id=res_a.data['plan']['id'])
        assignments_a = plan_a.assignments.all()
        self.assertEqual(assignments_a.count(), 6) # 2 patients * 3 days = 6

        # Xác minh toàn bộ assignments_a thuộc Khoa A
        for asgn in assignments_a:
            self.assertIn(asgn.patient_id, ["PA1", "PA2"], "Bệnh nhân không thuộc Khoa A xuất hiện trong plan Khoa A!")
            self.assertIn(asgn.machine_id, ["M_A1", "M_A2"], "Máy không thuộc Khoa A xuất hiện trong plan Khoa A!")
            self.assertEqual(asgn.patient.department, self.dept_a)
            self.assertEqual(asgn.machine.department, self.dept_a)

        # 2. Sinh lịch cho Khoa B
        res_b = self.client.post(url, {
            'department': self.dept_b.id,
            'week_start': self.week_start.isoformat(),
            'seed': 42
        })
        self.assertEqual(res_b.status_code, status.HTTP_201_CREATED)
        plan_b = SchedulePlan.objects.get(id=res_b.data['plan']['id'])
        assignments_b = plan_b.assignments.all()
        self.assertEqual(assignments_b.count(), 6) # 2 patients * 3 days = 6

        # Xác minh toàn bộ assignments_b thuộc Khoa B
        for asgn in assignments_b:
            self.assertIn(asgn.patient_id, ["PB1", "PB2"], "Bệnh nhân không thuộc Khoa B xuất hiện trong plan Khoa B!")
            self.assertIn(asgn.machine_id, ["M_B1", "M_B2"], "Máy không thuộc Khoa B xuất hiện trong plan Khoa B!")
            self.assertEqual(asgn.patient.department, self.dept_b)
            self.assertEqual(asgn.machine.department, self.dept_b)

    def test_max_one_session_per_patient_per_day(self):
        """
        C.2: Ràng buộc tối đa 1 phiên/bệnh nhân/ngày trong lịch GA định kỳ.
        1) Không bao giờ xếp 2 ca cho cùng 1 bệnh nhân trong 1 ngày.
        2) Nếu bệnh nhân đã có phiên định kỳ (từ ScheduleAssignment trước) trên cùng ngày, GA từ chối xếp thêm.
        3) validate_schedule_plan_for_approval từ chối kế hoạch nếu có >= 2 ca/ngày cho 1 bệnh nhân.
        """
        # 1. Kiểm tra GA bình thường tạo đúng 1 ca/ngày/bệnh nhân
        plan, err = GeneticAlgorithmService.generate_plan(self.dept_a.id, self.week_start, self.admin, seed=100)
        self.assertIsNone(err)
        self.assertIsNotNone(plan)
        
        assignments = plan.assignments.all()
        dates_pa1 = [a.scheduled_date for a in assignments.filter(patient=self.p_a1)]
        self.assertEqual(len(dates_pa1), len(set(dates_pa1)), "Bệnh nhân PA1 bị xếp trùng ngày trong GA!")

        # 2. Thử nghiệm khi bệnh nhân đã có một phiên định kỳ đã duyệt trước đó vào Thứ 2
        # Tạo phiên định kỳ cho PA1 vào Thứ 2 (2026-10-19)
        tz = timezone.get_current_timezone()
        dt_start = timezone.make_aware(datetime.combine(self.week_start, time(7, 0)), tz)
        dt_end = timezone.make_aware(datetime.combine(self.week_start, time(11, 0)), tz)

        # Tạo dummy plan và assignment trước
        prev_plan = SchedulePlan.objects.create(
            name="Plan cu", department=self.dept_a, week_start=self.week_start,
            week_end=self.week_start + timedelta(days=6), status=SchedulePlan.Status.APPROVED
        )
        prev_asgn = ScheduleAssignment.objects.create(
            schedule_plan=prev_plan, patient=self.p_a1, machine=self.machine_a1,
            scheduled_date=self.week_start, shift=Shift.SHIFT_1,
            start_datetime=dt_start, end_datetime=dt_end
        )
        DialysisSession.objects.create(
            session_id="S_PERIODIC_EXIST",
            patient=self.p_a1,
            machine=self.machine_a1,
            scheduled_start=dt_start,
            scheduled_end=dt_end,
            schedule_assignment=prev_asgn,
            status=DialysisSession.Status.SCHEDULED
        )

        # Bây giờ PA1 đã có phiên định kỳ Thứ 2. GA không thể xếp thêm ca Thứ 2 cho PA1 mà không vi phạm hard penalty.
        # Do PA1 có pattern T2_T4_T6, ca Thứ 2 của PA1 bị xung đột.
        plan_conflict, err_conflict = GeneticAlgorithmService.generate_plan(self.dept_a.id, self.week_start, self.admin, seed=100)
        self.assertIsNone(plan_conflict)
        self.assertIn("Không thể tạo lịch hợp lệ", err_conflict)

        # 3. Xác minh database unique constraint uq_asgn_plan_patient_date chặn tạo 2 ca/ngày cho cùng 1 bệnh nhân
        from django.db.utils import IntegrityError
        plan_manual = SchedulePlan.objects.create(
            name="Plan 2 ca 1 ngay", department=self.dept_a, week_start=self.week_start,
            week_end=self.week_start + timedelta(days=6), status=SchedulePlan.Status.PROPOSED
        )
        ScheduleAssignment.objects.create(
            schedule_plan=plan_manual, patient=self.p_a1, machine=self.machine_a1,
            scheduled_date=self.week_start, shift=Shift.SHIFT_1,
            start_datetime=dt_start, end_datetime=dt_end
        )
        dt_start_s2 = timezone.make_aware(datetime.combine(self.week_start, time(12, 0)), tz)
        dt_end_s2 = timezone.make_aware(datetime.combine(self.week_start, time(16, 0)), tz)
        with transaction.atomic():
            with self.assertRaises(IntegrityError):
                ScheduleAssignment.objects.create(
                    schedule_plan=plan_manual, patient=self.p_a1, machine=self.machine_a2,
                    scheduled_date=self.week_start, shift=Shift.SHIFT_2,
                    start_datetime=dt_start_s2, end_datetime=dt_end_s2
                )

        # 4. Kiểm tra validate_schedule_plan_for_approval từ chối nếu có mock assignments vi phạm
        class MockAssignment:
            def __init__(self, patient, machine, date_val, start, end, shift):
                self.patient = patient
                self.patient_id = patient.patient_id
                self.machine = machine
                self.machine_id = machine.machine_id
                self.scheduled_date = date_val
                self.start_datetime = start
                self.end_datetime = end
                self.shift = shift

        class MockPlan:
            def __init__(self, dept, w_start, w_end, asgns):
                self.status = SchedulePlan.Status.PROPOSED
                self.department = dept
                self.department_id = dept.id
                self.week_start = w_start
                self.week_end = w_end
                self._asgns = asgns
                self.assignments = self
            def all(self):
                return self._asgns
            def select_related(self, *args):
                return self

        mock_asgns = [
            MockAssignment(self.p_a1, self.machine_a1, self.week_start, dt_start, dt_end, Shift.SHIFT_1),
            MockAssignment(self.p_a1, self.machine_a2, self.week_start, dt_start_s2, dt_end_s2, Shift.SHIFT_2),
        ]
        mock_plan = MockPlan(self.dept_a, self.week_start, self.week_start + timedelta(days=6), mock_asgns)
        val_mock = validate_schedule_plan_for_approval(mock_plan)
        self.assertFalse(val_mock['is_valid'])
        self.assertTrue(any('Maximum 1 session per patient per day' in e for e in val_mock['errors']))

    def test_seed_reproducibility(self):
        """
        C.3: Kiểm tra tính tái lập khi sử dụng random seed.
        Chạy 2 lần với cùng seed trên cùng dataset phải cho kết quả giống hệt nhau 100%.
        """
        # Lần chạy 1 với seed=777
        plan1, err1 = GeneticAlgorithmService.generate_plan(self.dept_a.id, self.week_start, self.admin, seed=777)
        self.assertIsNone(err1)
        
        # Xóa các assignments của plan1 để không gây xung đột DB (nhưng giữ plan1 để so sánh)
        asgns1 = list(plan1.assignments.values('patient_id', 'machine_id', 'scheduled_date', 'shift'))
        plan1.assignments.all().delete()
        plan1.delete()

        # Lần chạy 2 với seed=777
        plan2, err2 = GeneticAlgorithmService.generate_plan(self.dept_a.id, self.week_start, self.admin, seed=777)
        self.assertIsNone(err2)
        asgns2 = list(plan2.assignments.values('patient_id', 'machine_id', 'scheduled_date', 'shift'))

        # So sánh fitness score và từng assignment
        self.assertEqual(plan1.fitness_score, plan2.fitness_score)
        self.assertEqual(len(asgns1), len(asgns2))
        self.assertEqual(asgns1, asgns2, "Hai lần chạy cùng seed phải cho kết quả phân công giống hệt nhau!")

        # Kiểm tra metadata lưu đầy đủ
        self.assertEqual(plan2.algorithm_metadata.get('seed'), 777)
        self.assertIn('fitness_history', plan2.algorithm_metadata)
        history = plan2.algorithm_metadata['fitness_history']
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        self.assertIn('best_fitness', history[0])
        self.assertIn('avg_fitness', history[0])

    def test_approve_rejects_cross_department_plan(self):
        """
        C.4: APPROVE từ chối kế hoạch nếu có bệnh nhân hoặc máy sai khoa.
        """
        plan = SchedulePlan.objects.create(
            name="Plan Test Cross Dept", department=self.dept_a, week_start=self.week_start,
            week_end=self.week_start + timedelta(days=6), status=SchedulePlan.Status.PROPOSED
        )
        tz = timezone.get_current_timezone()
        dt_start = timezone.make_aware(datetime.combine(self.week_start, time(7, 0)), tz)
        dt_end = timezone.make_aware(datetime.combine(self.week_start, time(11, 0)), tz)

        # 1. Thử gán máy thuộc Khoa B vào plan của Khoa A
        asgn_wrong_machine = ScheduleAssignment.objects.create(
            schedule_plan=plan, patient=self.p_a1, machine=self.machine_b1, # Machine B1 in Dept B!
            scheduled_date=self.week_start, shift=Shift.SHIFT_1,
            start_datetime=dt_start, end_datetime=dt_end
        )
        val = validate_schedule_plan_for_approval(plan)
        self.assertFalse(val['is_valid'])
        self.assertTrue(any('does not belong to plan department' in e for e in val['errors']))
        with self.assertRaises(ValidationError):
            approve_schedule_plan(plan.id, self.doctor_a)

        # Sửa lại máy đúng, nhưng thử gán bệnh nhân thuộc Khoa B
        asgn_wrong_machine.machine = self.machine_a1
        asgn_wrong_machine.patient = self.p_b1 # Patient B1 in Dept B!
        asgn_wrong_machine.save()

        val2 = validate_schedule_plan_for_approval(plan)
        self.assertFalse(val2['is_valid'])
        self.assertTrue(any('does not belong to plan department' in e for e in val2['errors']))
        with self.assertRaises(ValidationError):
            approve_schedule_plan(plan.id, self.doctor_a)

    def test_doctor_cannot_generate_for_other_department(self):
        """
        C.5: RBAC - Bác sĩ Khoa A không được phép tạo kế hoạch cho Khoa B.
        API trả về 403 Forbidden.
        """
        self.client.force_authenticate(user=self.doctor_a)
        url = reverse('scheduleplan-generate')
        res = self.client.post(url, {
            'department': self.dept_b.id,
            'week_start': self.week_start.isoformat(),
        })
        self.assertEqual(res.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn('khoa của mình', res.data.get('detail', ''))

    def test_doctor_cannot_view_or_approve_other_department_plan(self):
        """
        C.6: RBAC - Bác sĩ Khoa B không thể xem hoặc duyệt kế hoạch của Khoa A.
        """
        # Tạo plan cho Khoa A bởi Admin
        plan_a, err = GeneticAlgorithmService.generate_plan(self.dept_a.id, self.week_start, self.admin, seed=42)
        self.assertIsNone(err)

        # Bác sĩ Khoa B đăng nhập
        self.client.force_authenticate(user=self.doctor_b)

        # 1. Thử xem danh sách kế hoạch: không chứa plan_a
        res_list = self.client.get('/api/schedule-plans/')
        self.assertEqual(res_list.status_code, status.HTTP_200_OK)
        ids = [p['id'] for p in res_list.data.get('results', res_list.data)]
        self.assertNotIn(plan_a.id, ids)

        # 2. Thử xem chi tiết plan_a: 404 Not Found do queryset đã lọc theo khoa
        res_detail = self.client.get(f'/api/schedule-plans/{plan_a.id}/')
        self.assertEqual(res_detail.status_code, status.HTTP_404_NOT_FOUND)

        # 3. Thử phê duyệt plan_a: 404 Not Found
        res_approve = self.client.post(f'/api/schedule-plans/{plan_a.id}/approve/')
        self.assertEqual(res_approve.status_code, status.HTTP_404_NOT_FOUND)

        # 4. Thử từ chối plan_a: 404 Not Found
        res_reject = self.client.post(f'/api/schedule-plans/{plan_a.id}/reject/', {'reason': 'Doctor B reject'})
        self.assertEqual(res_reject.status_code, status.HTTP_404_NOT_FOUND)

        # 5. Bác sĩ Khoa A thao tác hợp lệ
        self.client.force_authenticate(user=self.doctor_a)
        res_a_detail = self.client.get(f'/api/schedule-plans/{plan_a.id}/')
        self.assertEqual(res_a_detail.status_code, status.HTTP_200_OK)

        res_a_approve = self.client.post(f'/api/schedule-plans/{plan_a.id}/approve/')
        self.assertEqual(res_a_approve.status_code, status.HTTP_200_OK)
        self.assertEqual(res_a_approve.data['plan']['status'], 'APPROVED')

    def test_nurse_cannot_generate_or_approve_plan(self):
        """
        C.7: RBAC - Điều dưỡng (NURSE) không có quyền sinh lịch hoặc phê duyệt.
        """
        plan_a, _ = GeneticAlgorithmService.generate_plan(self.dept_a.id, self.week_start, self.admin, seed=42)
        self.client.force_authenticate(user=self.nurse_a)

        # Thử generate
        url_gen = reverse('scheduleplan-generate')
        res_gen = self.client.post(url_gen, {
            'department': self.dept_a.id,
            'week_start': self.week_start.isoformat(),
        })
        self.assertEqual(res_gen.status_code, status.HTTP_403_FORBIDDEN)

        # Thử approve
        res_appr = self.client.post(f'/api/schedule-plans/{plan_a.id}/approve/')
        self.assertEqual(res_appr.status_code, status.HTTP_403_FORBIDDEN)

        # Thử reject
        res_rej = self.client.post(f'/api/schedule-plans/{plan_a.id}/reject/', {'reason': 'Nurse try'})
        self.assertEqual(res_rej.status_code, status.HTTP_403_FORBIDDEN)

    def test_admin_full_access_across_departments(self):
        """
        C.8: RBAC - ADMIN có toàn quyền tạo và xem kế hoạch của cả Khoa A và Khoa B.
        """
        self.client.force_authenticate(user=self.admin)
        url_gen = reverse('scheduleplan-generate')

        # Admin tạo plan Khoa A
        res_a = self.client.post(url_gen, {
            'department': self.dept_a.id,
            'week_start': self.week_start.isoformat(),
            'seed': 123
        })
        self.assertEqual(res_a.status_code, status.HTTP_201_CREATED)

        # Admin tạo plan Khoa B
        res_b = self.client.post(url_gen, {
            'department': self.dept_b.id,
            'week_start': self.week_start.isoformat(),
            'seed': 456
        })
        self.assertEqual(res_b.status_code, status.HTTP_201_CREATED)

        # Admin xem danh sách thấy cả hai
        res_list = self.client.get('/api/schedule-plans/')
        self.assertEqual(res_list.status_code, status.HTTP_200_OK)
        ids = [p['id'] for p in res_list.data.get('results', res_list.data)]
        self.assertIn(res_a.data['plan']['id'], ids)
        self.assertIn(res_b.data['plan']['id'], ids)

    def test_infeasible_department_ga_error_handling(self):
        """
        C.9: Xử lý lỗi khi khoa không có bệnh nhân hoặc không có máy lọc.
        """
        dept_empty = Department.objects.create(code="DEPT_EMPTY", name="Khoa Rong")
        self.client.force_authenticate(user=self.admin)
        url_gen = reverse('scheduleplan-generate')

        res = self.client.post(url_gen, {
            'department': dept_empty.id,
            'week_start': self.week_start.isoformat(),
        })
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('detail', res.data)

