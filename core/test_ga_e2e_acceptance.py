"""
End-to-End Acceptance Tests for Genetic Algorithm (GA V1) Scheduling & Web Integration
Task: T8-08 — GA V1 WEB INTEGRATION & FINAL ACCEPTANCE
Isolated verification using Django test database (test_dialysis_db) without touching dialysis_db.
"""

from datetime import date, timedelta, datetime, time
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from core.models import (
    CustomUser,
    Department,
    Patient,
    DialysisMachine,
    SchedulePlan,
    ScheduleAssignment,
    DialysisSession,
    Shift,
    TreatmentPattern,
)


class GAWebIntegrationE2EAcceptanceTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # 1. Department A: Khoa Loc Mau Co So 1
        cls.dept_a = Department.objects.create(
            name="Khoa Loc Mau Co So 1",
            code="KLM-CS1",
            description="Khoa loc mau co so 1 - Demo E2E",
        )

        # 2. Department B: Khoa Loc Mau Co So 2
        cls.dept_b = Department.objects.create(
            name="Khoa Loc Mau Co So 2",
            code="KLM-CS2",
            description="Khoa loc mau co so 2 - Demo E2E",
        )

        # 3. Users for Department A
        cls.doctor_a = CustomUser.objects.create_user(
            username="bs_khoa_a",
            email="bs_a@locmau.vn",
            password="Password123!",
            first_name="Bac si",
            last_name="Khoa A",
            role=CustomUser.Role.DOCTOR,
            department=cls.dept_a,
        )
        cls.nurse_a = CustomUser.objects.create_user(
            username="dd_khoa_a",
            email="dd_a@locmau.vn",
            password="Password123!",
            first_name="Dieu duong",
            last_name="Khoa A",
            role=CustomUser.Role.NURSE,
            department=cls.dept_a,
        )

        # 4. Users for Department B
        cls.doctor_b = CustomUser.objects.create_user(
            username="bs_khoa_b",
            email="bs_b@locmau.vn",
            password="Password123!",
            first_name="Bac si",
            last_name="Khoa B",
            role=CustomUser.Role.DOCTOR,
            department=cls.dept_b,
        )

        # 5. System Admin (Center-wide management)
        cls.admin_user = CustomUser.objects.create_user(
            username="admin_locmau",
            email="admin@locmau.vn",
            password="Password123!",
            first_name="Quan tri",
            last_name="Trung tam",
            role=CustomUser.Role.ADMIN,
            department=cls.dept_a,
        )

        # 6. Machines for Department A
        cls.mach_a1 = DialysisMachine.objects.create(
            machine_id="M-A01",
            name="Fresenius 4008S - A01",
            department=cls.dept_a,
            status=DialysisMachine.Status.AVAILABLE,
        )
        cls.mach_a2 = DialysisMachine.objects.create(
            machine_id="M-A02",
            name="Fresenius 5008S - A02",
            department=cls.dept_a,
            status=DialysisMachine.Status.AVAILABLE,
        )
        cls.mach_a3 = DialysisMachine.objects.create(
            machine_id="M-A03",
            name="Toray TR-8000 - A03",
            department=cls.dept_a,
            status=DialysisMachine.Status.AVAILABLE,
        )

        # 7. Machines for Department B
        cls.mach_b1 = DialysisMachine.objects.create(
            machine_id="M-B01",
            name="Nipro Surdial - B01",
            department=cls.dept_b,
            status=DialysisMachine.Status.AVAILABLE,
        )
        cls.mach_b2 = DialysisMachine.objects.create(
            machine_id="M-B02",
            name="Nipro Surdial - B02",
            department=cls.dept_b,
            status=DialysisMachine.Status.AVAILABLE,
        )

        # 8. Patients for Department A (4 patients)
        cls.patients_a = [
            Patient.objects.create(
                patient_id="PT-A001",
                full_name="Tran Van An",
                date_of_birth="1975-03-10",
                gender="MALE",
                status=Patient.Status.ACTIVE,
                treatment_pattern=TreatmentPattern.T2_T4_T6,
                preferred_shift=Shift.SHIFT_1,
                department=cls.dept_a,
            ),
            Patient.objects.create(
                patient_id="PT-A002",
                full_name="Le Thi Binh",
                date_of_birth="1982-07-22",
                gender="FEMALE",
                status=Patient.Status.ACTIVE,
                treatment_pattern=TreatmentPattern.T2_T4_T6,
                preferred_shift=Shift.SHIFT_2,
                department=cls.dept_a,
            ),
            Patient.objects.create(
                patient_id="PT-A003",
                full_name="Hoang Van Cuong",
                date_of_birth="1968-11-05",
                gender="MALE",
                status=Patient.Status.ACTIVE,
                treatment_pattern=TreatmentPattern.T3_T5_T7,
                preferred_shift=Shift.SHIFT_1,
                department=cls.dept_a,
            ),
            Patient.objects.create(
                patient_id="PT-A004",
                full_name="Pham Thi Dung",
                date_of_birth="1990-01-30",
                gender="FEMALE",
                status=Patient.Status.ACTIVE,
                treatment_pattern=TreatmentPattern.T3_T5_T7,
                preferred_shift=Shift.SHIFT_2,
                department=cls.dept_a,
            ),
        ]

        # 9. Patients for Department B (3 patients)
        cls.patients_b = [
            Patient.objects.create(
                patient_id="PT-B001",
                full_name="Doan Van Em",
                date_of_birth="1980-04-12",
                gender="MALE",
                status=Patient.Status.ACTIVE,
                treatment_pattern=TreatmentPattern.T2_T4_T6,
                preferred_shift=Shift.SHIFT_1,
                department=cls.dept_b,
            ),
            Patient.objects.create(
                patient_id="PT-B002",
                full_name="Vu Thi Giang",
                date_of_birth="1978-09-18",
                gender="FEMALE",
                status=Patient.Status.ACTIVE,
                treatment_pattern=TreatmentPattern.T3_T5_T7,
                preferred_shift=Shift.SHIFT_2,
                department=cls.dept_b,
            ),
            Patient.objects.create(
                patient_id="PT-B003",
                full_name="Ngo Van Hai",
                date_of_birth="1985-12-25",
                gender="MALE",
                status=Patient.Status.ACTIVE,
                treatment_pattern=TreatmentPattern.T2_T4_T6,
                preferred_shift=Shift.SHIFT_2,
                department=cls.dept_b,
            ),
        ]

        # Target week: Monday 2026-10-19 to Sunday 2026-10-25
        cls.target_monday = date(2026, 10, 19)

    def setUp(self):
        self.client = APIClient()

    def test_full_e2e_ga_lifecycle_department_a(self):
        """
        Step 1 to 5: Complete E2E flow for Department A.
        Doctor A logs in -> Generates GA plan (Seed=42) -> Inspects plan & convergence chart ->
        Approves plan -> Verifies DialysisSessions created -> Verifies sessions visible in schedule.
        """
        # --- 1. LOGIN AS DOCTOR A ---
        self.client.force_authenticate(user=self.doctor_a)

        # --- 2. GENERATE GA PLAN WITH SEED 42 ---
        url_generate = reverse("scheduleplan-generate")
        gen_res = self.client.post(
            url_generate,
            {
                "department": self.dept_a.id,
                "week_start": self.target_monday.isoformat(),
                "seed": 42,
            },
            format="json",
        )
        self.assertEqual(
            gen_res.status_code,
            status.HTTP_201_CREATED,
            f"Generate plan failed: {gen_res.data}",
        )
        plan_data = gen_res.data.get("plan", {})
        plan_id = plan_data.get("id")
        self.assertIsNotNone(plan_id)
        self.assertEqual(plan_data.get("status"), "PROPOSED")
        self.assertEqual(plan_data.get("department"), self.dept_a.id)

        # --- 3. VERIFY METADATA & FITNESS CONVERGENCE HISTORY ---
        meta = plan_data.get("algorithm_metadata", {})
        self.assertEqual(meta.get("seed"), 42)
        self.assertEqual(meta.get("engine"), "GA_V1")
        self.assertIn("fitness_history", meta)
        fitness_history = meta.get("fitness_history", [])
        self.assertIsInstance(fitness_history, list)
        self.assertGreater(len(fitness_history), 0)
        self.assertIn("best_fitness", fitness_history[0])
        self.assertIn("avg_fitness", fitness_history[0])

        # --- 4. GET PLAN DETAIL & ASSIGNMENTS ---
        url_detail = f"/api/schedule-plans/{plan_id}/"
        detail_res = self.client.get(url_detail)
        self.assertEqual(detail_res.status_code, status.HTTP_200_OK)

        url_assignments = f"/api/schedule-assignments/?schedule_plan={plan_id}&page_size=100"
        asgn_res = self.client.get(url_assignments)
        self.assertEqual(asgn_res.status_code, status.HTTP_200_OK)
        self.assertEqual(asgn_res.data.get("count"), 12)
        assignments = asgn_res.data.get("results", asgn_res.data)

        # 4 patients * 3 days each = 12 total sessions
        self.assertEqual(len(assignments), 12)

        # Verify all assignments strictly belong to Department A
        dept_a_patient_ids = [p.patient_id for p in self.patients_a]
        dept_a_machine_ids = [m.machine_id for m in [self.mach_a1, self.mach_a2, self.mach_a3]]
        for asgn in assignments:
            self.assertIn(asgn["patient"], dept_a_patient_ids)
            self.assertIn(asgn["machine"], dept_a_machine_ids)
            self.assertEqual(asgn["source"], "GA")

        # --- 5. APPROVE PLAN ---
        url_approve = f"/api/schedule-plans/{plan_id}/approve/"
        appr_res = self.client.post(url_approve)
        self.assertEqual(
            appr_res.status_code,
            status.HTTP_200_OK,
            f"Approve plan failed: {appr_res.data}",
        )
        self.assertEqual(appr_res.data.get("created_sessions_count"), 12)
        self.assertEqual(appr_res.data["plan"]["status"], "APPROVED")
        self.assertEqual(appr_res.data["plan"]["approved_by"], self.doctor_a.id)

        # --- 6. VERIFY DIALYSIS SESSIONS CREATED ON SCHEDULE ---
        # Query sessions created for Department A during target week
        sessions_res = self.client.get(
            f"/api/sessions/?scheduled_start__date={self.target_monday.isoformat()}"
        )
        self.assertEqual(sessions_res.status_code, status.HTTP_200_OK)

        # Check in DB that exactly 12 DialysisSessions were generated
        created_sessions = DialysisSession.objects.filter(
            schedule_assignment__schedule_plan_id=plan_id
        )
        self.assertEqual(created_sessions.count(), 12)

        for sess in created_sessions:
            self.assertEqual(sess.status, DialysisSession.Status.SCHEDULED)
            self.assertIsNone(sess.assigned_nurse)
            self.assertIn(sess.patient_id, dept_a_patient_ids)
            self.assertIn(sess.machine_id, dept_a_machine_ids)
            # Verify start is in target week
            self.assertGreaterEqual(sess.scheduled_start.date(), self.target_monday)
            self.assertLessEqual(sess.scheduled_end.date(), self.target_monday + timedelta(days=6))

        # --- 7. VERIFY SESSION DETAIL ENDPOINT ---
        first_session = created_sessions.first()
        sess_detail_res = self.client.get(f"/api/sessions/{first_session.session_id}/")
        self.assertEqual(sess_detail_res.status_code, status.HTTP_200_OK)
        self.assertEqual(sess_detail_res.data["session_id"], first_session.session_id)
        self.assertEqual(sess_detail_res.data["status"], "SCHEDULED")

    def test_multi_department_isolation_and_rbac(self):
        """
        Step 6 & 7: Verify two independent departments (A & B) with full RBAC enforcement.
        1. Doctor A creates Plan A -> Doctor B cannot see or approve Plan A.
        2. Doctor B creates Plan B -> Doctor B plan contains ONLY Dept B patients/machines.
        3. Doctor A cannot generate plan for Dept B (403 Forbidden).
        4. Nurse A cannot approve or reject plans (403 Forbidden).
        """
        # 1. Admin creates Plan A for Dept A
        self.client.force_authenticate(user=self.admin_user)
        gen_a = self.client.post(
            reverse("scheduleplan-generate"),
            {"department": self.dept_a.id, "week_start": self.target_monday.isoformat(), "seed": 10},
            format="json",
        )
        self.assertEqual(gen_a.status_code, status.HTTP_201_CREATED)
        plan_a_id = gen_a.data["plan"]["id"]

        # 2. Doctor B logs in
        self.client.force_authenticate(user=self.doctor_b)

        # Doctor B cannot see Plan A in list
        list_b = self.client.get("/api/schedule-plans/")
        self.assertEqual(list_b.status_code, status.HTTP_200_OK)
        ids_b = [p["id"] for p in list_b.data.get("results", list_b.data)]
        self.assertNotIn(plan_a_id, ids_b, "Doctor B can view Plan A in list!")

        # Doctor B cannot retrieve Plan A detail (404)
        detail_b = self.client.get(f"/api/schedule-plans/{plan_a_id}/")
        self.assertEqual(detail_b.status_code, status.HTTP_404_NOT_FOUND)

        # Doctor B cannot approve Plan A (404)
        appr_b = self.client.post(f"/api/schedule-plans/{plan_a_id}/approve/")
        self.assertEqual(appr_b.status_code, status.HTTP_404_NOT_FOUND)

        # 3. Doctor B generates Plan B for Dept B
        gen_b = self.client.post(
            reverse("scheduleplan-generate"),
            {"department": self.dept_b.id, "week_start": self.target_monday.isoformat(), "seed": 20},
            format="json",
        )
        self.assertEqual(gen_b.status_code, status.HTTP_201_CREATED)
        plan_b_id = gen_b.data["plan"]["id"]

        # Verify Plan B has 0% leakage from Dept A
        dept_b_asgns = ScheduleAssignment.objects.filter(schedule_plan_id=plan_b_id)
        dept_b_pat_ids = [p.patient_id for p in self.patients_b]
        dept_b_mach_ids = [m.machine_id for m in [self.mach_b1, self.mach_b2]]
        self.assertEqual(dept_b_asgns.count(), 9)  # 3 patients * 3 days = 9
        for asgn in dept_b_asgns:
            self.assertIn(asgn.patient_id, dept_b_pat_ids)
            self.assertIn(asgn.machine_id, dept_b_mach_ids)

        # 4. Doctor A attempts cross-department actions
        self.client.force_authenticate(user=self.doctor_a)

        # Doctor A generates for Dept B -> 403 Forbidden
        cross_gen = self.client.post(
            reverse("scheduleplan-generate"),
            {"department": self.dept_b.id, "week_start": self.target_monday.isoformat()},
            format="json",
        )
        self.assertEqual(cross_gen.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("khoa của mình", cross_gen.data.get("detail", ""))

        # Doctor A views Plan B -> 404
        cross_view = self.client.get(f"/api/schedule-plans/{plan_b_id}/")
        self.assertEqual(cross_view.status_code, status.HTTP_404_NOT_FOUND)

        # Doctor A approves Plan B -> 404
        cross_appr = self.client.post(f"/api/schedule-plans/{plan_b_id}/approve/")
        self.assertEqual(cross_appr.status_code, status.HTTP_404_NOT_FOUND)

        # 5. Nurse A attempts privileged operations
        self.client.force_authenticate(user=self.nurse_a)
        nurse_gen = self.client.post(
            reverse("scheduleplan-generate"),
            {"department": self.dept_a.id, "week_start": self.target_monday.isoformat()},
            format="json",
        )
        self.assertEqual(nurse_gen.status_code, status.HTTP_403_FORBIDDEN)

        nurse_appr = self.client.post(f"/api/schedule-plans/{plan_a_id}/approve/")
        self.assertEqual(nurse_appr.status_code, status.HTTP_403_FORBIDDEN)

        nurse_rej = self.client.post(f"/api/schedule-plans/{plan_a_id}/reject/", {"reason": "Test"})
        self.assertEqual(nurse_rej.status_code, status.HTTP_403_FORBIDDEN)

    def test_infeasible_schedule_and_validation_errors(self):
        """
        Step 8: Input validation & infeasible schedule error handling.
        1. Non-Monday week start returns 400 Bad Request.
        2. Department with 0 patients returns 400 Bad Request.
        3. Department with 0 available machines returns 400 Bad Request.
        4. Rejection without reason returns 400 Bad Request.
        """
        self.client.force_authenticate(user=self.admin_user)

        # 1. Non-Monday week_start (Tuesday 2026-10-20)
        tuesday = date(2026, 10, 20)
        res_tue = self.client.post(
            reverse("scheduleplan-generate"),
            {"department": self.dept_a.id, "week_start": tuesday.isoformat()},
            format="json",
        )
        self.assertEqual(res_tue.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Thứ 2", res_tue.data.get("detail", ""))

        # 2. Empty department (0 patients, 0 machines)
        dept_empty = Department.objects.create(name="Khoa Trong", code="KLM-EMPTY")
        res_empty = self.client.post(
            reverse("scheduleplan-generate"),
            {"department": dept_empty.id, "week_start": self.target_monday.isoformat()},
            format="json",
        )
        self.assertEqual(res_empty.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("detail", res_empty.data)

        # 3. Department with patients but 0 available machines
        dept_no_mach = Department.objects.create(name="Khoa Khong May", code="KLM-NOMACH")
        Patient.objects.create(
            patient_id="PT-NOMACH-1",
            full_name="Benh Nhan Test",
            date_of_birth="1980-01-01",
            status=Patient.Status.ACTIVE,
            treatment_pattern=TreatmentPattern.T2_T4_T6,
            department=dept_no_mach,
        )
        res_no_mach = self.client.post(
            reverse("scheduleplan-generate"),
            {"department": dept_no_mach.id, "week_start": self.target_monday.isoformat()},
            format="json",
        )
        self.assertEqual(res_no_mach.status_code, status.HTTP_400_BAD_REQUEST)

        # 4. Reject plan without reason
        plan_a = SchedulePlan.objects.create(
            name="Plan Test Reject",
            department=self.dept_a,
            week_start=self.target_monday,
            week_end=self.target_monday + timedelta(days=6),
            status=SchedulePlan.Status.PROPOSED,
        )
        res_blank_reject = self.client.post(
            f"/api/schedule-plans/{plan_a.id}/reject/",
            {"reason": "   "},
            format="json",
        )
        self.assertEqual(res_blank_reject.status_code, status.HTTP_400_BAD_REQUEST)
