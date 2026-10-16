"""
Integration and RBAC Tests for TOPSIS API Endpoints
Task: T9-01 — TOPSIS V1: ALGORITHM, BACKEND, FRONTEND & ACCEPTANCE
Author: Nguyen Ho Truong Tam
"""

from datetime import date, timedelta
from django.contrib.gis.geos import Point
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from core.models import CustomUser, Department, Patient, BloodSample, DialysisMachine, DialysisSession, SchedulePlan


class TopsisApiTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # 1. Departments
        cls.dept_a = Department.objects.create(name="Khoa Loc Mau A", code="KLM-A")
        cls.dept_b = Department.objects.create(name="Khoa Loc Mau B", code="KLM-B")

        # 2. Users
        cls.admin = CustomUser.objects.create_user(
            username="admin_topsis",
            email="admin_top@example.com",
            password="Password123!",
            role=CustomUser.Role.ADMIN,
            department=cls.dept_a,
        )
        cls.doctor_a = CustomUser.objects.create_user(
            username="doc_top_a",
            email="doc_a@example.com",
            password="Password123!",
            role=CustomUser.Role.DOCTOR,
            department=cls.dept_a,
        )
        cls.doctor_b = CustomUser.objects.create_user(
            username="doc_top_b",
            email="doc_b@example.com",
            password="Password123!",
            role=CustomUser.Role.DOCTOR,
            department=cls.dept_b,
        )
        cls.nurse_a = CustomUser.objects.create_user(
            username="nurse_top_a",
            email="nurse_a@example.com",
            password="Password123!",
            role=CustomUser.Role.NURSE,
            department=cls.dept_a,
        )

        # 3. Patients for Dept A (with geolocation and blood samples)
        cls.pat_a1 = Patient.objects.create(
            patient_id="PT-TOP-A1",
            full_name="Nguyen Van A1",
            date_of_birth=date(1975, 5, 10),
            gender="MALE",
            status=Patient.Status.ACTIVE,
            department=cls.dept_a,
            dry_weight=60.0,
            location=Point(106.6900, 10.7700),  # near center
        )
        tz_now = timezone.now()
        BloodSample.objects.create(
            sample_id="BS-A1",
            patient=cls.pat_a1,
            collection_date=tz_now - timedelta(days=1),
            potassium_level=6.2,  # High potassium! Urgency
            hemoglobin_level=8.5,  # Low hemoglobin
            created_by=cls.admin,
        )

        cls.pat_a2 = Patient.objects.create(
            patient_id="PT-TOP-A2",
            full_name="Tran Thi A2",
            date_of_birth=date(1982, 3, 20),
            gender="FEMALE",
            status=Patient.Status.ACTIVE,
            department=cls.dept_a,
            dry_weight=52.0,
            location=Point(106.6500, 10.8200),
        )
        BloodSample.objects.create(
            sample_id="BS-A2",
            patient=cls.pat_a2,
            collection_date=tz_now - timedelta(days=2),
            potassium_level=4.8,  # Normal potassium
            hemoglobin_level=11.5,  # Moderate
            created_by=cls.admin,
        )

        # 4. Patients for Dept B
        cls.pat_b1 = Patient.objects.create(
            patient_id="PT-TOP-B1",
            full_name="Le Van B1",
            date_of_birth=date(1988, 8, 15),
            gender="MALE",
            status=Patient.Status.ACTIVE,
            department=cls.dept_b,
            dry_weight=65.0,
            location=Point(106.7200, 10.7900),
        )
        BloodSample.objects.create(
            sample_id="BS-B1",
            patient=cls.pat_b1,
            collection_date=tz_now - timedelta(days=1),
            potassium_level=5.9,
            hemoglobin_level=9.0,
            created_by=cls.admin,
        )

    def setUp(self):
        self.client = APIClient()

    def test_01_criteria_endpoint_authenticated(self):
        """GET /api/topsis/criteria/ returns criteria list and disclaimer."""
        self.client.force_authenticate(user=self.doctor_a)
        res = self.client.get(reverse("topsis-criteria"))
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertIn("criteria", res.data)
        self.assertEqual(len(res.data["criteria"]), 5)
        self.assertFalse(res.data["is_clinical_approved"])
        self.assertIn("disclaimer", res.data)

    def test_02_criteria_endpoint_unauthenticated_fails(self):
        """GET /api/topsis/criteria/ without login returns 401 Unauthorized."""
        res = self.client.get(reverse("topsis-criteria"))
        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_03_doctor_can_rank_own_department(self):
        """Doctor A successfully ranks patients of Department A."""
        self.client.force_authenticate(user=self.doctor_a)
        res = self.client.post(
            reverse("topsis-rank"),
            {"department": self.dept_a.id},
            format="json",
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data["total_patients"], 2)
        rankings = res.data["rankings"]
        self.assertEqual(len(rankings), 2)

        # Verify all ranked patients belong strictly to Department A
        for r in rankings:
            self.assertIn(r["alternative_id"], ["PT-TOP-A1", "PT-TOP-A2"])
            self.assertIn("closeness_score", r)
            self.assertIn("rank", r)
            self.assertIn("patient", r)

        # Since PT-TOP-A1 has high potassium (6.2) and low Hb (8.5), they should have higher priority
        self.assertEqual(rankings[0]["alternative_id"], "PT-TOP-A1")
        self.assertEqual(rankings[0]["rank"], 1)

    def test_04_doctor_cannot_rank_other_department(self):
        """Doctor A attempting to rank Department B is rejected with 403 Forbidden."""
        self.client.force_authenticate(user=self.doctor_a)
        res = self.client.post(
            reverse("topsis-rank"),
            {"department": self.dept_b.id},
            format="json",
        )
        self.assertEqual(res.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("khoa của mình", res.data.get("detail", ""))

    def test_05_doctor_b_isolated_to_dept_b(self):
        """Doctor B ranks Dept B with 0% leakage from Dept A."""
        self.client.force_authenticate(user=self.doctor_b)
        res = self.client.post(
            reverse("topsis-rank"),
            {"department": self.dept_b.id},
            format="json",
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data["total_patients"], 1)
        self.assertEqual(res.data["rankings"][0]["alternative_id"], "PT-TOP-B1")

    def test_06_admin_can_rank_any_department(self):
        """Admin has permission to rank any department or center-wide."""
        self.client.force_authenticate(user=self.admin)

        # Admin ranks Dept B
        res_b = self.client.post(reverse("topsis-rank"), {"department": self.dept_b.id}, format="json")
        self.assertEqual(res_b.status_code, status.HTTP_200_OK)
        self.assertEqual(res_b.data["total_patients"], 1)

        # Admin ranks all departments (center-wide)
        res_all = self.client.post(reverse("topsis-rank"), {}, format="json")
        self.assertEqual(res_all.status_code, status.HTTP_200_OK)
        self.assertEqual(res_all.data["total_patients"], 3)

    def test_07_custom_weights_and_coordinates_handled(self):
        """Validates custom weights and center coordinate overrides."""
        self.client.force_authenticate(user=self.admin)
        res = self.client.post(
            reverse("topsis-rank"),
            {
                "department": self.dept_a.id,
                "custom_weights": {
                    "C1_POTASSIUM": 0.5,
                    "C2_HEMOGLOBIN": 0.2,
                    "C3_DAYS_SINCE_SESSION": 0.1,
                    "C4_FLUID_OVERLOAD": 0.1,
                    "C5_DISTANCE_KM": 0.1,
                },
                "center_coordinates": [10.7769, 106.7009],
            },
            format="json",
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data["rankings"]), 2)

    def test_08_invalid_custom_weights_rejected(self):
        """Negative or non-numeric custom weights return 400 Bad Request."""
        self.client.force_authenticate(user=self.admin)
        res = self.client.post(
            reverse("topsis-rank"),
            {
                "department": self.dept_a.id,
                "custom_weights": {"C1_POTASSIUM": -0.5},
            },
            format="json",
        )
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    def test_09_read_only_safety_guarantee(self):
        """Ensures TOPSIS ranking does NOT modify any Patient, Session, or Plan in DB."""
        self.client.force_authenticate(user=self.admin)

        patients_before = list(Patient.objects.values("patient_id", "status", "updated_at"))
        sessions_count_before = DialysisSession.objects.count()
        plans_count_before = SchedulePlan.objects.count()

        res = self.client.post(reverse("topsis-rank"), {"department": self.dept_a.id}, format="json")
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        patients_after = list(Patient.objects.values("patient_id", "status", "updated_at"))
        self.assertEqual(patients_before, patients_after, "Patient table was modified during TOPSIS ranking!")
        self.assertEqual(DialysisSession.objects.count(), sessions_count_before)
        self.assertEqual(SchedulePlan.objects.count(), plans_count_before)
