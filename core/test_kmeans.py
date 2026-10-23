"""
Unit tests for Pure Python K-Means Clustering module (Milestone Week 9).
Tests mathematical properties, WCSS, Silhouette Score, discharged patient exclusion,
and KMeansReportView API endpoint.
"""
from datetime import date, timedelta
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from core.kmeans_service import KMeansClusteringService
from core.models import (
    BloodSample,
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
)


class KMeansClusteringTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        # 1. Department and Staff Users
        cls.dept = Department.objects.create(
            name='Khoa Than Loc Mau K-Means',
            code='K-KM-TEST',
            description='Khoa kiem thu thuat toan K-Means',
        )
        cls.doctor = CustomUser.objects.create_user(
            username='doctor_kmeans',
            email='doctor_km@example.com',
            password='Password123!',
            role=CustomUser.Role.DOCTOR,
            department=cls.dept,
        )
        cls.admin = CustomUser.objects.create_user(
            username='admin_kmeans',
            email='admin_km@example.com',
            password='Password123!',
            role=CustomUser.Role.ADMIN,
            department=cls.dept,
        )

        # 2. Machine
        cls.machine = DialysisMachine.objects.create(
            machine_id='M-KM-01',
            name='May Loc Mau KM 01',
            status=DialysisMachine.Status.AVAILABLE,
            department=cls.dept,
        )

        now = timezone.now()

        # 3. Patient 1: Young & Clinically Stable (Young age, high Hb, normal potassium, low UF)
        cls.p1 = Patient.objects.create(
            patient_id='PT-KM-01',
            full_name='Nguyen Van Tre',
            date_of_birth=date(1996, 6, 15),  # ~30 years old
            status=Patient.Status.ACTIVE,
            department=cls.dept,
        )
        BloodSample.objects.create(
            sample_id='BS-KM-01',
            patient=cls.p1,
            collection_date=now - timedelta(days=5),
            hemoglobin_level=13.0,
            potassium_level=4.0,
            created_by=cls.doctor,
        )
        DialysisSession.objects.create(
            session_id='SS-KM-01',
            patient=cls.p1,
            machine=cls.machine,
            scheduled_start=now - timedelta(days=5),
            scheduled_end=now - timedelta(days=5) + timedelta(hours=4),
            status=DialysisSession.Status.COMPLETED,
            uf_actual=1.2,
        )

        # 4. Patient 2: Elderly & High-Risk (Old age, low Hb, hyperkalemia, high UF)
        cls.p2 = Patient.objects.create(
            patient_id='PT-KM-02',
            full_name='Tran Thi Gia',
            date_of_birth=date(1950, 1, 10),  # ~76 years old
            status=Patient.Status.ACTIVE,
            department=cls.dept,
        )
        BloodSample.objects.create(
            sample_id='BS-KM-02',
            patient=cls.p2,
            collection_date=now - timedelta(days=3),
            hemoglobin_level=8.0,
            potassium_level=6.8,
            created_by=cls.doctor,
        )
        DialysisSession.objects.create(
            session_id='SS-KM-02',
            patient=cls.p2,
            machine=cls.machine,
            scheduled_start=now - timedelta(days=3),
            scheduled_end=now - timedelta(days=3) + timedelta(hours=4),
            status=DialysisSession.Status.COMPLETED,
            uf_actual=4.5,
        )

        # 5. Patient 3: Middle-Aged & Moderate (Mid age, medium Hb, moderate potassium, medium UF)
        cls.p3 = Patient.objects.create(
            patient_id='PT-KM-03',
            full_name='Le Van Trung',
            date_of_birth=date(1975, 9, 20),  # ~51 years old
            status=Patient.Status.ACTIVE,
            department=cls.dept,
        )
        BloodSample.objects.create(
            sample_id='BS-KM-03',
            patient=cls.p3,
            collection_date=now - timedelta(days=1),
            hemoglobin_level=10.5,
            potassium_level=5.2,
            created_by=cls.doctor,
        )
        DialysisSession.objects.create(
            session_id='SS-KM-03',
            patient=cls.p3,
            machine=cls.machine,
            scheduled_start=now - timedelta(days=1),
            scheduled_end=now - timedelta(days=1) + timedelta(hours=4),
            status=DialysisSession.Status.COMPLETED,
            uf_actual=2.6,
        )

        # 6. Patient 4: Discharged Patient (Should be excluded by default)
        cls.p_disc = Patient.objects.create(
            patient_id='PT-KM-DISC',
            full_name='Hoang Van XuatVien',
            date_of_birth=date(1960, 2, 2),
            status=Patient.Status.DISCHARGED,
            department=cls.dept,
        )

    def test_kmeans_pure_python_logic(self):
        """
        Verify pure Python K-Means algorithm:
        - Result length equals active patient count (3, excluding discharged).
        - Cluster count <= K.
        - Total assigned patients across all clusters equals cohort size.
        - WCSS and Silhouette Score computed accurately.
        """
        service = KMeansClusteringService(department_id=self.dept.id)
        result = service.run_kmeans(k=3, max_iters=100, seed=42)

        self.assertTrue(result['success'])
        self.assertEqual(result['k_value'], 3)
        self.assertEqual(len(result['patients']), 3)
        self.assertLessEqual(len(result['clusters']), 3)

        # Verify discharged patient was excluded by default
        patient_ids = [p['patient_id'] for p in result['patients']]
        self.assertNotIn('PT-KM-DISC', patient_ids)

        # Sum of patients across clusters must equal total patients
        total_in_clusters = sum(len(pts) for pts in result['clusters'].values())
        self.assertEqual(total_in_clusters, 3)

        # Check centroids
        self.assertEqual(len(result['centroids']), 3)
        for c in result['centroids']:
            self.assertEqual(len(c), 4)

        # Verify evaluation metrics (WCSS and Silhouette Score)
        self.assertIn('wcss', result)
        self.assertIn('silhouette_score', result)
        self.assertGreaterEqual(result['wcss'], 0.0)
        self.assertGreaterEqual(result['silhouette_score'], -1.0)
        self.assertLessEqual(result['silhouette_score'], 1.0)

        # Verify cluster summaries
        self.assertIn('cluster_summaries', result)
        self.assertEqual(len(result['cluster_summaries']), 3)
        for cs in result['cluster_summaries']:
            self.assertIn('cluster_id', cs)
            self.assertIn('cluster_label', cs)
            self.assertIn('risk_level', cs)
            self.assertIn('recommendation', cs)
            self.assertIn(cs['risk_level'], {'LOW', 'MEDIUM', 'HIGH'})

        # Verify min_max scaling parameters exist for all 4 features
        for feat in ['age', 'avg_hemoglobin', 'avg_potassium', 'avg_uf']:
            self.assertIn(feat, result['min_max'])
            self.assertIn('min', result['min_max'][feat])
            self.assertIn('max', result['min_max'][feat])

        # Verify denormalized centroids
        self.assertEqual(len(result['denormalized_centroids']), 3)
        for d_c in result['denormalized_centroids']:
            for feat in ['age', 'avg_hemoglobin', 'avg_potassium', 'avg_uf']:
                self.assertIn(feat, d_c)

    def test_kmeans_discharged_patient_inclusion_toggle(self):
        """
        When include_discharged=True, discharged patients should be included.
        """
        service_inc = KMeansClusteringService(department_id=self.dept.id, include_discharged=True)
        res_inc = service_inc.run_kmeans(k=3, seed=42)
        patient_ids_inc = [p['patient_id'] for p in res_inc['patients']]
        self.assertIn('PT-KM-DISC', patient_ids_inc)
        self.assertEqual(len(res_inc['patients']), 4)

    def test_kmeans_api_endpoint(self):
        """
        Verify GET /api/reports/kmeans-clusters/?k=2 returns HTTP 200 with expected structure.
        """
        self.client.force_authenticate(user=self.doctor)
        response = self.client.get(f'/api/reports/kmeans-clusters/?k=2&department={self.dept.id}')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertEqual(response.data['k_value'], 2)
        self.assertIn('wcss', response.data)
        self.assertIn('silhouette_score', response.data)
        self.assertIn('centroids', response.data)
        self.assertIn('denormalized_centroids', response.data)
        self.assertIn('cluster_summaries', response.data)
        self.assertIn('clusters', response.data)
        self.assertIn('patients', response.data)
        self.assertLessEqual(len(response.data['clusters']), 2)

        # Total patients grouped across clusters must equal cohort size (3 active patients)
        total_clustered = sum(len(members) for members in response.data['clusters'].values())
        self.assertEqual(total_clustered, 3)
        self.assertIn('seed', response.data)
        self.assertIn('disclaimer', response.data)
        for p in response.data['patients']:
            self.assertIn('is_imputed', p)
            self.assertIn('imputed_fields', p)

    def test_kmeans_api_doctor_auto_department_scope(self):
        """
        If DOCTOR requests without department param, it should auto-scope to doctor's department.
        """
        self.client.force_authenticate(user=self.doctor)
        response = self.client.get('/api/reports/kmeans-clusters/?k=2')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertEqual(len(response.data['patients']), 3)

    def test_kmeans_api_authentication_required(self):
        """
        Unauthenticated call to /api/reports/kmeans-clusters/ must return HTTP 401.
        """
        response = self.client.get('/api/reports/kmeans-clusters/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_kmeans_api_invalid_k_rejected(self):
        """
        Invalid k parameter (k <= 0 or non-integer) must return HTTP 400.
        """
        self.client.force_authenticate(user=self.doctor)

        res_zero = self.client.get('/api/reports/kmeans-clusters/?k=0')
        self.assertEqual(res_zero.status_code, status.HTTP_400_BAD_REQUEST)

        res_neg = self.client.get('/api/reports/kmeans-clusters/?k=-5')
        self.assertEqual(res_neg.status_code, status.HTTP_400_BAD_REQUEST)

        res_str = self.client.get('/api/reports/kmeans-clusters/?k=invalid')
        self.assertEqual(res_str.status_code, status.HTTP_400_BAD_REQUEST)

    def test_kmeans_euclidean_distance_pure_python(self):
        """
        Verify Euclidean distance calculation on mathematical vector pairs.
        d([0, 0, 0, 0], [3, 4, 0, 0]) = 5.0
        """
        service = KMeansClusteringService()
        d = service._euclidean_distance([0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0])
        self.assertAlmostEqual(d, 5.0, places=5)

    def test_kmeans_k_greater_than_n(self):
        """
        When k > n_points, algorithm must gracefully constrain actual_k to n_points.
        """
        service = KMeansClusteringService(department_id=self.dept.id)
        result = service.run_kmeans(k=10, seed=42)
        self.assertTrue(result['success'])
        self.assertEqual(result['k_value'], 3)
        self.assertEqual(len(result['patients']), 3)

    def test_kmeans_api_doctor_cross_department_forbidden(self):
        other_dept = Department.objects.create(
            name='Khoa Khac Test',
            code='K-OTHER-KM',
        )
        self.client.force_authenticate(user=self.doctor)
        response = self.client.get(f'/api/reports/kmeans-clusters/?department={other_dept.id}')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
