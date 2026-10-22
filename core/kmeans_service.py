"""
Pure Python implementation of K-Means Clustering for Hemodialysis Patients.
Course milestone: Week 9 unsupervised learning objective (Chapter 2.6 & 3.4).

Constraints:
- Strictly pure Python standard library (math, random).
- Absolutely NO sklearn, numpy, or pandas.
- Clusters patients based on 4 clinical features:
    1. age (from date_of_birth)
    2. avg_hemoglobin (from BloodSample)
    3. avg_potassium (from BloodSample)
    4. avg_uf (from COMPLETED DialysisSession)
"""
import math
import random
from django.utils import timezone

from .models import DialysisSession, Patient


class KMeansClusteringService:
    """
    Self-contained pure-Python K-Means clustering service.
    Includes data extraction, Min-Max normalization, Euclidean distance,
    iterative Lloyd's clustering algorithm, evaluation metrics (WCSS, Silhouette Score),
    and clinical cluster profiling.
    """

    FEATURE_NAMES = ['age', 'avg_hemoglobin', 'avg_potassium', 'avg_uf']

    # Clinical default baselines when lab/session history is not available
    DEFAULT_AGE = 50.0
    DEFAULT_HEMOGLOBIN = 11.0  # g/dL normal dialysis target
    DEFAULT_POTASSIUM = 4.5    # mEq/L normal serum potassium
    DEFAULT_UF = 0.0           # Liters

    def __init__(self, department_id=None, include_discharged=False):
        self.department_id = department_id
        self.include_discharged = include_discharged

    def _get_patient_features(self):
        """
        Query Django ORM for patients and calculate 4 clinical features for each:
        - age: calculated from Patient.date_of_birth to today.
        - avg_hemoglobin: mean of BloodSample.hemoglobin_level.
        - avg_potassium: mean of BloodSample.potassium_level.
        - avg_uf: mean of uf_actual across COMPLETED DialysisSession records.
        By default, DISCHARGED patients are excluded unless include_discharged=True.
        """
        qs = Patient.objects.all().order_by('patient_id')
        if not self.include_discharged:
            qs = qs.exclude(status=Patient.Status.DISCHARGED)

        if self.department_id is not None:
            qs = qs.filter(department_id=self.department_id)

        # Prefetch related blood samples and completed sessions for query efficiency
        qs = qs.prefetch_related('blood_samples', 'dialysis_sessions')

        today = timezone.localdate()
        features_list = []

        for patient in qs:
            # 1. Age
            if patient.date_of_birth:
                dob = patient.date_of_birth
                age = float(
                    today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                )
                age = max(0.0, age)
            else:
                age = self.DEFAULT_AGE

            # 2 & 3. Lab indicators: Hemoglobin and Potassium
            samples = list(patient.blood_samples.all())
            if samples:
                avg_hb = sum(s.hemoglobin_level for s in samples) / len(samples)
                avg_k = sum(s.potassium_level for s in samples) / len(samples)
            else:
                avg_hb = self.DEFAULT_HEMOGLOBIN
                avg_k = self.DEFAULT_POTASSIUM

            # 4. Ultrafiltration (UF actual) from completed sessions
            sessions = [
                s for s in patient.dialysis_sessions.all()
                if s.status == DialysisSession.Status.COMPLETED and s.uf_actual is not None
            ]
            if sessions:
                avg_uf = sum(s.uf_actual for s in sessions) / len(sessions)
            else:
                avg_uf = self.DEFAULT_UF

            raw_dict = {
                'age': round(age, 2),
                'avg_hemoglobin': round(avg_hb, 2),
                'avg_potassium': round(avg_k, 2),
                'avg_uf': round(avg_uf, 2),
            }
            raw_vector = [age, avg_hb, avg_k, avg_uf]

            features_list.append({
                'patient_id': patient.patient_id,
                'full_name': patient.full_name,
                'status': patient.status,
                'raw_features': raw_dict,
                'raw_vector': raw_vector,
            })

        return features_list

    def _normalize_data(self, features_list):
        """
        Apply Min-Max Scaling to map all 4 features into [0, 1].
        Keeps unrounded raw_bounds (float) for precise scaling and denormalization.
        Returns (min_max_display, raw_bounds, features_list).
        """
        if not features_list:
            return {}, {}, []

        min_max_display = {}
        raw_bounds = {}

        for j, feat_name in enumerate(self.FEATURE_NAMES):
            values = [p['raw_vector'][j] for p in features_list]
            raw_min = float(min(values))
            raw_max = float(max(values))
            raw_bounds[feat_name] = {'min': raw_min, 'max': raw_max}
            min_max_display[feat_name] = {
                'min': round(raw_min, 4),
                'max': round(raw_max, 4),
            }

        # Normalize each patient vector using unrounded raw_bounds
        for p in features_list:
            norm_vector = []
            norm_dict = {}
            for j, feat_name in enumerate(self.FEATURE_NAMES):
                raw_min = raw_bounds[feat_name]['min']
                raw_max = raw_bounds[feat_name]['max']
                diff = raw_max - raw_min
                if diff > 1e-9:
                    val = (p['raw_vector'][j] - raw_min) / diff
                else:
                    val = 0.5  # Neutral midpoint when all values in cohort are identical
                norm_vector.append(val)
                norm_dict[feat_name] = round(val, 4)

            p['normalized_vector'] = norm_vector
            p['normalized_features'] = norm_dict

        return min_max_display, raw_bounds, features_list

    def _euclidean_distance(self, pt1, pt2):
        """
        Compute standard Euclidean distance between two vectors using pure Python math.
        d(p, q) = sqrt(sum((p_i - q_i)^2))
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pt1, pt2)))

    def _calculate_metrics(self, clusters, centroids, actual_k, features_list):
        """
        Calculate mathematical evaluation metrics for K-Means using pure Python:
        1. WCSS (Within-Cluster Sum of Squares / Inertia):
           sum of squared Euclidean distances of each point to its assigned centroid.
        2. Silhouette Score:
           mean silhouette coefficient across all points.
        """
        # 1. WCSS (Inertia)
        wcss = 0.0
        for c_idx in range(actual_k):
            c_vec = centroids[c_idx]
            for p in clusters[c_idx]:
                dist = self._euclidean_distance(p['normalized_vector'], c_vec)
                wcss += dist ** 2
        wcss_rounded = round(wcss, 4)

        # 2. Silhouette Score
        total_points = len(features_list)
        if actual_k <= 1 or total_points <= actual_k:
            return wcss_rounded, 0.0

        s_scores = []
        for c_idx in range(actual_k):
            members = clusters[c_idx]
            n_members = len(members)

            if n_members <= 1:
                # A single-point cluster has silhouette score of 0 by definition
                for _ in members:
                    s_scores.append(0.0)
                continue

            for p in members:
                p_vec = p['normalized_vector']

                # a_i: mean intra-cluster distance to other members in C_k
                intra_dists = [
                    self._euclidean_distance(p_vec, other['normalized_vector'])
                    for other in members
                    if other is not p
                ]
                a_i = sum(intra_dists) / len(intra_dists) if intra_dists else 0.0

                # b_i: min mean distance to members of any other cluster C_m (m != c_idx)
                b_i = float('inf')
                for other_c in range(actual_k):
                    if other_c == c_idx:
                        continue
                    other_members = clusters[other_c]
                    if not other_members:
                        continue
                    inter_dists = [
                        self._euclidean_distance(p_vec, om['normalized_vector'])
                        for om in other_members
                    ]
                    mean_inter = sum(inter_dists) / len(inter_dists)
                    if mean_inter < b_i:
                        b_i = mean_inter

                if b_i == float('inf'):
                    s_i = 0.0
                else:
                    max_ab = max(a_i, b_i)
                    if max_ab > 1e-9:
                        s_i = (b_i - a_i) / max_ab
                    else:
                        s_i = 0.0

                s_scores.append(s_i)

        avg_silhouette = round(sum(s_scores) / len(s_scores), 4) if s_scores else 0.0
        return wcss_rounded, avg_silhouette

    def _generate_cluster_summaries(self, denormalized_centroids, clusters):
        """
        Generate automated clinical summaries for each cluster based on centroid values:
        - Evaluates Age, Hemoglobin, Potassium, and UF against clinical standards.
        - Assigns clinical tags and overall risk tier.
        """
        summaries = []
        for c_idx, denorm in enumerate(denormalized_centroids):
            age = denorm.get('age', self.DEFAULT_AGE)
            hb = denorm.get('avg_hemoglobin', self.DEFAULT_HEMOGLOBIN)
            k_val = denorm.get('avg_potassium', self.DEFAULT_POTASSIUM)
            uf = denorm.get('avg_uf', self.DEFAULT_UF)

            tags = []
            risk_points = 0

            # Age evaluation
            if age < 45:
                age_label = 'Trẻ tuổi (<45)'
            elif age <= 65:
                age_label = 'Trung niên (45-65)'
            else:
                age_label = 'Cao tuổi (>65)'
                risk_points += 1
            tags.append(age_label)

            # Hemoglobin evaluation (Normal hemodialysis target: 10 - 12 g/dL)
            if hb < 9.5:
                hb_label = 'Thiếu máu nặng (Hb < 9.5)'
                risk_points += 2
            elif hb < 11.0:
                hb_label = 'Thiếu máu nhẹ (Hb 9.5-11)'
                risk_points += 1
            else:
                hb_label = 'Hemoglobin đạt mục tiêu (Hb >= 11)'
            tags.append(hb_label)

            # Potassium evaluation (Normal serum K: 3.5 - 5.0 mEq/L)
            if k_val < 3.5:
                k_label = 'Hạ kali máu (<3.5)'
                risk_points += 1
            elif k_val <= 5.0:
                k_label = 'Kali ổn định (3.5-5.0)'
            elif k_val <= 6.0:
                k_label = 'Tăng kali vừa (5.0-6.0)'
                risk_points += 2
            else:
                k_label = 'Tăng kali nguy hiểm (>6.0)'
                risk_points += 3
            tags.append(k_label)

            # Ultrafiltration evaluation
            if uf < 2.0:
                uf_label = 'Siêu lọc thấp (UF < 2.0L)'
            elif uf <= 3.5:
                uf_label = 'Siêu lọc trung bình (UF 2.0-3.5L)'
                risk_points += 1
            else:
                uf_label = 'Rút dịch nhiều (UF > 3.5L)'
                risk_points += 2
            tags.append(uf_label)

            if risk_points >= 4:
                risk_level = 'HIGH'
                clinical_recommendation = (
                    'Nguy cơ biến cố tim mạch và quá tải tuần hoàn cao. '
                    'Cần theo dõi sát điện giải đồ, điều chỉnh chế độ ăn giảm kali và điều chỉnh dịch lọc.'
                )
            elif risk_points >= 2:
                risk_level = 'MEDIUM'
                clinical_recommendation = (
                    'Nguy cơ trung bình. Cần duy trì phác đồ hiện tại, '
                    'kiểm tra định kỳ hemoglobin và nhắc nhở hạn chế muối nước gian lọc.'
                )
            else:
                risk_level = 'LOW'
                clinical_recommendation = (
                    'Chỉ số sinh hóa và dịch lọc tương đối ổn định. '
                    'Tiếp tục theo dõi định kỳ theo lịch thường quy.'
                )

            summaries.append({
                'cluster_id': c_idx,
                'cluster_label': f'Cụm {c_idx + 1}',
                'patient_count': len(clusters.get(c_idx, [])),
                'risk_level': risk_level,
                'tags': tags,
                'recommendation': clinical_recommendation,
                'centroid': denorm,
            })

        return summaries

    def run_kmeans(self, k=3, max_iters=100, seed=42):
        """
        Core iterative K-Means algorithm using pure Python:
        1. Fetch patient features (excluding DISCHARGED by default).
        2. Normalize features using Min-Max scaling with unrounded bounds.
        3. Seed random generator and sample K initial centroids.
        4. Loop: Assign clusters -> Update centroids -> Check convergence.
        5. Compute WCSS and Silhouette Score.
        6. Generate clinical cluster summaries.
        7. Return structured clusters, centroids, metrics, and metadata.
        """
        features_list = self._get_patient_features()
        n_points = len(features_list)

        # Handle empty cohort edge case
        if n_points == 0:
            return {
                'success': True,
                'k_value': 0,
                'wcss': 0.0,
                'silhouette_score': 0.0,
                'centroids': [],
                'denormalized_centroids': [],
                'cluster_summaries': [],
                'min_max': {},
                'clusters': {},
                'patients': [],
                'iterations': 0,
                'converged': True,
            }

        # Constrain K to number of available data points
        actual_k = min(max(1, int(k)), n_points)

        min_max_display, raw_bounds, features_list = self._normalize_data(features_list)

        # Initialize centroids randomly from data points with deterministic seed
        rng = random.Random(seed)
        chosen_indices = rng.sample(range(n_points), actual_k)
        centroids = [list(features_list[idx]['normalized_vector']) for idx in chosen_indices]

        iteration = 0
        converged = False
        clusters = {c: [] for c in range(actual_k)}

        while iteration < max_iters and not converged:
            iteration += 1

            # 1. Assignment step: Assign each patient to closest centroid
            clusters = {c: [] for c in range(actual_k)}
            for p in features_list:
                p_vec = p['normalized_vector']
                best_c = 0
                min_dist = float('inf')
                for c_idx, c_vec in enumerate(centroids):
                    dist = self._euclidean_distance(p_vec, c_vec)
                    if dist < min_dist:
                        min_dist = dist
                        best_c = c_idx
                p['cluster_id'] = best_c
                clusters[best_c].append(p)

            # 2. Update step: Recalculate centroids as cluster means
            new_centroids = []
            dim = len(self.FEATURE_NAMES)
            for c_idx in range(actual_k):
                members = clusters[c_idx]
                if members:
                    new_c = [
                        sum(m['normalized_vector'][d] for m in members) / len(members)
                        for d in range(dim)
                    ]
                else:
                    # Empty cluster safeguard: retain previous centroid
                    new_c = list(centroids[c_idx])
                new_centroids.append(new_c)

            # 3. Convergence check: Centroids movement < threshold
            shift = sum(
                self._euclidean_distance(old_c, new_c)
                for old_c, new_c in zip(centroids, new_centroids)
            )
            if shift < 1e-7:
                converged = True
            centroids = new_centroids

        # Calculate denormalized centroids in original clinical units using unrounded raw_bounds
        denormalized_centroids = []
        for c in centroids:
            denorm = {}
            for d, feat_name in enumerate(self.FEATURE_NAMES):
                raw_min = raw_bounds[feat_name]['min']
                raw_max = raw_bounds[feat_name]['max']
                diff = raw_max - raw_min
                if diff > 1e-9:
                    val = c[d] * diff + raw_min
                else:
                    val = raw_min
                denorm[feat_name] = round(val, 2)
            denormalized_centroids.append(denorm)

        centroids_rounded = [[round(coord, 4) for coord in c] for c in centroids]

        # Calculate evaluation metrics: WCSS & Silhouette Score
        wcss, silhouette = self._calculate_metrics(clusters, centroids, actual_k, features_list)

        # Generate clinical cluster summaries
        cluster_summaries = self._generate_cluster_summaries(denormalized_centroids, clusters)

        # Format output clusters (string keys for JSON compatibility)
        clusters_output = {}
        for c_idx in range(actual_k):
            clusters_output[str(c_idx)] = [
                {
                    'patient_id': p['patient_id'],
                    'full_name': p['full_name'],
                    'cluster_id': c_idx,
                    'status': p['status'],
                    'features': p['raw_features'],
                    'normalized_features': p['normalized_features'],
                }
                for p in clusters[c_idx]
            ]

        return {
            'success': True,
            'k_value': actual_k,
            'wcss': wcss,
            'silhouette_score': silhouette,
            'centroids': centroids_rounded,
            'denormalized_centroids': denormalized_centroids,
            'cluster_summaries': cluster_summaries,
            'min_max': min_max_display,
            'clusters': clusters_output,
            'patients': [
                {
                    'patient_id': p['patient_id'],
                    'full_name': p['full_name'],
                    'cluster_id': p['cluster_id'],
                    'status': p['status'],
                    'features': p['raw_features'],
                    'normalized_features': p['normalized_features'],
                }
                for p in features_list
            ],
            'iterations': iteration,
            'converged': converged,
        }
