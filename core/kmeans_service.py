"""
Pure Python implementation of K-Means Clustering for Hemodialysis Patients.
Course milestone: Week 9 unsupervised learning objective (Chapter 2.6 & 3.4).

Constraints:
- Strictly pure Python standard library (math, random, typing).
- Absolutely NO sklearn, numpy, or pandas.
- Clusters patients based on 4 clinical/biochemical features:
    1. age (from date_of_birth)
    2. avg_hemoglobin (from BloodSample, g/dL)
    3. avg_potassium (from BloodSample, mmol/L)
    4. avg_idwg_pct (percentage of interdialytic weight gain / UF relative to dry weight, %)
       (aliased as avg_uf for backward compatibility with existing tests and UI)
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple
from django.utils import timezone

from .models import DialysisSession, Patient


class KMeansClusteringService:
    """
    Self-contained pure-Python K-Means clustering service.
    Includes data extraction, Min-Max normalization, Euclidean distance,
    K-Means++ centroid initialization (D^2 weighting), iterative Lloyd's clustering,
    empty cluster safeguards, multi-K evaluation (Elbow WCSS & Silhouette Score),
    and clinical cluster profiling compared against cohort baselines.
    """

    FEATURE_NAMES = ['age', 'avg_hemoglobin', 'avg_potassium', 'avg_idwg_pct']

    # Clinical default baselines when lab/session history is not available
    DEFAULT_AGE = 50.0
    DEFAULT_HEMOGLOBIN = 11.0    # g/dL normal dialysis target
    DEFAULT_POTASSIUM = 4.5      # mmol/L normal serum potassium
    DEFAULT_IDWG_PCT = 2.5       # % of dry weight (safe baseline IDWG)
    DEFAULT_UF = 2.5             # Backward compatibility alias

    def __init__(self, department_id=None, include_discharged=False):
        self.department_id = department_id
        self.include_discharged = include_discharged

    def _get_patient_features(self) -> List[Dict[str, Any]]:
        """
        Query Django ORM for patients and calculate 4 clinical features for each:
        - age: calculated from Patient.date_of_birth to today.
        - avg_hemoglobin: mean of BloodSample.hemoglobin_level.
        - avg_potassium: mean of BloodSample.potassium_level.
        - avg_idwg_pct: mean percentage of interdialytic weight gain / UF relative to dry weight
          from completed DialysisSession records: (pre_weight - dry_weight) / dry_weight * 100%.
          Falls back to (uf_actual / dry_weight) * 100% if pre_weight is absent.
        Tracks is_imputed and imputed_fields for clinical transparency.
        """
        qs = Patient.objects.all()

        if self.department_id is not None:
            qs = qs.filter(department_id=self.department_id)

        if not self.include_discharged:
            qs = qs.exclude(status=Patient.Status.DISCHARGED)

        qs = qs.prefetch_related('blood_samples', 'dialysis_sessions')

        today = timezone.now().date()
        features = []

        for p in qs:
            imputed_fields = []

            # 1. Age (years)
            if p.date_of_birth:
                age_years = (today - p.date_of_birth).days / 365.25
                age = round(age_years, 1)
            else:
                age = self.DEFAULT_AGE
                imputed_fields.append('age')

            # 2 & 3. Lab tests (Hemoglobin & Potassium)
            samples = list(p.blood_samples.all())
            valid_hb = [s.hemoglobin_level for s in samples if s.hemoglobin_level is not None]
            valid_k = [s.potassium_level for s in samples if s.potassium_level is not None]

            if valid_hb:
                avg_hb = round(sum(valid_hb) / len(valid_hb), 2)
            else:
                avg_hb = self.DEFAULT_HEMOGLOBIN
                imputed_fields.append('avg_hemoglobin')

            if valid_k:
                avg_k = round(sum(valid_k) / len(valid_k), 2)
            else:
                avg_k = self.DEFAULT_POTASSIUM
                imputed_fields.append('avg_potassium')

            # 4. Interdialytic Weight Gain Percentage (% IDWG)
            # Calculated from COMPLETED dialysis sessions relative to dry weight
            sessions = [
                s for s in p.dialysis_sessions.all()
                if s.status == DialysisSession.Status.COMPLETED
            ]

            dry_w = float(p.dry_weight) if p.dry_weight and p.dry_weight > 0 else 0.0
            idwg_pct_list = []

            for s in sessions:
                pre_w = float(s.pre_weight) if s.pre_weight is not None else None
                uf_act = float(s.uf_actual) if s.uf_actual is not None else None

                if pre_w is not None and dry_w > 0:
                    # % IDWG = max(0, pre_weight - dry_weight) / dry_weight * 100%
                    pct = max(0.0, pre_w - dry_w) / dry_w * 100.0
                    idwg_pct_list.append(pct)
                elif uf_act is not None and dry_w > 0:
                    # Approximation: UF (Liters ~ kg) / dry_weight * 100%
                    pct = max(0.0, uf_act) / dry_w * 100.0
                    idwg_pct_list.append(pct)

            if idwg_pct_list:
                avg_idwg = round(sum(idwg_pct_list) / len(idwg_pct_list), 2)
            else:
                avg_idwg = self.DEFAULT_IDWG_PCT
                imputed_fields.append('avg_idwg_pct')

            is_imputed = len(imputed_fields) > 0

            raw_dict = {
                'age': age,
                'avg_hemoglobin': avg_hb,
                'avg_potassium': avg_k,
                'avg_idwg_pct': avg_idwg,
                'avg_uf': avg_idwg,  # Backward compatibility alias
            }

            features.append({
                'patient_id': p.patient_id,
                'full_name': p.full_name,
                'status': getattr(p, 'status', 'ACTIVE'),
                'dry_weight': dry_w,
                'raw_features': raw_dict,
                'is_imputed': is_imputed,
                'imputed_fields': imputed_fields,
            })

        return features

    def _normalize_features(
        self, patient_features: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        """
        Pure Python Min-Max feature normalization: x_norm = (x - min) / (max - min).
        Maps all 4 clinical features to the range [0.0, 1.0].
        If max == min, normalized value is assigned 0.5.
        Also calculates min_max metadata for both avg_idwg_pct and avg_uf alias.
        """
        if not patient_features:
            return [], {}

        # 1. Find min and max for each feature
        min_max = {}
        all_feature_keys = ['age', 'avg_hemoglobin', 'avg_potassium', 'avg_idwg_pct', 'avg_uf']
        for feat in all_feature_keys:
            vals = [p['raw_features'][feat] for p in patient_features]
            min_max[feat] = {
                'min': min(vals),
                'max': max(vals),
            }

        # 2. Normalize features into a 4-dimensional mathematical vector
        normalized = []
        for p in patient_features:
            norm_vector = []
            norm_dict = {}
            for feat in self.FEATURE_NAMES:
                val = p['raw_features'][feat]
                f_min = min_max[feat]['min']
                f_max = min_max[feat]['max']
                if math.isclose(f_max, f_min, abs_tol=1e-9):
                    norm_val = 0.5
                else:
                    norm_val = (val - f_min) / (f_max - f_min)
                norm_vector.append(round(norm_val, 6))
                norm_dict[feat] = round(norm_val, 6)

            # Mirror to avg_uf for backward compatibility
            norm_dict['avg_uf'] = norm_dict['avg_idwg_pct']

            normalized.append({
                'patient_id': p['patient_id'],
                'full_name': p['full_name'],
                'status': p.get('status', 'ACTIVE'),
                'dry_weight': p['dry_weight'],
                'raw_features': p['raw_features'],
                'normalized_vector': norm_vector,
                'normalized_dict': norm_dict,
                'is_imputed': p['is_imputed'],
                'imputed_fields': p['imputed_fields'],
            })

        return normalized, min_max

    def extract_features(self) -> List[Dict[str, Any]]:
        """
        Extract and return normalized patient features list.
        Convenience helper for external callers and testing.
        """
        raw = self._get_patient_features()
        normalized, _ = self._normalize_features(raw)
        return normalized

    def _euclidean_distance(self, pt1: List[float], pt2: List[float]) -> float:
        """
        Compute standard Euclidean distance between two vectors using pure Python math.
        d(p, q) = sqrt(sum((p_i - q_i)^2))
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pt1, pt2)))

    def _init_centroids_plusplus(
        self, features_list: List[Any], k: int, rng: random.Random
    ) -> List[List[float]]:
        """
        K-Means++ initialization algorithm using pure Python standard library:
        1. Choose the first centroid uniformly at random from data points.
        2. For each data point x, compute D(x)^2: squared distance to the nearest already chosen centroid.
        3. Choose the next centroid with probability proportional to D(x)^2 using cumulative sum.
        4. Repeat until K centroids are chosen.
        Supports both List[Dict] with 'normalized_vector' and raw List[List[float]].
        """
        n_points = len(features_list)
        if n_points == 0 or k <= 0:
            return []

        def get_vec(item):
            if isinstance(item, dict) and 'normalized_vector' in item:
                return list(item['normalized_vector'])
            return list(item)

        if k >= n_points:
            return [get_vec(p) for p in features_list[:k]]

        # 1. First centroid chosen uniformly at random
        first_idx = rng.randint(0, n_points - 1)
        centroids = [get_vec(features_list[first_idx])]

        # 2. Iteratively choose remaining k - 1 centroids with D^2 probability weighting
        for _ in range(1, k):
            dist_sq_list = []
            for p in features_list:
                p_vec = get_vec(p)
                min_d_sq = min(
                    self._euclidean_distance(p_vec, c) ** 2
                    for c in centroids
                )
                dist_sq_list.append(min_d_sq)

            total_weight = sum(dist_sq_list)
            if total_weight > 1e-9:
                rand_val = rng.uniform(0.0, total_weight)
                cum_sum = 0.0
                chosen_idx = 0
                for idx, d_sq in enumerate(dist_sq_list):
                    cum_sum += d_sq
                    if cum_sum >= rand_val:
                        chosen_idx = idx
                        break
            else:
                # Fallback if points are all identical
                chosen_idx = rng.randint(0, n_points - 1)

            centroids.append(get_vec(features_list[chosen_idx]))

        return centroids

    def _cluster_iteration(
        self,
        features_list: List[Dict[str, Any]],
        initial_centroids: List[List[float]],
        max_iters: int = 100,
    ) -> Tuple[Dict[int, List[Dict[str, Any]]], List[List[float]], int, bool]:
        """
        Pure Python iterative Lloyd's clustering algorithm with empty cluster safeguard.
        Returns:
            (clusters_dict, final_centroids, iterations_run, converged_flag)
        """
        centroids = [list(c) for c in initial_centroids]
        actual_k = len(centroids)
        converged = False
        iteration = 0

        while iteration < max_iters and not converged:
            iteration += 1
            clusters = {i: [] for i in range(actual_k)}

            # 1. Assignment step: Assign each point to the nearest centroid
            for p in features_list:
                p_vec = p['normalized_vector']
                distances = [self._euclidean_distance(p_vec, c) for c in centroids]
                closest_c = min(range(actual_k), key=lambda idx: distances[idx])
                clusters[closest_c].append(p)

            # 2. Update step: Calculate new centroids as the mean vector of assigned points
            new_centroids = []
            for c_idx in range(actual_k):
                members = clusters[c_idx]
                if members:
                    dim = len(centroids[c_idx])
                    mean_vec = [
                        sum(p['normalized_vector'][d] for p in members) / len(members)
                        for d in range(dim)
                    ]
                    new_centroids.append(mean_vec)
                else:
                    # Empty cluster safeguard: reassign centroid to the point furthest from any populated centroid
                    populated = [c for i, c in enumerate(centroids) if clusters[i]]
                    furthest_point = None
                    max_min_dist = -1.0
                    for c_pop_idx in range(actual_k):
                        for p in clusters[c_pop_idx]:
                            p_vec = p['normalized_vector']
                            min_d = min(self._euclidean_distance(p_vec, pc) for pc in populated)
                            if min_d > max_min_dist:
                                max_min_dist = min_d
                                furthest_point = p_vec
                    new_c = list(furthest_point) if furthest_point is not None else list(centroids[c_idx])
                    new_centroids.append(new_c)

            # 3. Convergence check: Centroids movement < threshold
            shift = sum(
                self._euclidean_distance(old_c, new_c)
                for old_c, new_c in zip(centroids, new_centroids)
            )
            if shift < 1e-7:
                converged = True
            centroids = new_centroids

        return clusters, centroids, iteration, converged

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

        # Build point-to-cluster lookup map
        pt_cluster_map = {}
        for c_idx, members in clusters.items():
            for p in members:
                pt_cluster_map[p['patient_id']] = c_idx

        s_scores = []
        for p in features_list:
            p_id = p['patient_id']
            p_vec = p['normalized_vector']
            c_own = pt_cluster_map[p_id]
            own_members = clusters[c_own]

            if len(own_members) <= 1:
                # Single-point cluster has silhouette score of 0
                s_scores.append(0.0)
                continue

            # a(i): Mean distance to other points in the same cluster
            dist_same = [
                self._euclidean_distance(p_vec, other['normalized_vector'])
                for other in own_members if other['patient_id'] != p_id
            ]
            a_i = sum(dist_same) / len(dist_same) if dist_same else 0.0

            # b(i): Smallest mean distance to points in any other cluster
            b_i = float('inf')
            for c_other, other_members in clusters.items():
                if c_other == c_own or not other_members:
                    continue
                dist_other = [
                    self._euclidean_distance(p_vec, other['normalized_vector'])
                    for other in other_members
                ]
                mean_dist_other = sum(dist_other) / len(dist_other)
                if mean_dist_other < b_i:
                    b_i = mean_dist_other

            if math.isinf(b_i):
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

    def _evaluate_k_range(
        self,
        features_list: Optional[List[Dict[str, Any]]] = None,
        min_k: int = 2,
        max_k: int = 6,
        seed: int = 42,
        max_iters: int = 100,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Survey multiple K values in range [min_k, max_k] to provide Elbow and Silhouette curves.
        Returns:
            (k_evaluations: List of {'k': int, 'wcss': float, 'silhouette_score': float},
             recommended_k: int)
        """
        if features_list is None:
            features_list = self.extract_features()
        if 'k_min' in kwargs:
            min_k = kwargs['k_min']
        if 'k_max' in kwargs:
            max_k = kwargs['k_max']

        n_points = len(features_list)
        if n_points < 3:
            return [], min(max(1, n_points), 3)

        upper_bound = min(max_k, n_points - 1)
        lower_bound = min(min_k, upper_bound)
        candidate_ks = list(range(lower_bound, upper_bound + 1))

        k_evaluations = []
        best_k = candidate_ks[0] if candidate_ks else 3
        best_silhouette = -2.0

        for test_k in candidate_ks:
            rng = random.Random(seed)
            init_c = self._init_centroids_plusplus(features_list, test_k, rng)
            cls, cnts, _, _ = self._cluster_iteration(features_list, init_c, max_iters=max_iters)
            wcss, sil = self._calculate_metrics(cls, cnts, test_k, features_list)

            k_evaluations.append({
                'k': test_k,
                'wcss': wcss,
                'silhouette_score': sil,
            })

            # Recommend K that maximizes Silhouette Score
            if sil > best_silhouette:
                best_silhouette = sil
                best_k = test_k

        return k_evaluations, best_k

    def _denormalize_centroids(
        self, centroids: List[List[float]], min_max: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Convert normalized centroid coordinates back to clinical unit scales.
        Includes both avg_idwg_pct and avg_uf alias.
        """
        denorm = []
        for c_vec in centroids:
            c_dict = {}
            for idx, feat in enumerate(self.FEATURE_NAMES):
                val_norm = c_vec[idx]
                f_min = min_max[feat]['min']
                f_max = min_max[feat]['max']
                real_val = f_min + val_norm * (f_max - f_min)
                c_dict[feat] = round(real_val, 2)
            c_dict['avg_uf'] = c_dict['avg_idwg_pct']
            denorm.append(c_dict)
        return denorm

    def _generate_cluster_summaries(
        self,
        clusters: Dict[int, List[Dict[str, Any]]],
        denorm_centroids: List[Dict[str, float]],
        cohort_means: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Generate statistical cluster profiles comparing centroid metrics against
        both the cohort average and clinical target thresholds:
        - Target Hemoglobin: 10.0 - 11.5 g/dL (KDIGO)
        - Target Potassium: <= 5.0 mmol/L
        - Target IDWG: <= 3.5% dry weight
        """
        summaries = []

        for c_idx, centroid in enumerate(denorm_centroids):
            pts = clusters.get(c_idx, [])
            pt_count = len(pts)

            age_val = centroid.get('age', 0.0)
            hb_val = centroid.get('avg_hemoglobin', 0.0)
            k_val = centroid.get('avg_potassium', 0.0)
            idwg_val = centroid.get('avg_idwg_pct', centroid.get('avg_uf', 0.0))

            mean_age = cohort_means.get('age', 50.0)
            mean_hb = cohort_means.get('avg_hemoglobin', 11.0)
            mean_k = cohort_means.get('avg_potassium', 4.5)
            mean_idwg = cohort_means.get('avg_idwg_pct', cohort_means.get('avg_uf', 2.5))

            # Build clinical contrast descriptions
            traits = []
            if age_val > mean_age + 5.0:
                traits.append('tuổi cao hơn TB')
            elif age_val < mean_age - 5.0:
                traits.append('tuổi trẻ hơn TB')

            if hb_val < 10.0 or hb_val < mean_hb - 1.0:
                traits.append('Hb thấp (thiếu máu)')
            elif hb_val >= 11.5:
                traits.append('Hb ổn định')

            if k_val >= 5.5 or k_val > mean_k + 0.5:
                traits.append('Kali huyết cao')
            elif k_val <= 5.0:
                traits.append('Kali huyết bình thường')

            if idwg_val >= 4.5 or idwg_val > mean_idwg + 1.0:
                traits.append('tăng cân gian lọc cao (%IDWG cao)')
            elif idwg_val <= 3.0:
                traits.append('%IDWG kiểm soát tốt')

            label_traits = ', '.join(traits) if traits else 'chỉ số trung bình'
            cluster_label = f'Cụm {c_idx + 1}: {label_traits}'

            # Risk level and neutral recommendations
            if k_val >= 5.5:
                risk_level = 'HIGH'
                recommendation = (
                    f'Nhóm nguy cơ cao tăng Kali huyết (TB {k_val:.2f} mmol/L). '
                    'Khuyến nghị điều chỉnh nồng độ Kali dịch lọc (K+ bath), theo dõi sát điện tim và tư vấn chế độ ăn giảm Kali nghiêm ngặt.'
                )
            elif idwg_val >= 5.0 and k_val >= 5.2:
                risk_level = 'HIGH'
                recommendation = (
                    f'Nhóm nguy cơ tim mạch phức hợp: Tăng cân gian lọc cao (%IDWG {idwg_val:.1f}%) kết hợp Kali máu cao ({k_val:.2f} mmol/L). '
                    'Cần đánh giá lại cân nặng khô và tư vấn giảm dịch/muối đưa vào.'
                )
            elif idwg_val >= 4.5:
                risk_level = 'MEDIUM'
                recommendation = (
                    f'Nhóm tăng cân gian lọc cao (%IDWG TB {idwg_val:.1f}%), Kali ổn định ({k_val:.2f} mmol/L). '
                    'Khuyến nghị đánh giá lại cân nặng khô, kiểm soát lượng muối/nước đưa vào giữa 2 kỳ lọc.'
                )
            elif hb_val < 9.5:
                risk_level = 'MEDIUM'
                recommendation = (
                    f'Nhóm thiếu máu rõ rệt (Hb TB {hb_val:.1f} g/dL), Kali ({k_val:.2f} mmol/L) và %IDWG ({idwg_val:.1f}%) bình thường. '
                    'Khuyến nghị tối ưu hóa liều ESA (EPO), bổ sung sắt tĩnh mạch và kiểm tra dự trữ sắt (Ferritin, TSAT).'
                )
            elif hb_val < 10.5 or k_val > 5.0 or idwg_val > 3.5:
                risk_level = 'MEDIUM'
                recommendation = (
                    f'Nhóm mức độ nguy cơ trung bình: %IDWG TB {idwg_val:.1f}%, '
                    f'Kali TB {k_val:.2f} mmol/L, Hb TB {hb_val:.1f} g/dL. '
                    'Khuyến nghị theo dõi định kỳ cân nặng và xét nghiệm máu.'
                )
            else:
                risk_level = 'LOW'
                recommendation = (
                    f'Nhóm tương đối ổn định: %IDWG TB {idwg_val:.1f}%, '
                    f'Kali TB {k_val:.2f} mmol/L, Hb TB {hb_val:.1f} g/dL. '
                    'Duy trì phác đồ lọc máu và chế độ dinh dưỡng hiện tại.'
                )

            summaries.append({
                'cluster_id': c_idx,
                'cluster_label': cluster_label,
                'patient_count': pt_count,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'centroid': centroid,
            })

        return summaries

    def run_kmeans(
        self,
        k: int = 3,
        max_iters: int = 100,
        seed: int = 42,
        min_k: int = 2,
        max_k: int = 6,
    ) -> Dict[str, Any]:
        """
        Execute full pure Python K-Means workflow:
        1. Query patient records and extract 4 clinical features.
        2. Min-Max normalize features to [0.0, 1.0].
        3. Evaluate K range [min_k, max_k] using K-Means++ and compute Elbow (WCSS) & Silhouette curves.
        4. Initialize K centroids with K-Means++ (D^2 weighting) for requested K.
        5. Iteratively run Lloyd's algorithm with empty-cluster safeguards until convergence.
        6. Compute WCSS and Silhouette Score.
        7. Denormalize centroids and generate clinical profile summaries.
        8. Return comprehensive payload for UI visualization.
        """
        raw_features = self._get_patient_features()
        n_points = len(raw_features)

        if n_points == 0:
            return {
                'success': False,
                'error': 'Không có dữ liệu bệnh nhân nào để thực hiện phân cụm.',
                'patients': [],
                'clusters': {},
                'centroids': [],
                'denormalized_centroids': [],
                'min_max': {},
                'wcss': 0.0,
                'silhouette_score': 0.0,
                'cluster_summaries': [],
                'k_evaluation': [],
                'recommended_k': 3,
                'k_value': 0,
                'total_patients': 0,
            }

        # Constrain K to valid range
        actual_k = max(1, min(k, n_points))

        # Min-Max normalize
        normalized_features, min_max = self._normalize_features(raw_features)

        # Multi-K evaluation for Elbow & Silhouette curves
        k_evaluations, recommended_k = self._evaluate_k_range(
            normalized_features,
            min_k=min_k,
            max_k=max_k,
            seed=seed,
            max_iters=max_iters,
        )

        # Run K-Means for the requested K
        rng = random.Random(seed)
        initial_centroids = self._init_centroids_plusplus(normalized_features, actual_k, rng)

        clusters, final_centroids, iterations, converged = self._cluster_iteration(
            normalized_features,
            initial_centroids,
            max_iters=max_iters,
        )

        # Compute evaluation metrics
        wcss, silhouette = self._calculate_metrics(
            clusters, final_centroids, actual_k, normalized_features
        )

        # Denormalize centroids
        denorm_centroids = self._denormalize_centroids(final_centroids, min_max)

        # Calculate cohort means
        cohort_means = {}
        for feat in ['age', 'avg_hemoglobin', 'avg_potassium', 'avg_idwg_pct', 'avg_uf']:
            vals = [p['raw_features'][feat] for p in normalized_features]
            cohort_means[feat] = round(sum(vals) / len(vals), 2) if vals else 0.0

        # Generate cluster summaries
        cluster_summaries = self._generate_cluster_summaries(
            clusters, denorm_centroids, cohort_means
        )

        # Format cluster patient lists
        clusters_output = {}
        patients_output = []

        for c_idx in range(actual_k):
            members = clusters[c_idx]
            member_ids = []
            for p in members:
                p_copy = {
                    'patient_id': p['patient_id'],
                    'full_name': p['full_name'],
                    'dry_weight': p['dry_weight'],
                    'status': p.get('status', 'ACTIVE'),
                    'raw_features': p['raw_features'],
                    'features': p['raw_features'],  # Dual-compatibility for frontend
                    'normalized_vector': p['normalized_vector'],
                    'normalized_dict': p['normalized_dict'],
                    'cluster': c_idx,
                    'cluster_id': c_idx,  # Dual-compatibility for frontend
                    'age': p['raw_features'].get('age'),
                    'avg_hemoglobin': p['raw_features'].get('avg_hemoglobin'),
                    'avg_potassium': p['raw_features'].get('avg_potassium'),
                    'avg_idwg_pct': p['raw_features'].get('avg_idwg_pct'),
                    'avg_uf': p['raw_features'].get('avg_uf'),
                    'is_imputed': p['is_imputed'],
                    'imputed_fields': p['imputed_fields'],
                }
                member_ids.append(p['patient_id'])
                patients_output.append(p_copy)
            clusters_output[c_idx] = member_ids

        return {
            'success': True,
            'k_value': actual_k,
            'iterations': iterations,
            'converged': converged,
            'wcss': wcss,
            'silhouette_score': silhouette,
            'min_max': min_max,
            'centroids': final_centroids,
            'denormalized_centroids': denorm_centroids,
            'cohort_means': cohort_means,
            'cluster_summaries': cluster_summaries,
            'k_evaluation': k_evaluations,
            'recommended_k': recommended_k,
            'clusters': clusters_output,
            'patients': patients_output,
            'total_patients': n_points,
            'seed': seed,
            'disclaimer': (
                'Kết quả phân cụm hỗ trợ tham khảo chuyên môn, '
                'không thay thế chỉ định điều trị của bác sĩ chuyên khoa.'
            ),
        }
