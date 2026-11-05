"""
TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) Service
LocMau Dialysis Center Management & Decision Support System
Author: Nguyen Ho Truong Tam
Thesis Topic: Xay dung he thong quan ly va ho tro ra quyet dinh phuc vu trung tam loc mau

Self-implemented pure Python implementation (No external TOPSIS libraries).
Reference: Hwang, C. L., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications.
"""

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from django.utils import timezone


class TopsisEngine:
    """
    Pure Python TOPSIS computational engine.
    Independent of Django ORM and external libraries for high testability and mathematical rigor.
    """

    CRITERION_BENEFIT = "benefit"
    CRITERION_COST = "cost"

    @classmethod
    def evaluate(
        cls,
        matrix: List[List[float]],
        weights: List[float],
        criteria_types: List[str],
        alternative_ids: Optional[List[str]] = None,
        criteria_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete TOPSIS evaluation with vector normalization.

        Parameters:
        - matrix: 2D list of shape (m, n) containing raw decision values
        - weights: list of length n with positive weights
        - criteria_types: list of length n with 'benefit' or 'cost'
        - alternative_ids: optional list of length m with labels for alternatives
        - criteria_names: optional list of length n with labels for criteria

        Returns:
        - Dictionary containing:
          - rankings: sorted list of alternatives with rank, id, score (C_i), D+, D-
          - normalized_matrix: R = [r_ij]
          - weighted_matrix: V = [v_ij]
          - ideal_positive: A+ = [v_j+]
          - ideal_negative: A- = [v_j-]
          - closeness_scores: [C_i]
          - distances_positive: [D_i+]
          - distances_negative: [D_i-]
        """
        # --- 1. VALIDATION ---
        m = len(matrix)
        if m == 0:
            raise ValueError("Ma trận quyết định không được để trống (cần ít nhất 1 phương án).")

        n = len(matrix[0])
        if n == 0:
            raise ValueError("Ma trận quyết định không có tiêu chí nào.")

        for i, row in enumerate(matrix):
            if len(row) != n:
                raise ValueError(f"Hàng {i} có {len(row)} phần tử, không khớp số tiêu chí {n}.")
            for j, val in enumerate(row):
                if val is None or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Giá trị tại ({i}, {j}) không hợp lệ (None/NaN/Inf).")

        if len(weights) != n:
            raise ValueError(f"Số lượng trọng số ({len(weights)}) không khớp số tiêu chí ({n}).")

        if any(w < 0 for w in weights):
            raise ValueError("Trọng số của tiêu chí không được mang giá trị âm.")

        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Tổng trọng số của các tiêu chí phải lớn hơn 0.")

        if len(criteria_types) != n:
            raise ValueError(f"Số lượng kiểu tiêu chí ({len(criteria_types)}) không khớp ({n}).")

        norm_criteria_types = []
        for ct in criteria_types:
            ct_lower = str(ct).strip().lower()
            if ct_lower in ("benefit", "+", "max", "true", "1"):
                norm_criteria_types.append(cls.CRITERION_BENEFIT)
            elif ct_lower in ("cost", "-", "min", "false", "0"):
                norm_criteria_types.append(cls.CRITERION_COST)
            else:
                raise ValueError(f"Kiểu tiêu chí không hợp lệ: '{ct}'. Phải là 'benefit' hoặc 'cost'.")

        alt_ids = alternative_ids if alternative_ids and len(alternative_ids) == m else [f"A{i+1}" for i in range(m)]
        crit_names = criteria_names if criteria_names and len(criteria_names) == n else [f"C{j+1}" for j in range(n)]

        # --- 2. NORMALIZE WEIGHTS ---
        norm_weights = [w / total_weight for w in weights]

        # --- 3. VECTOR NORMALIZATION ---
        # r_ij = x_ij / sqrt(sum_k (x_kj^2))
        col_norms = []
        for j in range(n):
            sum_sq = sum(matrix[i][j] ** 2 for i in range(m))
            col_norms.append(math.sqrt(sum_sq))

        norm_matrix = []
        for i in range(m):
            norm_row = []
            for j in range(n):
                if col_norms[j] == 0:
                    norm_row.append(0.0)
                else:
                    norm_row.append(matrix[i][j] / col_norms[j])
            norm_matrix.append(norm_row)

        # --- 4. WEIGHTED NORMALIZED DECISION MATRIX ---
        # v_ij = w_j * r_ij
        weighted_matrix = []
        for i in range(m):
            weighted_row = [norm_matrix[i][j] * norm_weights[j] for j in range(n)]
            weighted_matrix.append(weighted_row)

        # --- 5. DETERMINE IDEAL SOLUTIONS (A+ and A-) ---
        ideal_pos = []
        ideal_neg = []
        for j in range(n):
            col_vals = [weighted_matrix[i][j] for i in range(m)]
            if norm_criteria_types[j] == cls.CRITERION_BENEFIT:
                ideal_pos.append(max(col_vals))
                ideal_neg.append(min(col_vals))
            else:  # COST
                ideal_pos.append(min(col_vals))
                ideal_neg.append(max(col_vals))

        # --- 6. CALCULATE SEPARATION MEASURES (Euclidean distances D+ and D-) ---
        dist_pos = []
        dist_neg = []
        for i in range(m):
            d_plus_sq = sum((weighted_matrix[i][j] - ideal_pos[j]) ** 2 for j in range(n))
            d_minus_sq = sum((weighted_matrix[i][j] - ideal_neg[j]) ** 2 for j in range(n))
            dist_pos.append(math.sqrt(d_plus_sq))
            dist_neg.append(math.sqrt(d_minus_sq))

        # --- 7. CALCULATE RELATIVE CLOSENESS COEFFICIENTS (C_i) ---
        # C_i = D_i^- / (D_i^+ + D_i^-)
        closeness = []
        for i in range(m):
            denom = dist_pos[i] + dist_neg[i]
            if denom == 0:
                # Edge case: alternative is simultaneously at ideal pos and neg (e.g. all alternatives identical or m=1)
                score = 0.5 if m > 1 else 1.0
            else:
                score = dist_neg[i] / denom
            closeness.append(score)

        # --- 8. RANKING (Descending by C_i) ---
        # Deterministic tie-breaking by original index
        indexed_scores = [(i, closeness[i]) for i in range(m)]
        indexed_scores.sort(key=lambda item: (-item[1], item[0]))

        rankings = []
        for rank_idx, (orig_idx, score) in enumerate(indexed_scores, start=1):
            rankings.append({
                "rank": rank_idx,
                "alternative_id": alt_ids[orig_idx],
                "original_index": orig_idx,
                "closeness_score": round(score, 6),
                "distance_positive": round(dist_pos[orig_idx], 6),
                "distance_negative": round(dist_neg[orig_idx], 6),
                "raw_values": matrix[orig_idx],
            })

        return {
            "m": m,
            "n": n,
            "criteria_names": crit_names,
            "criteria_types": norm_criteria_types,
            "weights": [round(w, 6) for w in norm_weights],
            "ideal_positive": [round(v, 6) for v in ideal_pos],
            "ideal_negative": [round(v, 6) for v in ideal_neg],
            "normalized_matrix": [[round(v, 6) for v in row] for row in norm_matrix],
            "weighted_matrix": [[round(v, 6) for v in row] for row in weighted_matrix],
            "distances_positive": [round(d, 6) for d in dist_pos],
            "distances_negative": [round(d, 6) for d in dist_neg],
            "closeness_scores": [round(c, 6) for c in closeness],
            "rankings": rankings,
        }


class TopsisTriageService:
    """
    Business service bridging Django ORM patient records and the pure TopsisEngine.
    Implements department scoping, robust data extraction, missing data imputation,
    and clinical disclaimer tagging.
    """

    # Default criteria catalogue for Hemodialysis Triage Simulation (V1 DEMO)
    CRITERIA_CATALOGUE = [
        {
            "code": "C1_POTASSIUM",
            "name": "Nồng độ Kali máu (K+)",
            "unit": "mmol/L",
            "type": TopsisEngine.CRITERION_BENEFIT,
            "weight": 0.30,
            "default_value": 4.5,
            "min_val": 2.5,
            "max_val": 9.0,
            "description": "Chỉ số Kali huyết thanh từ xét nghiệm máu gần nhất. Tăng kali máu (> 5.5 mmol/L) làm tăng nguy cơ rối loạn nhịp tim và tử vong, cần ưu tiên lọc máu sớm.",
        },
        {
            "code": "C2_HEMOGLOBIN",
            "name": "Mức Hemoglobin (Hb)",
            "unit": "g/dL",
            "type": TopsisEngine.CRITERION_COST,
            "weight": 0.20,
            "default_value": 11.0,
            "min_val": 4.0,
            "max_val": 18.0,
            "description": "Chỉ số Hemoglobin gần nhất. Thiếu máu nặng (Hb thấp) ở bệnh nhân suy thận giai đoạn cuối làm giảm oxy mô và tăng gánh nặng tim mạch.",
        },
        {
            "code": "C3_DAYS_SINCE_SESSION",
            "name": "Số ngày từ phiên lọc cuối",
            "unit": "ngày",
            "type": TopsisEngine.CRITERION_BENEFIT,
            "weight": 0.25,
            "default_value": 3.0,
            "min_val": 0.0,
            "max_val": 14.0,
            "description": "Khoảng thời gian tính bằng ngày kể từ phiên lọc máu hoàn thành gần nhất. Số ngày cách quãng càng lớn, tích lũy độc tố uremic càng cao.",
        },
        {
            "code": "C4_FLUID_OVERLOAD",
            "name": "Tải lượng thừa dịch ước tính (IDWG)",
            "unit": "kg",
            "type": TopsisEngine.CRITERION_BENEFIT,
            "weight": 0.15,
            "default_value": 1.5,
            "min_val": 0.0,
            "max_val": 10.0,
            "description": "Mức tăng cân giữa hai kỳ lọc (Interdialytic Weight Gain = Cân nặng đầu phiên - Trọng lượng khô). Tăng trên 3kg có nguy cơ phù phổi cấp.",
        },
        {
            "code": "C5_DISTANCE_KM",
            "name": "Khoảng cách địa lý đến cơ sở",
            "unit": "km",
            "type": TopsisEngine.CRITERION_COST,
            "weight": 0.10,
            "default_value": 5.0,
            "min_val": 0.1,
            "max_val": 100.0,
            "description": "Khoảng cách trắc địa từ tọa độ bệnh nhân (PostGIS Point) đến trung tâm/khoa lọc máu Bệnh viện Chợ Rẫy TP.HCM (10.7556, 106.6602). Khoảng cách gần hơn giúp bệnh nhân dễ tiếp cận ca cấp cứu nhanh hơn.",
        },
    ]

    # Reference coordinates for Dialysis Center (Bệnh viện Chợ Rẫy TP.HCM: 10.7556° N, 106.6602° E)
    DEFAULT_CENTER_LAT = 10.7556
    DEFAULT_CENTER_LON = 106.6602

    @classmethod
    def get_criteria_catalogue(cls) -> List[Dict[str, Any]]:
        """Returns the criteria catalogue with metadata and descriptions."""
        return cls.CRITERIA_CATALOGUE

    @classmethod
    def calculate_distance_km(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine formula to calculate geodesic distance in km."""
        r = 6371.0  # Earth radius in kilometers
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    @classmethod
    def extract_patient_features(
        cls,
        patient: Any,
        center_lat: float = DEFAULT_CENTER_LAT,
        center_lon: float = DEFAULT_CENTER_LON,
        now: Optional[datetime] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Extracts the 5 numerical criteria for a single patient with safe imputation.
        """
        now = now or timezone.now()

        # 1. Potassium (C1) & Hemoglobin (C2) from latest BloodSample
        latest_sample = (
            patient.blood_samples.all().order_by("-collection_date").first()
            if hasattr(patient, "blood_samples")
            else None
        )
        if latest_sample:
            k_val = float(latest_sample.potassium_level)
            hb_val = float(latest_sample.hemoglobin_level)
            sample_date = latest_sample.collection_date.strftime("%d/%m/%Y")
        else:
            k_val = 4.5
            hb_val = 11.0
            sample_date = None

        # 2. Days since last session (C3) & Fluid Overload (C4) from latest completed/in-progress session
        latest_session = (
            patient.dialysis_sessions.exclude(status="CANCELLED")
            .order_by("-scheduled_start")
            .first()
            if hasattr(patient, "dialysis_sessions")
            else None
        )

        if latest_session and latest_session.scheduled_start:
            delta = now - latest_session.scheduled_start
            days_since = max(0.0, round(delta.total_seconds() / 86400.0, 1))
            session_date = latest_session.scheduled_start.strftime("%d/%m/%Y")

            # Fluid overload calculation
            if latest_session.pre_weight and patient.dry_weight:
                fluid_overload = max(0.0, round(latest_session.pre_weight - patient.dry_weight, 2))
            elif latest_session.uf_target:
                fluid_overload = max(0.0, round(float(latest_session.uf_target), 2))
            else:
                fluid_overload = 1.5
        else:
            days_since = 3.0
            fluid_overload = 1.5
            session_date = None

        # 3. Geographic Distance (C5) via PostGIS location
        if patient.location:
            # PostGIS Point: location.x is lon, location.y is lat
            pat_lon = patient.location.x
            pat_lat = patient.location.y
            dist_km = round(cls.calculate_distance_km(pat_lat, pat_lon, center_lat, center_lon), 2)
            has_location = True
        else:
            dist_km = 5.0
            has_location = False

        raw_vector = [k_val, hb_val, days_since, fluid_overload, dist_km]
        details = {
            "potassium_level": k_val,
            "hemoglobin_level": hb_val,
            "blood_sample_date": sample_date,
            "days_since_dialysis": days_since,
            "last_session_date": session_date,
            "fluid_overload_kg": fluid_overload,
            "distance_km": dist_km,
            "has_location": has_location,
        }
        return raw_vector, details

    @classmethod
    def rank_department_patients(
        cls,
        department_id: Optional[int],
        user: Any,
        custom_weights: Optional[Dict[str, float]] = None,
        patient_ids: Optional[List[str]] = None,
        center_coords: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes TOPSIS triage ranking for patients of a department with RBAC enforcement.
        """
        from core.models import Patient, Department

        # --- RBAC ENFORCEMENT ---
        if user.role == "DOCTOR":
            if not user.department_id:
                raise PermissionError("Bác sĩ chưa được gán khoa quản lý.")
            if department_id and int(department_id) != user.department_id:
                raise PermissionError("Bác sĩ chỉ có quyền xếp hạng bệnh nhân trong khoa của mình.")
            effective_dept_id = user.department_id
        elif user.role == "NURSE":
            if not user.department_id:
                raise PermissionError("Điều dưỡng chưa được gán khoa quản lý.")
            if department_id and int(department_id) != user.department_id:
                raise PermissionError("Điều dưỡng chỉ có quyền xem xếp hạng bệnh nhân trong khoa của mình.")
            effective_dept_id = user.department_id
        else:
            # ADMIN can query specific department or all
            effective_dept_id = department_id

        # Query patients
        qs = Patient.objects.all().select_related("department").prefetch_related("blood_samples", "dialysis_sessions")
        if effective_dept_id:
            qs = qs.filter(department_id=effective_dept_id)

        # Exclude discharged patients
        qs = qs.exclude(status=Patient.Status.DISCHARGED).order_by("patient_id")

        if patient_ids:
            qs = qs.filter(patient_id__in=patient_ids)

        patients = list(qs)
        if not patients:
            dept_name = ""
            if effective_dept_id:
                dept_obj = Department.objects.filter(id=effective_dept_id).first()
                dept_name = f" thuộc khoa '{dept_obj.name}'" if dept_obj else ""
            return {
                "department_id": effective_dept_id,
                "total_patients": 0,
                "rankings": [],
                "detail": f"Không có bệnh nhân hợp lệ nào{dept_name} để xếp hạng TOPSIS.",
                "is_clinical_approved": False,
                "disclaimer": "Kết quả tính toán mô phỏng kỹ thuật (DEMO), chưa dùng cho quyết định lâm sàng thực tế.",
            }

        # Build weights and criteria types
        catalog = cls.CRITERIA_CATALOGUE
        weights = []
        criteria_types = []
        criteria_names = []
        for crit in catalog:
            code = crit["code"]
            criteria_types.append(crit["type"])
            criteria_names.append(crit["name"])
            if custom_weights and code in custom_weights:
                weights.append(float(custom_weights[code]))
            else:
                weights.append(float(crit["weight"]))

        center_lat = center_coords[0] if center_coords else cls.DEFAULT_CENTER_LAT
        center_lon = center_coords[1] if center_coords else cls.DEFAULT_CENTER_LON

        now = timezone.now()
        matrix = []
        patient_detail_map = {}
        alt_ids = []

        for p in patients:
            raw_vec, details = cls.extract_patient_features(p, center_lat, center_lon, now=now)
            matrix.append(raw_vec)
            alt_ids.append(p.patient_id)
            patient_detail_map[p.patient_id] = {
                "patient_id": p.patient_id,
                "full_name": p.full_name,
                "date_of_birth": p.date_of_birth.isoformat() if p.date_of_birth else None,
                "gender": p.gender,
                "status": p.status,
                "treatment_pattern": p.treatment_pattern,
                "preferred_shift": p.preferred_shift,
                "department_id": p.department_id,
                "department_name": p.department.name if p.department else None,
                **details,
            }

        # Run TopsisEngine
        eval_result = TopsisEngine.evaluate(
            matrix=matrix,
            weights=weights,
            criteria_types=criteria_types,
            alternative_ids=alt_ids,
            criteria_names=criteria_names,
        )

        # Merge patient info into rankings
        enriched_rankings = []
        for item in eval_result["rankings"]:
            pid = item["alternative_id"]
            p_info = patient_detail_map.get(pid, {})
            enriched_rankings.append({
                **item,
                "patient": p_info,
            })

        dept_obj = Department.objects.filter(id=effective_dept_id).first() if effective_dept_id else None

        return {
            "department_id": effective_dept_id,
            "department_name": dept_obj.name if dept_obj else "Tất cả khoa (Trung tâm)",
            "calculated_at": timezone.now().isoformat(),
            "total_patients": len(patients),
            "criteria": [
                {
                    "code": c["code"],
                    "name": c["name"],
                    "unit": c["unit"],
                    "type": c["type"],
                    "weight": eval_result["weights"][idx],
                    "description": c["description"],
                }
                for idx, c in enumerate(catalog)
            ],
            "ideal_positive": eval_result["ideal_positive"],
            "ideal_negative": eval_result["ideal_negative"],
            "rankings": enriched_rankings,
            "is_clinical_approved": False,
            "disclaimer": (
                "CHÚ Ý CHUYÊN MÔN: Danh sách xếp hạng này chỉ là công cụ hỗ trợ ra quyết định (Decision Support Simulation) "
                "dựa trên thuật toán đa tiêu chí TOPSIS và dữ liệu mẫu kỹ thuật (DEMO). "
                "Kết quả này CHƯA ĐƯỢC THẨM ĐỊNH LÂM SÀNG và tuyệt đối KHÔNG THAY THẾ quyết định của Bác sĩ điều trị."
            ),
        }
