import os
import sys
import json
import csv
import time
import math

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dialysis_system.settings')

import django
django.setup()

from core.topsis_service import TopsisEngine

def run_benchmarks():
    print("Running TOPSIS benchmark calculations...")

    # Criteria definition
    criteria = [
        {"code": "C1", "name": "Potassium (K+)", "unit": "mEq/L", "direction": "benefit", "weight": 0.30},
        {"code": "C2", "name": "Hemoglobin (Hb)", "unit": "g/dL", "direction": "cost", "weight": 0.20},
        {"code": "C3", "name": "Days Since Dialysis", "unit": "days", "direction": "benefit", "weight": 0.25},
        {"code": "C4", "name": "Fluid Overload (IDWG)", "unit": "kg", "direction": "benefit", "weight": 0.15},
        {"code": "C5", "name": "Travel Distance", "unit": "km", "direction": "cost", "weight": 0.10},
    ]
    weights = [c["weight"] for c in criteria]
    directions = [c["direction"] for c in criteria]

    # Benchmark Dataset: 12 diverse hemodialysis patients
    cohort = [
        {"id": "BN001", "name": "Nguyen Van An", "dept": "Khoa A", "raw": [6.2, 8.2, 4.0, 3.8, 4.5]},
        {"id": "BN002", "name": "Tran Thi Binh", "dept": "Khoa A", "raw": [5.5, 9.1, 3.0, 2.7, 8.2]},
        {"id": "BN003", "name": "Le Hoang Cuong", "dept": "Khoa A", "raw": [4.8, 11.5, 2.0, 1.5, 12.0]},
        {"id": "BN004", "name": "Pham Minh Duc", "dept": "Khoa A", "raw": [5.9, 7.8, 3.0, 4.1, 3.1]},
        {"id": "BN005", "name": "Vo Thi Em", "dept": "Khoa A", "raw": [4.2, 12.0, 1.0, 0.8, 15.4]},
        {"id": "BN006", "name": "Dang Van Giang", "dept": "Khoa B", "raw": [6.5, 8.0, 4.0, 3.5, 6.0]},
        {"id": "BN007", "name": "Bui Thi Hoa", "dept": "Khoa B", "raw": [5.1, 10.2, 2.0, 2.0, 7.5]},
        {"id": "BN008", "name": "Hoang Van Khoi", "dept": "Khoa B", "raw": [4.6, 10.8, 2.0, 1.9, 9.2]},
        {"id": "BN009", "name": "Ngo Thi Lan", "dept": "Khoa B", "raw": [5.7, 8.6, 3.0, 3.2, 5.8]},
        {"id": "BN010", "name": "Doan Minh Nam", "dept": "Khoa B", "raw": [6.0, 7.5, 4.0, 4.0, 2.5]},
        {"id": "BN011", "name": "Truong Thi Oanh", "dept": "Khoa A", "raw": [4.9, 11.0, 2.0, 1.8, 10.5]},
        {"id": "BN012", "name": "Vu Van Phuc", "dept": "Khoa B", "raw": [5.3, 9.5, 3.0, 2.5, 11.0]},
    ]

    matrix_X = [p["raw"] for p in cohort]
    ids = [p["id"] for p in cohort]

    # Benchmark computation & timing
    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        res = TopsisEngine.evaluate(matrix_X, weights, directions, alternative_ids=ids)
    t1 = time.perf_counter()
    avg_runtime_ms = ((t1 - t0) / iterations) * 1000.0

    print(f"Computed TOPSIS for {len(cohort)} patients in {avg_runtime_ms:.4f} ms (averaged over {iterations} runs)")

    matrix_R = res["normalized_matrix"]
    matrix_V = res["weighted_matrix"]
    w_norm = res["weights"]
    ideal_pos = res["ideal_positive"]
    ideal_neg = res["ideal_negative"]

    # Compute column Euclidean norms
    m = len(matrix_X)
    n = len(weights)
    col_norms = [math.sqrt(sum(matrix_X[i][j] ** 2 for i in range(m))) for j in range(n)]

    # Combine benchmark results
    patient_map = {p["id"]: p for p in cohort}

    ranked_items = []
    for item in res["rankings"]:
        p = patient_map[item["alternative_id"]]
        orig_idx = item["original_index"]
        ranked_items.append({
            "rank": item["rank"],
            "patient_id": item["alternative_id"],
            "patient_name": p["name"],
            "department": p["dept"],
            "closeness_score": round(item["closeness_score"], 6),
            "distance_positive": round(item["distance_positive"], 6),
            "distance_negative": round(item["distance_negative"], 6),
            "raw_values": {c["code"]: p["raw"][j] for j, c in enumerate(criteria)},
            "normalized_values": {c["code"]: round(matrix_R[orig_idx][j], 6) for j, c in enumerate(criteria)},
            "weighted_values": {c["code"]: round(matrix_V[orig_idx][j], 6) for j, c in enumerate(criteria)},
        })

    benchmark_data = {
        "benchmark_metadata": {
            "algorithm": "Pure Python TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)",
            "implementation": "core.topsis_service.TopsisEngine",
            "execution_environment": "Python 3.12 / Django 5.2.x on Windows",
            "is_clinical_approved": False,
            "status": "TECHNICAL DEMO / CLINICAL CRITERIA PENDING",
            "total_alternatives": len(cohort),
            "num_criteria": len(criteria),
            "benchmark_iterations": iterations,
            "average_execution_time_ms": round(avg_runtime_ms, 4),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "criteria_definition": criteria,
        "normalized_weights": [round(w, 4) for w in w_norm],
        "column_norms": [round(norm, 6) for norm in col_norms],
        "ideal_solutions": {
            "positive_ideal_A_plus": [round(v, 6) for v in ideal_pos],
            "negative_ideal_A_minus": [round(v, 6) for v in ideal_neg],
        },
        "rankings": ranked_items,
    }

    # Write JSON artifact
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backups", "topsis_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    print(f"Exported JSON benchmark to {json_path}")

    # Write CSV artifact
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backups", "topsis_benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Patient_ID", "Patient_Name", "Department",
            "Closeness_Score_Ci", "Distance_Positive_D+", "Distance_Negative_D-",
            "K+_mEq_L", "Hb_g_dL", "Days_Since_Dialysis", "IDWG_kg", "Distance_km",
            "Weighted_C1", "Weighted_C2", "Weighted_C3", "Weighted_C4", "Weighted_C5"
        ])
        for item in ranked_items:
            orig_idx = [p["id"] for p in cohort].index(item["patient_id"])
            writer.writerow([
                item["rank"],
                item["patient_id"],
                item["patient_name"],
                item["department"],
                item["closeness_score"],
                item["distance_positive"],
                item["distance_negative"],
                item["raw_values"]["C1"],
                item["raw_values"]["C2"],
                item["raw_values"]["C3"],
                item["raw_values"]["C4"],
                item["raw_values"]["C5"],
                item["weighted_values"]["C1"],
                item["weighted_values"]["C2"],
                item["weighted_values"]["C3"],
                item["weighted_values"]["C4"],
                item["weighted_values"]["C5"],
            ])
    print(f"Exported CSV benchmark to {csv_path}")

if __name__ == "__main__":
    run_benchmarks()
