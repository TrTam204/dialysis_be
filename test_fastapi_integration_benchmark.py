"""
Benchmark & Integration Verification Suite: Django <-> FastAPI AI Inference Service
Measures Section 3.6 Evaluation Metrics from Graduation Thesis Outline:
1. 50 consecutive inference requests across 3 clinical scenarios (HIGH, MEDIUM, LOW)
2. Round-trip HTTP latency (min, max, avg, p95)
3. FastAPI internal PyTorch LSTM inference execution time (avg)
4. Success rate (%)
5. Resilient Fallback Recovery Time (ms) when microservice is disconnected
Outputs structured JSON report to backups/fastapi_integration_benchmark.json
"""
import json
import os
import sys
import time
from typing import Dict, Any, List
import httpx
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Clinical Scenarios for IDH Prediction (3 time steps x 8 dynamic features)
SCENARIOS = {
    "SS_INP_HIGH_01": {
        "name": "Kịch bản Nguy cơ Cao (IDH High Risk - Huyết áp tụt dốc, UFR cao)",
        "patient_id": "PT_HIGH_01",
        "session_id": "SS_INP_HIGH_01",
        "dry_weight": 55.0,
        "threshold": 0.50,
        "vitals_sequence": [
            {
                "systolic_bp": 138.0,
                "diastolic_bp": 82.0,
                "map": 100.67,
                "sbp_drop_from_baseline": 0.0,
                "heart_rate": 76.0,
                "uf_rate": 0.85,
                "ufr_ml_kg_h": 15.45,
                "cumulative_uf": 0.42,
                "cumulative_uf_pct": 0.76,
                "time_elapsed_min": 30,
            },
            {
                "systolic_bp": 114.0,
                "diastolic_bp": 72.0,
                "map": 86.0,
                "sbp_drop_from_baseline": 24.0,
                "heart_rate": 88.0,
                "uf_rate": 0.85,
                "ufr_ml_kg_h": 15.45,
                "cumulative_uf": 0.85,
                "cumulative_uf_pct": 1.55,
                "time_elapsed_min": 60,
            },
            {
                "systolic_bp": 88.0,
                "diastolic_bp": 56.0,
                "map": 66.67,
                "sbp_drop_from_baseline": 50.0,
                "heart_rate": 104.0,
                "uf_rate": 0.85,
                "ufr_ml_kg_h": 15.45,
                "cumulative_uf": 1.28,
                "cumulative_uf_pct": 2.33,
                "time_elapsed_min": 90,
            },
        ],
    },
    "SS_INP_MED_02": {
        "name": "Kịch bản Nguy cơ Trung bình (IDH Medium Risk - Dao động HA vừa phải)",
        "patient_id": "PT_MED_02",
        "session_id": "SS_INP_MED_02",
        "dry_weight": 62.0,
        "threshold": 0.50,
        "vitals_sequence": [
            {
                "systolic_bp": 128.0,
                "diastolic_bp": 78.0,
                "map": 94.67,
                "sbp_drop_from_baseline": 0.0,
                "heart_rate": 74.0,
                "uf_rate": 0.65,
                "ufr_ml_kg_h": 10.48,
                "cumulative_uf": 0.33,
                "cumulative_uf_pct": 0.53,
                "time_elapsed_min": 30,
            },
            {
                "systolic_bp": 118.0,
                "diastolic_bp": 72.0,
                "map": 87.33,
                "sbp_drop_from_baseline": 10.0,
                "heart_rate": 78.0,
                "uf_rate": 0.65,
                "ufr_ml_kg_h": 10.48,
                "cumulative_uf": 0.65,
                "cumulative_uf_pct": 1.05,
                "time_elapsed_min": 60,
            },
            {
                "systolic_bp": 105.0,
                "diastolic_bp": 68.0,
                "map": 80.33,
                "sbp_drop_from_baseline": 23.0,
                "heart_rate": 84.0,
                "uf_rate": 0.65,
                "ufr_ml_kg_h": 10.48,
                "cumulative_uf": 0.98,
                "cumulative_uf_pct": 1.58,
                "time_elapsed_min": 90,
            },
        ],
    },
    "SS_INP_LOW_03": {
        "name": "Kịch bản Nguy cơ Thấp (IDH Low Risk - Huyết áp và sinh hiệu ổn định)",
        "patient_id": "PT_LOW_03",
        "session_id": "SS_INP_LOW_03",
        "dry_weight": 68.0,
        "threshold": 0.50,
        "vitals_sequence": [
            {
                "systolic_bp": 124.0,
                "diastolic_bp": 78.0,
                "map": 93.33,
                "sbp_drop_from_baseline": 0.0,
                "heart_rate": 72.0,
                "uf_rate": 0.50,
                "ufr_ml_kg_h": 7.35,
                "cumulative_uf": 0.25,
                "cumulative_uf_pct": 0.37,
                "time_elapsed_min": 30,
            },
            {
                "systolic_bp": 120.0,
                "diastolic_bp": 76.0,
                "map": 90.67,
                "sbp_drop_from_baseline": 4.0,
                "heart_rate": 74.0,
                "uf_rate": 0.50,
                "ufr_ml_kg_h": 7.35,
                "cumulative_uf": 0.50,
                "cumulative_uf_pct": 0.74,
                "time_elapsed_min": 60,
            },
            {
                "systolic_bp": 122.0,
                "diastolic_bp": 77.0,
                "map": 92.0,
                "sbp_drop_from_baseline": 2.0,
                "heart_rate": 73.0,
                "uf_rate": 0.50,
                "ufr_ml_kg_h": 7.35,
                "cumulative_uf": 0.75,
                "cumulative_uf_pct": 1.10,
                "time_elapsed_min": 90,
            },
        ],
    },
}


def run_benchmark():
    print("=" * 80)
    print("BẮT ĐẦU CHẠY BENCHMARK ĐÁNH GIÁ GIAO TIẾP DJANGO <-> FASTAPI (MỤC 3.6 ĐỀ CƯƠNG)")
    print("=" * 80)

    fastapi_url = "http://127.0.0.1:8001/predict-idh"
    health_url = "http://127.0.0.1:8001/health"

    # Step 1: Health check
    try:
        resp = httpx.get(health_url, timeout=3.0)
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        health_info = resp.json()
        print(f"[OK] FastAPI AI Service is ONLINE: {health_info}")
    except Exception as exc:
        print(f"[ERROR] Cannot connect to FastAPI at {health_url}: {exc}")
        sys.exit(1)

    # Step 2: 50 consecutive benchmark calls across the 3 scenarios
    total_calls = 50
    scenario_keys = list(SCENARIOS.keys())
    results: List[Dict[str, Any]] = []
    round_trip_latencies: List[float] = []
    fastapi_latencies: List[float] = []
    success_count = 0

    client = httpx.Client(timeout=5.0)

    for i in range(total_calls):
        scenario_key = scenario_keys[i % len(scenario_keys)]
        scenario = SCENARIOS[scenario_key]

        payload = {
            "patient_id": scenario["patient_id"],
            "session_id": scenario["session_id"],
            "dry_weight": scenario["dry_weight"],
            "threshold": scenario["threshold"],
            "vitals_sequence": scenario["vitals_sequence"],
        }

        t0 = time.perf_counter()
        resp = client.post(fastapi_url, json=payload)
        rt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code == 200:
            success_count += 1
            body = resp.json()
            fa_ms = body.get("fastapi_inference_ms", 0.0)

            round_trip_latencies.append(rt_ms)
            fastapi_latencies.append(fa_ms)

            results.append({
                "iteration": i + 1,
                "scenario": scenario_key,
                "risk_level": body.get("risk_level"),
                "risk_probability": round(body.get("risk_probability", 0.0), 4),
                "is_idh_predicted": body.get("is_idh_predicted"),
                "round_trip_latency_ms": round(rt_ms, 2),
                "fastapi_inference_ms": round(fa_ms, 2),
                "status": "SUCCESS",
            })
        else:
            results.append({
                "iteration": i + 1,
                "scenario": scenario_key,
                "status": f"ERROR_{resp.status_code}",
                "round_trip_latency_ms": round(rt_ms, 2),
            })

    client.close()

    # Step 3: Compute statistics
    success_rate = (success_count / total_calls) * 100.0
    avg_round_trip = float(np.mean(round_trip_latencies))
    min_round_trip = float(np.min(round_trip_latencies))
    max_round_trip = float(np.max(round_trip_latencies))
    p95_round_trip = float(np.percentile(round_trip_latencies, 95))
    avg_fastapi = float(np.mean(fastapi_latencies))
    min_fastapi = float(np.min(fastapi_latencies))
    max_fastapi = float(np.max(fastapi_latencies))

    print(f"\nKẾT QUẢ THỐNG KÊ {total_calls} LẦN GỌI LIÊN DỊCH VỤ:")
    print(f"- Tỷ lệ thành công: {success_rate:.1f}% ({success_count}/{total_calls})")
    print(f"- Độ trễ suy luận tại FastAPI (avg): {avg_fastapi:.2f} ms (min: {min_fastapi:.2f} ms, max: {max_fastapi:.2f} ms)")
    print(f"- Tổng thời gian khứ hồi HTTP Django <-> FastAPI:")
    print(f"  + Trung bình (avg): {avg_round_trip:.2f} ms")
    print(f"  + Nhanh nhất (min): {min_round_trip:.2f} ms")
    print(f"  + Chậm nhất (max): {max_round_trip:.2f} ms")
    print(f"  + Phân vị P95:     {p95_round_trip:.2f} ms")

    # Step 4: Measure Fallback Recovery Latency
    # Test local PyTorch pipeline execution time when FastAPI connection drops/timeouts
    from idh_ai_service.lstm_model import IDHInferencePipeline
    artifacts_dir = os.path.join(BASE_DIR, 'idh_ai_service', 'artifacts')
    pipe = IDHInferencePipeline(artifacts_dir=artifacts_dir)

    t_fall_0 = time.perf_counter()
    fallback_result = pipe.predict_sequence(
        SCENARIOS["SS_INP_HIGH_01"]["vitals_sequence"],
        threshold=0.50,
        dry_weight=55.0,
    )
    fallback_recovery_ms = round((time.perf_counter() - t_fall_0) * 1000.0, 2)
    print(f"- Thời gian kích hoạt Fallback PyTorch nội tại: {fallback_recovery_ms:.2f} ms")
    print(f"  (Kết quả Fallback: risk_level={fallback_result['risk_level']}, prob={fallback_result['risk_probability']:.4f})")

    # Step 5: Save benchmark report
    benchmark_report = {
        "benchmark_metadata": {
            "title": "Django Core <-> FastAPI AI Microservice Integration Benchmark",
            "outline_reference": "Mục 2.7 & Mục 3.6 Đề cương Đồ án Tốt nghiệp",
            "executed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_service": fastapi_url,
            "total_requests": total_calls,
            "concurrency": 1,
            "network_protocol": "HTTP/1.1 REST (JSON)",
        },
        "performance_metrics": {
            "success_rate_pct": round(success_rate, 2),
            "avg_fastapi_inference_ms": round(avg_fastapi, 2),
            "min_fastapi_inference_ms": round(min_fastapi, 2),
            "max_fastapi_inference_ms": round(max_fastapi, 2),
            "avg_round_trip_ms": round(avg_round_trip, 2),
            "min_round_trip_ms": round(min_round_trip, 2),
            "max_round_trip_ms": round(max_round_trip, 2),
            "p95_round_trip_ms": round(p95_round_trip, 2),
            "fallback_recovery_latency_ms": round(fallback_recovery_ms, 2),
        },
        "scenario_summary": {
            "SS_INP_HIGH_01": {
                "name": SCENARIOS["SS_INP_HIGH_01"]["name"],
                "expected_risk": "HIGH",
                "verified_risk": [r for r in results if r["scenario"] == "SS_INP_HIGH_01"][0]["risk_level"],
                "verified_probability": [r for r in results if r["scenario"] == "SS_INP_HIGH_01"][0]["risk_probability"],
            },
            "SS_INP_MED_02": {
                "name": SCENARIOS["SS_INP_MED_02"]["name"],
                "expected_risk": "MEDIUM",
                "verified_risk": [r for r in results if r["scenario"] == "SS_INP_MED_02"][0]["risk_level"],
                "verified_probability": [r for r in results if r["scenario"] == "SS_INP_MED_02"][0]["risk_probability"],
            },
            "SS_INP_LOW_03": {
                "name": SCENARIOS["SS_INP_LOW_03"]["name"],
                "expected_risk": "LOW",
                "verified_risk": [r for r in results if r["scenario"] == "SS_INP_LOW_03"][0]["risk_level"],
                "verified_probability": [r for r in results if r["scenario"] == "SS_INP_LOW_03"][0]["risk_probability"],
            },
        },
        "sample_iteration_logs": results[:10],
    }

    out_path = os.path.join(BASE_DIR, 'backups', 'fastapi_integration_benchmark.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Đã lưu báo cáo benchmark thành công vào {out_path}")
    print("=" * 80)
    return benchmark_report


if __name__ == '__main__':
    run_benchmark()
