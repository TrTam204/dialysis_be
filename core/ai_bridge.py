"""
AI Bridge Service: Bridges Django Core Models with FastAPI LSTM IDH Prediction Service.
Handles clinical 8-feature vital signs sequence aggregation, midpoint interpolation for 2-point sequences,
and FastAPI REST communication with resilient automatic local PyTorch engine fallback.
"""
from typing import Any, Dict, List
import json
import logging
import os
import uuid
import httpx
from django.conf import settings
from django.utils import timezone

from idh_ai_service.lstm_model import IDHInferencePipeline
from core.models import DialysisSession, VitalSign

logger = logging.getLogger(__name__)

# Cached local inference pipeline for in-process fallback
_local_pipeline: IDHInferencePipeline = None


def get_local_pipeline() -> IDHInferencePipeline:
    global _local_pipeline
    if _local_pipeline is None:
        artifacts_dir = os.path.join(settings.BASE_DIR, 'idh_ai_service', 'artifacts')
        _local_pipeline = IDHInferencePipeline(artifacts_dir=artifacts_dir)
    return _local_pipeline


def build_vitals_sequence(session: DialysisSession, vitals_qs=None) -> Dict[str, Any]:
    """
    Construct chronological 8-dimensional time-series sequence from session vital signs.
    Requires at least 2 recorded vital signs (recorded_count >= 2) to capture temporal delta.
    - If recorded_count < 2: returns INSUFFICIENT_DATA.
    - If recorded_count == 2: creates smooth 3-step sequence using linear midpoint interpolation.
    - If recorded_count >= 3: uses actual recorded chronological steps.
    """
    if vitals_qs is None:
        vitals_qs = session.vital_signs.all().order_by('recorded_at')

    vitals = list(vitals_qs)
    recorded_count = len(vitals)

    if recorded_count < 2:
        return {
            'has_prediction': False,
            'status': 'INSUFFICIENT_DATA',
            'message': 'Cần tối thiểu 2 lần đo sinh hiệu trong phiên lọc để phân tích động học biến thiên huyết áp và dự báo sớm IDH.',
            'vitals_sequence': [],
            'is_interpolated': False,
            'recorded_count': recorded_count,
        }

    # Reference start time
    base_time = vitals[0].recorded_at or session.actual_start or session.scheduled_start

    # Resolve dry weight (kg)
    patient = getattr(session, 'patient', None)
    dry_w = float(patient.dry_weight) if patient and patient.dry_weight and patient.dry_weight > 0 else float(session.pre_weight or 60.0)
    if dry_w <= 0:
        dry_w = 60.0

    # Estimated UF target and hourly UF rate
    session_uf = float(session.uf_target or session.uf_actual or 2.4)
    duration_hours = 4.0
    if session.scheduled_start and session.scheduled_end:
        diff_hours = (session.scheduled_end - session.scheduled_start).total_seconds() / 3600.0
        if diff_hours > 0.5:
            duration_hours = diff_hours
    hourly_ufr = round(session_uf / duration_hours, 2)  # L/hour
    ufr_ml_kg_h = round((hourly_ufr * 1000.0) / dry_w, 2)  # mL/kg/h

    base_sbp = float(vitals[0].systolic_bp if vitals[0].systolic_bp is not None else 120.0)

    raw_steps = []
    for idx, vs in enumerate(vitals):
        sbp = float(vs.systolic_bp if vs.systolic_bp is not None else 120.0)
        dbp = float(vs.diastolic_bp if vs.diastolic_bp is not None else 75.0)
        hr = float(vs.heart_rate if vs.heart_rate is not None else 78.0)
        map_val = round(dbp + (sbp - dbp) / 3.0, 2)
        sbp_drop = round(base_sbp - sbp, 2)

        if vs.recorded_at and base_time:
            t_min = int(max(0, (vs.recorded_at - base_time).total_seconds() / 60.0))
        else:
            t_min = idx * 30

        cum_uf = round(min(session_uf, hourly_ufr * (t_min / 60.0)), 2)
        cum_uf_pct = round((cum_uf / dry_w) * 100.0, 2)

        raw_steps.append({
            'systolic_bp': sbp,
            'diastolic_bp': dbp,
            'map': map_val,
            'sbp_drop_from_baseline': sbp_drop,
            'heart_rate': hr,
            'uf_rate': hourly_ufr,
            'ufr_ml_kg_h': ufr_ml_kg_h,
            'cumulative_uf': cum_uf,
            'cumulative_uf_pct': cum_uf_pct,
            'time_elapsed_min': t_min,
            'dry_weight': dry_w,
        })

    is_interpolated = False
    if recorded_count == 2:
        is_interpolated = True
        # Linear midpoint interpolation between step 0 and step 1
        s0, s1 = raw_steps[0], raw_steps[1]
        mid_step = {
            'systolic_bp': round((s0['systolic_bp'] + s1['systolic_bp']) / 2.0, 2),
            'diastolic_bp': round((s0['diastolic_bp'] + s1['diastolic_bp']) / 2.0, 2),
            'map': round((s0['map'] + s1['map']) / 2.0, 2),
            'sbp_drop_from_baseline': round((s0['sbp_drop_from_baseline'] + s1['sbp_drop_from_baseline']) / 2.0, 2),
            'heart_rate': round((s0['heart_rate'] + s1['heart_rate']) / 2.0, 2),
            'uf_rate': s1['uf_rate'],
            'ufr_ml_kg_h': s1['ufr_ml_kg_h'],
            'cumulative_uf': round((s0['cumulative_uf'] + s1['cumulative_uf']) / 2.0, 2),
            'cumulative_uf_pct': round((s0['cumulative_uf_pct'] + s1['cumulative_uf_pct']) / 2.0, 2),
            'time_elapsed_min': int((s0['time_elapsed_min'] + s1['time_elapsed_min']) / 2),
            'dry_weight': dry_w,
        }
        sequence_steps = [s0, mid_step, s1]
    else:
        sequence_steps = list(raw_steps)

    # Ensure non-decreasing time_elapsed_min
    for i in range(1, len(sequence_steps)):
        sequence_steps[i]['time_elapsed_min'] = max(
            sequence_steps[i]['time_elapsed_min'],
            sequence_steps[i - 1]['time_elapsed_min'],
        )

    return {
        'has_prediction': True,
        'status': 'READY',
        'message': 'Đủ dữ liệu sinh hiệu để phân tích dự báo.',
        'vitals_sequence': sequence_steps,
        'is_interpolated': is_interpolated,
        'is_padded': is_interpolated,  # Backward compatibility flag
        'recorded_count': recorded_count,
        'dry_weight': dry_w,
    }


def predict_idh_for_session(session: DialysisSession) -> Dict[str, Any]:
    """
    Predict IDH risk for a DialysisSession.
    Attempts FastAPI microservice call; falls back to in-process PyTorch pipeline if unreachable.
    """
    seq_data = build_vitals_sequence(session)

    if not seq_data['has_prediction']:
        seq_data['model_version'] = '1.0.0'
        seq_data['predicted_at'] = timezone.now().isoformat()
        seq_data['disclaimer'] = (
            'Cảnh báo từ mô hình AI chỉ mang tính chất hỗ trợ theo dõi sớm, '
            'không thay thế quyết định chuyên môn của bác sĩ.'
        )
        return seq_data

    dry_w = seq_data.get('dry_weight', 60.0)
    payload = {
        'patient_id': session.patient.patient_id if session.patient else None,
        'session_id': session.session_id,
        'dry_weight': dry_w,
        'threshold': 0.50,
        'vitals_sequence': seq_data['vitals_sequence'],
    }

    # Attempt 1: Call FastAPI Microservice
    service_url = getattr(settings, 'FASTAPI_IDH_SERVICE_URL', 'http://127.0.0.1:8001')
    endpoint = f"{service_url.rstrip('/')}/predict-idh"

    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.post(endpoint, json=payload)
            if resp.status_code == 200:
                result = resp.json()
                result['has_prediction'] = True
                result['status'] = 'READY'
                result['inference_source'] = 'FASTAPI_MICROSERVICE'
                result['is_interpolated'] = seq_data.get('is_interpolated', False)
                result['is_padded'] = seq_data.get('is_padded', False)
                result['recorded_vitals_count'] = seq_data['recorded_count']
                result['dry_weight'] = dry_w
                result['model_version'] = '1.0.0'
                result['predicted_at'] = timezone.now().isoformat()
                result['disclaimer'] = (
                    'Cảnh báo từ mô hình AI chỉ mang tính chất hỗ trợ theo dõi sớm, '
                    'không thay thế quyết định chuyên môn của bác sĩ.'
                )
                return result
    except Exception as exc:
        logger.warning(f"FastAPI IDH service unavailable ({str(exc)}), falling back to local LSTM engine.")

    # Attempt 2: Fallback to in-process PyTorch LSTM inference pipeline
    pipeline = get_local_pipeline()
    prediction = pipeline.predict_sequence(
        seq_data['vitals_sequence'],
        threshold=0.50,
        dry_weight=dry_w,
    )

    return {
        'has_prediction': True,
        'status': 'READY',
        'prediction_id': str(uuid.uuid4()),
        'patient_id': session.patient.patient_id if session.patient else None,
        'session_id': session.session_id,
        'risk_probability': prediction['risk_probability'],
        'risk_level': prediction['risk_level'],
        'is_idh_predicted': prediction['is_idh_predicted'],
        'threshold_used': prediction['threshold_used'],
        'contributing_factors': prediction['contributing_factors'],
        'clinical_recommendations': prediction['clinical_recommendations'],
        'sequence_length_received': prediction['sequence_length_received'],
        'inference_source': 'LOCAL_LSTM_ENGINE',
        'is_interpolated': seq_data.get('is_interpolated', False),
        'is_padded': seq_data.get('is_padded', False),
        'recorded_vitals_count': seq_data['recorded_count'],
        'dry_weight': dry_w,
        'model_version': '1.0.0',
        'predicted_at': timezone.now().isoformat(),
        'disclaimer': (
            'Cảnh báo từ mô hình AI chỉ mang tính chất hỗ trợ theo dõi sớm, '
            'không thay thế quyết định chuyên môn của bác sĩ.'
        ),
    }


def get_lstm_model_info() -> Dict[str, Any]:
    """
    Retrieve model evaluation metrics and real training history.
    Tries FastAPI GET /model-info, falls back to artifacts/evaluation_metrics.json.
    NO synthetic curve generation — reads 100% genuine training logs.
    """
    service_url = getattr(settings, 'FASTAPI_IDH_SERVICE_URL', 'http://127.0.0.1:8001')
    endpoint = f"{service_url.rstrip('/')}/model-info"

    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(endpoint)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass

    # Fallback to local artifacts directory
    artifacts_dir = os.path.join(settings.BASE_DIR, 'idh_ai_service', 'artifacts')
    metrics_path = os.path.join(artifacts_dir, 'evaluation_metrics.json')

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return {
        'model_name': 'IDHLSTMNet',
        'status': 'NO_ARTIFACTS_FOUND',
        'metrics': {},
    }
