"""
FastAPI Microservice for Hemodialysis Intradialytic Hypotension (IDH) AI Inference.
Runs independently on port 8001.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from idh_ai_service import __version__
from idh_ai_service.lstm_model import IDHInferencePipeline
from idh_ai_service.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictIDHRequest,
    PredictIDHResponse,
)

# Global pipeline & metrics instances
pipeline: Optional[IDHInferencePipeline] = None
evaluation_report: Dict[str, Any] = {}


def get_pipeline() -> IDHInferencePipeline:
    """Lazy-load or return existing inference pipeline instance."""
    global pipeline
    if pipeline is None:
        artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
        pipeline = IDHInferencePipeline(artifacts_dir=artifacts_dir)
    return pipeline


def get_evaluation_report() -> Dict[str, Any]:
    """Lazy-load or return existing model evaluation report."""
    global evaluation_report
    if not evaluation_report:
        artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
        metrics_path = os.path.join(artifacts_dir, 'evaluation_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                evaluation_report = json.load(f)
    return evaluation_report


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    get_evaluation_report()
    yield


app = FastAPI(
    title="LocMau IDH Deep Learning Prediction Service",
    description="FastAPI service serving PyTorch LSTM model for early warning of Intradialytic Hypotension (IDH).",
    version=__version__,
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend & backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Monitoring"],
)
async def health_check():
    """Verify microservice status and model availability."""
    pipe = get_pipeline()
    is_ready = bool(pipe and pipe.is_loaded)
    return HealthResponse(
        status="HEALTHY" if is_ready else "INITIALIZING",
        model_loaded=is_ready,
        version=__version__,
        device="cpu",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="LSTM Model Architecture & Evaluation Metrics",
    tags=["Model Metadata"],
)
async def get_model_info():
    """Return model architecture, input features, and evaluation metrics (Accuracy, Precision, Recall, F1, Loss)."""
    report = get_evaluation_report()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model evaluation report not found. Please train model first.",
        )
    return report


@app.post(
    "/predict-idh",
    response_model=PredictIDHResponse,
    summary="Predict IDH Risk from Vital Signs Time-Series",
    tags=["Inference"],
)
async def predict_idh(request: PredictIDHRequest):
    """
    Predict intradialytic hypotension risk score from sequential vital signs measurements.
    Returns:
    - risk_probability (0.0 - 1.0)
    - risk_level (LOW, MEDIUM, HIGH)
    - is_idh_predicted (boolean alert)
    - physiological contributing factors
    - recommended preventative clinical interventions
    """
    pipe = get_pipeline()
    if not pipe or not pipe.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LSTM Model is not loaded or weights are missing.",
        )

    # Convert Pydantic models to dicts
    vitals_dicts = [step.model_dump() for step in request.vitals_sequence]

    try:
        prediction_result = pipe.predict_sequence(
            vitals_list=vitals_dicts,
            threshold=request.threshold or 0.50,
            dry_weight=request.dry_weight,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(exc)}",
        )

    return PredictIDHResponse(
        prediction_id=str(uuid.uuid4()),
        patient_id=request.patient_id,
        session_id=request.session_id,
        risk_probability=prediction_result['risk_probability'],
        risk_level=prediction_result['risk_level'],
        is_idh_predicted=prediction_result['is_idh_predicted'],
        threshold_used=prediction_result['threshold_used'],
        contributing_factors=prediction_result['contributing_factors'],
        clinical_recommendations=prediction_result['clinical_recommendations'],
        sequence_length_received=prediction_result['sequence_length_received'],
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("idh_ai_service.main:app", host="0.0.0.0", port=8001, reload=True)
