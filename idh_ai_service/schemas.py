"""
Pydantic Data Schemas for FastAPI IDH Prediction Service.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VitalSignTimeStep(BaseModel):
    """Vital sign measurement at a single observation time step."""
    systolic_bp: float = Field(..., description="Systolic Blood Pressure (mmHg)", ge=40, le=280)
    diastolic_bp: float = Field(..., description="Diastolic Blood Pressure (mmHg)", ge=20, le=180)
    map: Optional[float] = Field(None, description="Mean Arterial Pressure (calculated if null)")
    sbp_drop_from_baseline: Optional[float] = Field(None, description="Drop in SBP from baseline (mmHg)")
    heart_rate: float = Field(..., description="Heart Rate (beats per minute)", ge=30, le=220)
    uf_rate: Optional[float] = Field(None, description="Current Ultrafiltration Rate (Liters/hour)", ge=0.0, le=5.0)
    ufr_ml_kg_h: Optional[float] = Field(None, description="Ultrafiltration Rate normalized to dry weight (mL/kg/h)")
    cumulative_uf: Optional[float] = Field(None, description="Cumulative Ultrafiltration Removed (Liters)", ge=0.0, le=15.0)
    cumulative_uf_pct: Optional[float] = Field(None, description="Cumulative UF as % of dry weight")
    time_elapsed_min: int = Field(..., description="Time elapsed in session (minutes)", ge=0, le=480)


class PredictIDHRequest(BaseModel):
    """Inference request payload containing time-series sequence of vital signs."""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    session_id: Optional[str] = Field(None, description="Hemodialysis session identifier")
    dry_weight: Optional[float] = Field(None, description="Patient dry weight (kg)", ge=30, le=150)
    threshold: Optional[float] = Field(0.50, description="Decision threshold for binary IDH alert", ge=0.1, le=0.9)
    vitals_sequence: List[VitalSignTimeStep] = Field(
        ...,
        description="Chronological sequence of consecutive vital signs (minimum 3 steps)",
        min_length=3,
    )


class PredictIDHResponse(BaseModel):
    """Inference response with risk score, alert status, and clinical explanations."""
    prediction_id: str
    patient_id: Optional[str]
    session_id: Optional[str]
    risk_probability: float = Field(..., description="Predicted probability of IDH in upcoming 30-60 min (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    is_idh_predicted: bool = Field(..., description="True if risk_probability >= threshold")
    threshold_used: float
    contributing_factors: List[str]
    clinical_recommendations: List[str]
    sequence_length_received: int


class MetricSet(BaseModel):
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    sample_count: int


class ModelInfoResponse(BaseModel):
    """Model architecture and evaluation metrics report."""
    model_name: str
    framework: str
    architecture: str
    input_features: List[str]
    sequence_length: int
    split_strategy: Optional[str] = None
    dataset_splits: Optional[Dict[str, Any]] = None
    metrics: Dict[str, MetricSet]
    training_history: Optional[List[Dict[str, Any]]] = None
    training_hyperparameters: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
    device: str
    timestamp: str
