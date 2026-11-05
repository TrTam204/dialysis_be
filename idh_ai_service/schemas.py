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
    fastapi_inference_ms: Optional[float] = Field(None, description="Internal model inference execution time at FastAPI microservice (ms)")


class MetricSet(BaseModel):
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    sample_count: int


class HyperparameterExperiment(BaseModel):
    """Result of a single hyperparameter configuration in grid search."""
    exp_id: str = Field(..., description="Experiment Identifier")
    num_layers: int = Field(..., description="Number of stacked LSTM layers")
    hidden_dim: int = Field(..., description="Hidden dimension size per layer")
    dropout: float = Field(..., description="Dropout rate")
    learning_rate: float = Field(..., description="Initial learning rate for Adam optimizer")
    batch_size: int = Field(..., description="Mini-batch size")
    val_loss: float = Field(..., description="Validation Loss")
    val_accuracy: float = Field(..., description="Validation Accuracy")
    val_f1: float = Field(..., description="Validation F1-score")
    val_roc_auc: float = Field(..., description="Validation ROC-AUC")
    selected: bool = Field(False, description="Whether this experiment was selected as optimal")
    note: str = Field("", description="Clinical and technical evaluation note")


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
    hyperparameter_tuning: Optional[List[HyperparameterExperiment]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
    device: str
    timestamp: str


# Syllabus Aliases for Microservice Schemas
IDHPredictionRequest = PredictIDHRequest
IDHPredictionResponse = PredictIDHResponse
VitalStepInput = VitalSignTimeStep
