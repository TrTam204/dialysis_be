"""
LSTM Deep Neural Network Architecture and Inference Pipeline for IDH Prediction.
Uses a 2-layer Recurrent LSTM with Forget/Input/Cell/Output gates for sequential vital signs modeling.
"""
from typing import Any, Dict, List, Optional
import json
import os
import torch
import torch.nn as nn
import numpy as np


class IDHLSTMNet(nn.Module):
    """
    Standard PyTorch LSTM Classifier for Time-Series IDH Early Warning.
    Contains 2 stacked LSTM recurrent layers followed by fully-connected
    classification layers with dropout and layer normalization.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Core LSTM recurrent unit
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head on final hidden state
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: Output tensor of shape (batch_size, 1)
        """
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Extract last time-step hidden representation
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        normed = self.layer_norm(last_hidden)

        out = self.fc1(normed)
        out = self.relu(out)
        out = self.drop(out)
        logits = self.fc2(out)  # (batch_size, 1)
        return logits


class IDHInferencePipeline:
    """
    Production inference pipeline:
    - Preprocesses raw vital signs time-series sequences.
    - Scales features using training distribution parameters.
    - Runs PyTorch forward inference on CPU.
    - Synthesizes physiological contributing factors and clinical action recommendations.
    """

    DEFAULT_FEATURE_NAMES = [
        'systolic_bp',
        'diastolic_bp',
        'map',
        'heart_rate',
        'uf_rate',
        'cumulative_uf',
        'time_elapsed_min',
    ]

    def __init__(self, artifacts_dir: Optional[str] = None):
        if artifacts_dir is None:
            artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
        self.artifacts_dir = artifacts_dir

        self.model: Optional[IDHLSTMNet] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.feature_names: List[str] = self.DEFAULT_FEATURE_NAMES
        self.is_loaded: bool = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_path = os.path.join(self.artifacts_dir, 'lstm_idh_model.pt')
        scaler_path = os.path.join(self.artifacts_dir, 'scaler_params.json')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            self.is_loaded = False
            return

        # Load scaler
        with open(scaler_path, 'r', encoding='utf-8') as f:
            scaler_data = json.load(f)
            self.scaler_mean = np.array(scaler_data['mean'], dtype=np.float32)
            self.scaler_scale = np.array(scaler_data['scale'], dtype=np.float32)
            self.feature_names = scaler_data.get('feature_names', self.DEFAULT_FEATURE_NAMES)

        # Load model weights
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        input_dim = checkpoint.get('input_dim', len(self.feature_names))
        hidden_dim = checkpoint.get('hidden_dim', 64)
        num_layers = checkpoint.get('num_layers', 2)

        self.model = IDHLSTMNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.is_loaded = True

    def predict_sequence(
        self,
        vitals_list: List[Dict[str, Any]],
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Run inference on a sequence of vital signs.
        Args:
            vitals_list: List of dicts representing consecutive time steps.
            threshold: Probability threshold for binary IDH prediction.
        Returns:
            Dict containing risk_probability, risk_level, contributing factors, recommendations.
        """
        if not self.is_loaded or self.model is None or self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("IDH Model or Scaler parameters not loaded. Run train_lstm.py first.")

        n_steps = len(vitals_list)
        if n_steps < 3:
            raise ValueError(f"Sequence length must be at least 3 time steps, got {n_steps}.")

        # Convert dicts to 2D numpy array
        raw_matrix = []
        for step in vitals_list:
            sbp = float(step.get('systolic_bp', 120.0))
            dbp = float(step.get('diastolic_bp', 75.0))
            raw_map = step.get('map')
            map_val = float(raw_map) if raw_map is not None else float(dbp + (sbp - dbp) / 3.0)
            hr = float(step.get('heart_rate', 75.0))
            ufr = float(step.get('uf_rate', 0.8))
            cum_uf = float(step.get('cumulative_uf', 1.0))
            t_min = float(step.get('time_elapsed_min', 0))

            raw_matrix.append([sbp, dbp, map_val, hr, ufr, cum_uf, t_min])

        raw_array = np.array(raw_matrix, dtype=np.float32)

        # Standardize using saved scaler mean and scale
        scaled_array = (raw_array - self.scaler_mean) / self.scaler_scale

        # Prepare tensor of shape (1, seq_len, input_dim)
        input_tensor = torch.tensor(scaled_array, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).item()

        # Risk classification
        if prob < 0.30:
            risk_level = "LOW"
        elif prob < 0.65:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        is_idh = bool(prob >= threshold)

        # Physiological factor analysis & recommendations
        contributing_factors = self._analyze_factors(raw_array)
        recommendations = self._generate_recommendations(risk_level, raw_array)

        return {
            'risk_probability': round(prob, 4),
            'risk_level': risk_level,
            'is_idh_predicted': is_idh,
            'threshold_used': threshold,
            'contributing_factors': contributing_factors,
            'clinical_recommendations': recommendations,
            'sequence_length_received': n_steps,
        }

    def _analyze_factors(self, raw_array: np.ndarray) -> List[str]:
        """Examine trajectory for specific clinical alert triggers."""
        factors = []
        sbp_values = raw_array[:, 0]
        map_values = raw_array[:, 2]
        hr_values = raw_array[:, 3]
        ufr_values = raw_array[:, 4]

        # SBP trajectory drop
        sbp_drop = sbp_values[0] - sbp_values[-1]
        if sbp_drop >= 15.0:
            factors.append(f"Huyết áp tâm thu có xu hướng giảm nhanh ({sbp_drop:.1f} mmHg trong các bước vừa qua).")
        if sbp_values[-1] < 100.0:
            factors.append(f"Huyết áp tâm thu hiện tại thấp gần ngưỡng tụt ({sbp_values[-1]:.0f} mmHg).")

        # MAP threshold
        if map_values[-1] < 72.0:
            factors.append(f"Huyết áp động mạch trung bình (MAP) thấp ({map_values[-1]:.1f} mmHg).")

        # Ultrafiltration intensity
        if ufr_values[-1] >= 1.2:
            factors.append(f"Tốc độ rút nước (UFR) cao ({ufr_values[-1]:.2f} L/h), làm tăng nguy cơ giảm thể tích nội mạch.")

        # Heart rate dynamics
        hr_change = hr_values[-1] - hr_values[0]
        if hr_change > 15:
            factors.append(f"Nhịp tim tăng nhanh bù trừ ({hr_values[-1]:.0f} bpm).")
        elif hr_change < -10 and sbp_drop > 10:
            factors.append("Hiện tượng nhịp tim chậm nghịch lý kết hợp tụt huyết áp (nghi ngờ phản xạ Bezold-Jarisch).")

        if not factors:
            factors.append("Các chỉ số sinh hiệu và tốc độ siêu lọc trong giới hạn ổn định thường quy.")

        return factors

    def _generate_recommendations(self, risk_level: str, raw_array: np.ndarray) -> List[str]:
        """Generate targeted clinical interventions based on risk tier."""
        if risk_level == "HIGH":
            return [
                "CẢNH BÁO CAO: Chuẩn bị xử trí tụt huyết áp theo phác đồ cấp cứu trung tâm.",
                "Giảm tốc độ rút nước siêu lọc (UF rate) hoặc tạm dừng siêu lọc ngay lập tức.",
                "Đặt bệnh nhân ở tư thế nằm đầu thấp (Trendelenburg) nếu không có chống chỉ định khó thở.",
                "Chuẩn bị bolus 100 - 200 mL NaCl 0.9% qua đường tuần hoàn ngoài cơ thể.",
                "Đo lại huyết áp mỗi 5 - 10 phút và báo ngay cho Bác sĩ phụ trách ca lọc.",
            ]
        elif risk_level == "MEDIUM":
            return [
                "NGUY CƠ TRUNG BÌNH: Tăng cường tần suất theo dõi huyết áp (mỗi 15 phút).",
                "Xem xét điều chỉnh giảm tốc độ siêu lọc (UF rate) về mức an toàn hơn.",
                "Cân nhắc hạ nhiệt độ dịch lọc (Dialysate temperature) xuống 35.5 - 36.0 °C để kích hoạt co mạch ngoại vi.",
                "Nhắc nhở người bệnh thông báo ngay cho điều dưỡng nếu xuất hiện triệu chứng hoa mắt, chóng mặt, buồn nôn hoặc ngáp.",
            ]
        else:
            return [
                "NGUY CƠ THẤP: Tiếp tục duy trì thông số lọc máu và phác đồ hiện tại.",
                "Đo huyết áp và ghi nhận sinh hiệu định kỳ theo quy trình thường quy (mỗi 30 - 60 phút).",
            ]
