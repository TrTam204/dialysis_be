"""
Unit and Integration Tests for LSTM IDH Model and FastAPI Service.
Tests PyTorch model architecture, inference pipeline, and FastAPI endpoints.
"""
import unittest
import torch
from fastapi.testclient import TestClient

from idh_ai_service.dataset_generator import generate_hemodialysis_sessions
from idh_ai_service.lstm_model import IDHInferencePipeline, IDHLSTMNet
from idh_ai_service.main import app


class TestIDHAIService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.pipeline = IDHInferencePipeline()

    def test_01_dataset_generator_shape_and_labels(self):
        """Verify data generator outputs correct 3D sequence and binary label tensors."""
        sequences, labels = generate_hemodialysis_sessions(n_sessions=20, seed=42)
        # Sliced windows: 20 sessions * (9 - 5) = 80 samples
        self.assertEqual(len(sequences.shape), 3)
        self.assertEqual(sequences.shape[1], 5)  # seq_len = 5
        self.assertEqual(sequences.shape[2], 7)  # n_features = 7
        self.assertEqual(len(labels), len(sequences))
        self.assertTrue(set(labels).issubset({0.0, 1.0}))

    def test_02_lstm_model_architecture_forward(self):
        """Verify forward pass of PyTorch LSTM neural network."""
        model = IDHLSTMNet(input_dim=7, hidden_dim=32, num_layers=2)
        model.eval()
        dummy_input = torch.randn(4, 5, 7)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (4, 1))

    def test_03_inference_pipeline_loaded(self):
        """Verify that inference pipeline successfully loads artifacts."""
        self.assertTrue(self.pipeline.is_loaded)
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.scaler_mean)

    def test_04_fastapi_health_endpoint(self):
        """Verify GET /health returns status HEALTHY and model_loaded True."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["device"], "cpu")

    def test_05_fastapi_model_info_endpoint(self):
        """Verify GET /model-info returns required evaluation metrics."""
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metrics", data)
        self.assertIn("test", data["metrics"])

        test_metrics = data["metrics"]["test"]
        self.assertIn("accuracy", test_metrics)
        self.assertIn("precision", test_metrics)
        self.assertIn("recall", test_metrics)
        self.assertIn("f1_score", test_metrics)
        self.assertIn("loss", test_metrics)

        # Check performance thresholds per diploma syllabus
        self.assertGreater(test_metrics["accuracy"], 0.90)
        self.assertGreater(test_metrics["f1_score"], 0.85)

    def test_06_fastapi_predict_idh_stable_trajectory(self):
        """Verify POST /predict-idh on stable vital signs trajectory yields LOW risk."""
        payload = {
            "patient_id": "PT-TEST-STABLE",
            "session_id": "SS-TEST-001",
            "threshold": 0.50,
            "vitals_sequence": [
                {"systolic_bp": 135.0, "diastolic_bp": 80.0, "heart_rate": 72.0, "uf_rate": 0.6, "cumulative_uf": 0.3, "time_elapsed_min": 0},
                {"systolic_bp": 132.0, "diastolic_bp": 78.0, "heart_rate": 74.0, "uf_rate": 0.6, "cumulative_uf": 0.6, "time_elapsed_min": 30},
                {"systolic_bp": 130.0, "diastolic_bp": 77.0, "heart_rate": 73.0, "uf_rate": 0.6, "cumulative_uf": 0.9, "time_elapsed_min": 60},
                {"systolic_bp": 128.0, "diastolic_bp": 76.0, "heart_rate": 75.0, "uf_rate": 0.6, "cumulative_uf": 1.2, "time_elapsed_min": 90},
                {"systolic_bp": 126.0, "diastolic_bp": 75.0, "heart_rate": 74.0, "uf_rate": 0.6, "cumulative_uf": 1.5, "time_elapsed_min": 120},
            ],
        }
        response = self.client.post("/predict-idh", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("risk_probability", data)
        self.assertEqual(data["risk_level"], "LOW")
        self.assertFalse(data["is_idh_predicted"])
        self.assertGreater(len(data["clinical_recommendations"]), 0)

    def test_07_fastapi_predict_idh_hypotension_alert(self):
        """Verify POST /predict-idh on dropping SBP trajectory triggers HIGH risk IDH alert."""
        payload = {
            "patient_id": "PT-TEST-ALERT",
            "session_id": "SS-TEST-002",
            "threshold": 0.50,
            "vitals_sequence": [
                {"systolic_bp": 130.0, "diastolic_bp": 80.0, "heart_rate": 70.0, "uf_rate": 1.2, "cumulative_uf": 0.6, "time_elapsed_min": 0},
                {"systolic_bp": 120.0, "diastolic_bp": 72.0, "heart_rate": 74.0, "uf_rate": 1.3, "cumulative_uf": 1.2, "time_elapsed_min": 30},
                {"systolic_bp": 105.0, "diastolic_bp": 65.0, "heart_rate": 84.0, "uf_rate": 1.4, "cumulative_uf": 1.9, "time_elapsed_min": 60},
                {"systolic_bp": 92.0, "diastolic_bp": 58.0, "heart_rate": 92.0, "uf_rate": 1.4, "cumulative_uf": 2.6, "time_elapsed_min": 90},
                {"systolic_bp": 82.0, "diastolic_bp": 50.0, "heart_rate": 96.0, "uf_rate": 1.4, "cumulative_uf": 3.3, "time_elapsed_min": 120},
            ],
        }
        response = self.client.post("/predict-idh", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("risk_probability", data)
        self.assertGreaterEqual(data["risk_probability"], 0.65)
        self.assertEqual(data["risk_level"], "HIGH")
        self.assertTrue(data["is_idh_predicted"])
        self.assertTrue(any("CẢNH BÁO CAO" in rec for rec in data["clinical_recommendations"]))

    def test_08_fastapi_predict_idh_validation_error_short_sequence(self):
        """Verify POST /predict-idh with < 3 steps returns 422 Unprocessable Entity."""
        payload = {
            "vitals_sequence": [
                {"systolic_bp": 120.0, "diastolic_bp": 70.0, "heart_rate": 70.0, "uf_rate": 0.5, "cumulative_uf": 0.5, "time_elapsed_min": 0},
            ],
        }
        response = self.client.post("/predict-idh", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
