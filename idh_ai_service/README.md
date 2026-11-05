# Microservice Dự báo Sớm Tụt Huyết Áp (IDH) Trong Ca Lọc Máu bằng Học Sâu LSTM (FastAPI)

Dịch vụ AI Microservice độc lập phục vụ suy luận mô hình Mạng nơ-ron hồi quy **2-Layer Stacked LSTM (PyTorch 2.x)** nhằm cảnh báo sớm nguy cơ tụt huyết áp trong ca lọc máu (**Intradialytic Hypotension - IDH**) trước 30–60 phút.

---

## 1. Thông Tin Kiến Trúc & Tiêu Chuẩn Y Khoa

- **Mô hình AI:** `IDHLSTMNet` (2-Layer Stacked LSTM + LayerNorm + Dropout + Linear + Sigmoid).
- **Khuôn khổ (Framework):** PyTorch 2.x, FastAPI, Pydantic v2, Scikit-learn (StandardScaler).
- **Cổng dịch vụ mặc định:** `8001` (Microservice độc lập tách biệt khỏi Django Core port `8000`).
- **Độ dài chuỗi quan sát ($seq\_len$):** `3` bước sinh hiệu liên tiếp (tương đương 90 phút lâm sàng).
- **8 Đặc trưng sinh hiệu đầu vào ($input\_dim = 8$):**
  1. `systolic_bp`: Huyết áp tâm thu ($mmHg$)
  2. `diastolic_bp`: Huyết áp tâm trương ($mmHg$)
  3. `map`: Huyết áp động mạch trung bình ($MAP = DBP + \frac{SBP - DBP}{3}$)
  4. `sbp_drop_from_baseline`: Độ tụt huyết áp tâm thu so với lúc bắt đầu ca lọc ($mmHg$)
  5. `heart_rate`: Nhịp tim ($nhịp/phút$)
  6. `ufr_ml_kg_h`: Tốc độ rút nước siêu lọc chuẩn hóa theo cân nặng khô ($mL/kg/giờ$)
  7. `cumulative_uf_pct`: Tỷ lệ dịch rút lũy kế so với cân nặng khô ($\%$)
  8. `time_elapsed_min`: Thời gian đã trôi qua trong phiên lọc ($phút$)
- **Tiêu chuẩn Y khoa Gán nhãn IDH (KDOQI / European Best Practice Guidelines):**
  $$\text{IDH} = 1 \iff (\text{SBP}_{t+1} < 90\text{ mmHg}) \lor (\text{Base SBP} - \text{SBP}_{t+1} \ge 20\text{ mmHg} \land \text{MAP}_{t+1} < 72\text{ mmHg})$$
- **Chiến lược Phân chia Dữ liệu Chống rò rỉ (Zero Data Leakage):**
  - Phân chia theo cấp độ phiên lọc (**Session-level split**): 840 phiên Train (70%), 180 phiên Validation (15%), 180 phiên Test (15%).
  - Chuẩn hóa `StandardScaler` chỉ fit trên tập Train, sau đó transform cho Val/Test và lưu tham số vào `artifacts/scaler_params.json`.

---

## 2. Hướng Dẫn Chạy Độc Lập Microservice

### Bước 1: Kích hoạt môi trường ảo Python
```powershell
cd d:\DATN\LocMau_Project\dialysis_be
.venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện (nếu thiết lập mới)
```powershell
pip install -r idh_ai_service/requirements.txt
```

### Bước 3: Khởi động FastAPI Microservice trên cổng 8001
```powershell
python -m uvicorn idh_ai_service.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## 3. Tài Liệu API & Kiểm Thử Tương Tác Qua Trình Duyệt

Khi dịch vụ đang chạy trên cổng `8001`, bạn có thể truy cập trực tiếp:
- **Swagger UI tương tác:** [http://localhost:8001/docs](http://localhost:8001/docs)
- **ReDoc giao diện chuyên sâu:** [http://localhost:8001/redoc](http://localhost:8001/redoc)

### Danh sách API Endpoints:

| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái microservice và tình trạng nạp model PyTorch |
| `GET` | `/model-info` | Trả về thông số kiến trúc mạng, siêu tham số tối ưu, 5 chỉ số đánh giá (Accuracy, Precision, Recall, F1, Loss) và bảng Grid Search |
| `POST` | `/predict-idh` | Nhận chuỗi sinh hiệu ca lọc, chạy suy luận qua LSTM và trả về xác suất nguy cơ IDH |

---

## 4. Mẫu Yêu Cầu & Phản Hồi Suy Luận (`POST /predict-idh`)

### Payload yêu cầu (`application/json`):
```json
{
  "patient_id": "P_A_05",
  "session_id": "SS-COMP-P_A_05-2",
  "dry_weight": 48.0,
  "threshold": 0.50,
  "vitals_sequence": [
    {
      "systolic_bp": 135.0,
      "diastolic_bp": 80.0,
      "heart_rate": 72.0,
      "uf_rate": 0.6,
      "cumulative_uf": 0.3,
      "time_elapsed_min": 0
    },
    {
      "systolic_bp": 122.0,
      "diastolic_bp": 74.0,
      "heart_rate": 78.0,
      "uf_rate": 0.6,
      "cumulative_uf": 0.6,
      "time_elapsed_min": 30
    },
    {
      "systolic_bp": 108.0,
      "diastolic_bp": 68.0,
      "heart_rate": 84.0,
      "uf_rate": 0.6,
      "cumulative_uf": 0.9,
      "time_elapsed_min": 60
    }
  ]
}
```

### Kết quả trả về (`200 OK`):
```json
{
  "prediction_id": "a5e8486f-2b7f-4402-9860-bd4ef5506041",
  "patient_id": "P_A_05",
  "session_id": "SS-COMP-P_A_05-2",
  "risk_probability": 0.842,
  "risk_level": "HIGH",
  "is_idh_predicted": true,
  "threshold_used": 0.5,
  "contributing_factors": [
    "Huyết áp tâm thu giảm liên tục 27.0 mmHg so với mức nền",
    "Tốc độ rút nước 12.5 mL/kg/h đạt ngưỡng nhạy cảm"
  ],
  "clinical_recommendations": [
    "Cân nhắc giảm tốc độ siêu lọc (UF rate) tạm thời",
    "Chuẩn bị tư thế Trendelenburg nếu bệnh nhân có triệu chứng chóng mặt",
    "Đo lại huyết áp sau 15 phút"
  ],
  "sequence_length_received": 3
}
```

---

## 5. Chạy Kiểm Thử Tự Động (Unit Tests)

```powershell
python -m unittest idh_ai_service/test_fastapi_lstm.py
```
Toàn bộ 8 ca kiểm thử bao phủ tính toàn vẹn của mô hình, chuẩn hóa dữ liệu, xác thực Pydantic và luồng suy luận FastAPI.

