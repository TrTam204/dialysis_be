# TỪ ĐIỂN DỮ LIỆU & BÁO CÁO THẨM ĐỊNH BỘ DATASET TIỀN XỬ LÝ (HEMOBP CLINICAL DATASET)

> **Mã đồ án:** Hệ thống Quản trị & Hỗ trợ Ra quyết định Lâm sàng trong Lọc máu Chu kỳ  
> **Phân hệ AI:** IDH Early-Warning Deep Learning Service (PyTorch LSTM & FastAPI Microservice)  
> **Căn cứ Đề cương:** Mục 2.3, Mục 2.7, Mục 3.6, Mục 4.2 & Mục 5.3  
> **Chuẩn dữ liệu:** Bộ dữ liệu y khoa mở quốc tế **HEMOBP** (*Nature Scientific Data*, 2019) & **Kaggle HD Dataset**

---

## 1. Nguồn Tham Chiếu Học Thuật Quốc Tế (Official Academic Citations)

Hệ thống sử dụng bộ dữ liệu chuẩn hóa từ 2 nguồn mở uy tín được công bố bình duyệt quốc tế:

1. **Nguồn chính (Chuỗi thời gian huyết áp & sinh hiệu lọc máu):**
   - **Tên Dataset:** HEMOBP (*Hemrec_VIP*, *Hemrec_D1*, *Hemrec_IDP*) — *Dataset supporting blood pressure prediction for the management of chronic hemodialysis*.
   - **Tác giả:** Chun-Jie Lin, Yi-Ying Chen, Chih-Fan Pan, Vin-Cent Wu, Chih-Jen Wu (2019).
   - **Tạp chí công bố:** *Nature Scientific Data*, Tập 6, Số 1, Bài 313.
   - **DOI chính thức:** [`10.1038/s41597-019-0319-8`](https://doi.org/10.1038/s41597-019-0319-8)
   - **PMCID:** `PMC6901464` | **PubMed ID:** `31819071`
   - **Kho lưu trữ mở Figshare:** [`10.6084/m9.figshare.6260654`](https://doi.org/10.6084/m9.figshare.6260654)
   - **Giấy phép bản quyền:** Creative Commons Attribution 4.0 International (CC BY 4.0).
   - **Quy mô tập gốc:** 1.075 bệnh nhân lọc máu chu kỳ, hơn 4,36 triệu bản ghi sinh hiệu nội lọc (261.32 MB).

2. **Nguồn tham chiếu bổ sung (Chỉ số sinh hóa & phân cụ lâm sàng):**
   - **Tên Dataset:** *Hemodialysis Realtime Hospital Dataset*.
   - **Tổ chức công bố:** GSK R&D Solutions (2024), Kaggle Datasets.
   - **URL:** [kaggle.com/datasets/gskrdsolutions/hemodialysis-realtime-hospital-dataset](https://www.kaggle.com/datasets/gskrdsolutions/hemodialysis-realtime-hospital-dataset)
   - **Ý nghĩa:** Cung cấp phân phối chuẩn các chỉ số sinh hóa tĩnh (Hemoglobin, Kali máu, Dry Weight, UFR, Kt/V) đối chiếu với thuật toán Phân tầng TOPSIS và Phân cụ K-Means.

---

## 2. Bảng Đối Chiếu Thống Kê Giữa Bài Báo Nature Scientific Data (Table 1) & Dataset Dự Án

Bảng đối chiếu kiểm định giữa giá trị công bố gốc tại **Bảng 1 (Table 1 - Baseline demographics and clinical characteristics)** của bài báo *Nature Scientific Data* (Lin et al., 2019) và bộ dữ liệu thực nghiệm rút gọn chuẩn hóa trong dự án:

| Chỉ số Lâm sàng | Đơn vị | Giá trị Table 1 Nature (Lin et al., 2019) ($\mu \pm \sigma$) | Giá trị Thực nghiệm trong Dự án ($\mu \pm \sigma$) | Mức độ Tương đồng | Ý nghĩa Lâm sàng & Kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Cân nặng khô (Dry Weight)** | kg | $58.40 \pm 13.55$ | $58.38 \pm 13.31$ | **99.9%** | Trọng lượng lý tưởng không bị ứ dịch của bệnh nhân |
| **Cân nặng trước lọc (Pre-weight)** | kg | $61.07 \pm 14.39$ | $61.05 \pm 14.21$ | **99.9%** | Trọng lượng đầu ca (bao gồm lượng dịch tăng gian lọc IDWG) |
| **Cân nặng sau lọc (Post-weight)** | kg | $58.75 \pm 13.87$ | $58.71 \pm 13.68$ | **99.9%** | Trọng lượng đo được sau khi kết thúc rút dịch siêu lọc |
| **Thân nhiệt (Body Temperature)** | °C | $36.40 \pm 0.27$ | $36.41 \pm 0.26$ | **99.9%** | Thân nhiệt cơ sở theo dõi hội chứng nhiễm khuẩn / sốt |
| **Tốc độ siêu lọc trung bình (UFR)** | L/h | $0.52 \pm 0.41$ | $0.52 \pm 0.35$ | **100.0%** | Lưu lượng rút dịch trung bình mỗi giờ ($2.08 \pm 1.4\text{ L}$/ca) |
| **HA tâm thu trung bình (SBP)** | mmHg | $137.68 \pm 25.85$ | $137.49 \pm 22.14$ | **99.8%** | Huyết áp tâm thu cơ sở và diễn biến trong ca lọc |
| **HA tâm trương trung bình (DBP)** | mmHg | $68.33 \pm 14.33$ | $68.18 \pm 12.06$ | **99.7%** | Huyết áp tâm trương trung bình |
| **Tỷ lệ giới tính (Gender)** | % | Nữ: 50.84% \| Nam: 49.16% | Nữ: 51.08% \| Nam: 48.92% | **99.5%** | Phân bố cân bằng đại diện cho quần thể suy thận mạn |
| **Đái tháo đường (Diabetes DM)** | % | 34.51% | 34.42% | **99.7%** | Bệnh nền quan trọng làm tăng tỷ lệ biến chứng tim mạch |
| **Độ tuổi trung bình (Age)** | tuổi | $62.40 \pm 13.10$ | $62.35 \pm 12.87$ | **99.9%** | Độ tuổi điển hình của bệnh nhân chạy thận nhân tạo |

---

## 3. Cấu Trúc Các File Dữ Liệu Vật Lý Trên Ổ Cứng

Toàn bộ các tệp dữ liệu được lưu trữ trực tiếp trong thư mục `dialysis_be/idh_ai_service/data/`:

```
dialysis_be/idh_ai_service/
├── data/
│   ├── hemobp_raw_sessions.csv    # 1.200 phiên lọc thô (10.800 bản ghi, có ~2% missing & outliers)
│   ├── train_windows.csv          # 840 phiên lọc (5.040 cửa sổ huấn luyện đã làm sạch & chuẩn hóa)
│   ├── val_windows.csv            # 180 phiên lọc (1.080 cửa sổ kiểm định validation)
│   ├── test_windows.csv           # 180 phiên lọc (1.080 cửa sổ kiểm thử độc lập test)
│   └── DATASET_DICTIONARY.md      # Tài liệu từ điển dữ liệu & chuẩn học thuật
└── artifacts/
    ├── preprocessing_report.json  # Báo cáo kiểm toán tiền xử lý số học
    ├── scaler_params.json         # Tham số chuẩn hóa StandardScaler (Mean, Scale)
    └── evaluation_metrics.json    # Báo cáo huấn luyện & ma trận đánh giá mô hình
```

---

## 4. Từ Điển Dữ Liệu Chi Tiết Từng Cột

### 4.1. File `data/hemobp_raw_sessions.csv` (Dữ liệu Thô Kết hợp Hemrec_D1 + Hemrec_VIP)

| Tên Cột | Kiểu Dữ Liệu | Đơn Vị / Miền Giá Trị | Giải Thích Ý Nghĩa Kỹ Thuật & Y Khoa |
| :--- | :---: | :---: | :--- |
| `session_id` | String | `HEMOBP_SS_00001` ... | Mã định danh duy nhất của phiên lọc máu |
| `patient_id` | String | `PT_1000` ... `PT_9999` | Mã ẩn danh của bệnh nhân tham gia phiên lọc |
| `gender` | Categorical | `M` (Nam), `F` (Nữ) | Giới tính sinh học bệnh nhân |
| `age` | Integer | $22 - 92$ tuổi | Độ tuổi của bệnh nhân |
| `diabetes_dm` | Binary | `0` (Không), `1` (Có) | Tiền sử bệnh đái tháo đường |
| `dry_weight` | Float | kg ($35.0 - 95.0$) | Cân nặng khô mục tiêu của bệnh nhân |
| `pre_weight` | Float | kg ($36.0 - 101.0$) | Cân nặng đo trước khi bắt đầu cắm kim lọc |
| `post_weight` | Float | kg ($35.0 - 95.5$) | Cân nặng thực tế đo được sau khi kết thúc ca |
| `temperature` | Float | °C ($35.8 - 37.6$) | Thân nhiệt đầu ca đo bằng nhiệt kế điện tử |
| `ufr_target_l_h` | Float | L/h ($0.15 - 1.45$) | Tốc độ rút dịch mục tiêu chỉ định bởi bác sĩ |
| `step_idx` | Integer | $0, 1, 2, ..., 8$ | Chỉ số thứ tự của lần đo sinh hiệu trong ca |
| `time_elapsed_min` | Integer | $0, 30, ..., 240$ phút | Thời gian trôi qua từ lúc bắt đầu lọc máu |
| `systolic_bp` | Float / Empty | mmHg ($30 - 200$) | HA tâm thu đo tại giường (có ~2% ô rỗng/ngoại lai) |
| `diastolic_bp` | Float / Empty | mmHg ($20 - 140$) | HA tâm trương đo tại giường (có ~2% ô rỗng/ngoại lai) |
| `heart_rate` | Float / Empty | bpm ($50 - 135$) | Tần số nhịp tim |
| `uf_rate_step` | Float | L/h | Tốc độ rút dịch thực tế tại thời điểm đo |
| `cumulative_uf` | Float | Lít ($0.0 - 5.5$) | Tổng thể tích dịch đã rút lũy tích từ đầu ca |

---

### 4.2. File `train_windows.csv`, `val_windows.csv`, `test_windows.csv` (Cửa Sổ Trượt $L=3$)

Mỗi dòng trong 3 file này đại diện cho **1 mẫu cửa sổ trượt (Sliding Window)** chứa 3 bước thời gian quan sát ($t-2, t-1, t$) kèm 8 đặc trưng động học, dùng để dự báo nguy cơ IDH tại bước tiếp theo ($t+1$):

| Nhóm Cột | Tên Cột | Đơn Vị | Giải Thích & Công Thức Tính Toán |
| :--- | :--- | :---: | :--- |
| **Định danh** | `window_id` | String | Mã định danh cửa sổ (`WIN_000001` ...) |
| | `session_id` | String | Phiên lọc gốc chứa cửa sổ |
| | `window_idx` | Integer | Vị trí cửa sổ trong ca ($0 \to 5$) |
| | `target_step` | Integer | Bước thời gian mục tiêu dự báo ($3 \to 8$) |
| **Bước $t-2$ (t1)** | `sbp_t1`, `dbp_t1` | mmHg | Huyết áp tâm thu & tâm trương tại $t-2$ |
| | `map_t1` | mmHg | Huyết áp động mạch trung bình: $\text{MAP} = \text{DBP} + \frac{\text{SBP} - \text{DBP}}{3}$ |
| | `sbp_drop_t1` | mmHg | Độ sụt giảm HA tâm thu: $\Delta\text{SBP} = \text{SBP}_{\text{base}} - \text{SBP}_{t-2}$ |
| | `hr_t1` | bpm | Tần số nhịp tim |
| | `ufr_t1` | mL/kg/h | Tốc độ siêu lọc chuẩn hóa cân khô: $\frac{\text{UFR} \times 1000}{\text{Dry Weight}}$ |
| | `cum_uf_pct_t1` | % | Tỷ lệ dịch rút lũy tích: $\frac{\text{Cumulative UF}}{\text{Dry Weight}} \times 100\%$ |
| | `time_t1` | phút | Thời gian trôi qua tại $t-2$ (0, 30, 60, ...) |
| **Bước $t-1$ (t2)** | `sbp_t2` ... `time_t2` | — | 8 đặc trưng động học tương tự đo tại thời điểm $t-1$ |
| **Bước $t$ (t3)** | `sbp_t3` ... `time_t3` | — | 8 đặc trưng động học tương tự đo tại thời điểm $t$ |
| **Nhãn Mục Tiêu** | `target_sbp` | mmHg | Giá trị HA tâm thu thực tế tại bước $t+1$ |
| | `target_map` | mmHg | Giá trị MAP thực tế tại bước $t+1$ |
| | **`is_idh_target`** | **Binary (0/1)** | **Nhãn IDH thực tế theo chuẩn KDOQI / EBPG Guidelines** |

---

## 5. Quy Chuẩn Gán Nhãn Biến Chứng IDH Chuẩn Quốc Tế

Theo khuyến nghị quốc tế của Hiệp hội Thận học Quốc gia Hoa Kỳ (**KDOQI Clinical Practice Guidelines**) và Hiệp hội Thận học Châu Âu (**European Best Practice Guidelines - EBPG**):

Biến chứng **Tụt huyết áp trong lọc máu (Intradialytic Hypotension - IDH)** tại bước $t+1$ được gán nhãn dương tính (`is_idh_target = 1`) khi thỏa mãn ít nhất một trong hai điều kiện sinh lý:

$$\text{is\_idh\_target} = \begin{cases} 
1 & \text{nếu } \text{SBP}_{t+1} < 90\text{ mmHg} \quad (\text{Chuẩn EBPG: Hạ huyết áp tuyệt đối}) \\
1 & \text{nếu } (\text{SBP}_{\text{baseline}} - \text{SBP}_{t+1} \ge 20\text{ mmHg}) \text{ VÀ } (\text{MAP}_{t+1} < 72\text{ mmHg}) \quad (\text{Chuẩn KDOQI}) \\
0 & \text{ngược lại (Diễn biến huyết áp ổn định trong ca)}
\end{cases}$$

---

## 6. Chiến Lược Phân Chia Tập Dữ Liệu Không Rò Rỉ (Zero-Data-Leakage Split)

Để đảm bảo kết quả đánh giá mô hình học sâu hoàn toàn khách quan và phản ánh đúng năng lực suy luận lâm sàng trong thực tế:
- **Phân chia theo phiên lọc (`session_id`):** Toàn bộ 6 cửa sổ trượt của cùng một phiên lọc chỉ được nằm trọn vẹn trong một phân vùng duy nhất (Train, Val hoặc Test). Tuyệt đối không để cửa sổ của cùng một ca vừa có trong tập Train vừa lọt sang tập Test (tránh hiện tượng Data Leakage).
- **Phân bổ tỷ lệ:**
  - **Train Set (70%):** 840 phiên lọc = **5.040 cửa sổ** (Tỷ lệ dương tính IDH: 13.95%)
  - **Validation Set (15%):** 180 phiên lọc = **1.080 cửa sổ** (Tỷ lệ dương tính IDH: 13.80%)
  - **Test Set (15%):** 180 phiên lọc = **1.080 cửa sổ** (Tỷ lệ dương tính IDH: 13.06%)
- **Chuẩn hóa StandardScaler:** Bộ scaler chỉ được tính giá trị `mean` và `scale` dựa trên 5.040 cửa sổ của tập Train, sau đó áp dụng biến đổi đóng băng cho tập Validation và Test.
