# Design Document: Xây dựng Example Dự đoán Phiên Chiều VN30F1M với AutoTS4PS

- **Ngày tạo:** 2026-10-15
- **Tác giả:** Antigravity & Pair Programming Partner
- **Trạng thái:** Chờ phê duyệt (Pending Review)
- **Vị trí tệp triển khai:** `examples/afternoon_session_prediction.py`
- **Bộ dữ liệu mục tiêu:** `datasets/VN30F1M_5m.csv`

---

## 1. Bối cảnh & Mục tiêu (Context & Objectives)

### 1.1. Bối cảnh thị trường & Vấn đề kỹ thuật
Hợp đồng tương lai chỉ số VN30 kỳ hạn 1 tháng (VN30F1M) có phiên giao dịch gián đoạn:
- Phiên sáng: 08:45 – 11:30 (ATO 08:45 – 09:00).
- Giờ nghỉ trưa: 11:30 – 13:00 (90 phút thị trường đóng băng).
- Phiên chiều: 13:00 – 14:45 (ATC 14:30 – 14:45).

Các vấn đề khi áp dụng AutoML/Time-Series thông thường:
1. **Độ trễ thuật toán:** Giải thuật di truyền (Genetic Algorithm) của AutoTS mất từ 15–30 phút để tìm kiếm mô hình tối ưu $\to$ Không thể chạy AutoML search trực tiếp tại từng nến phút trong phiên.
2. **Khung thời gian nghỉ trưa là "thời điểm vàng":** Khoảng nghỉ 90 phút đủ để máy tính chạy 15–25 thế hệ AutoML tìm kiếm mô hình tối ưu nhất cho phiên chiều dựa trên dữ liệu sáng nay đã chốt.
3. **Đứt gãy chuỗi thời gian (Gaps):** Dữ liệu nến phút có khoảng trống nghỉ trưa (1.5h), qua đêm (18h) và 2 ngày cuối tuần, khiến DatetimeIndex bị chèn hàng loạt NaN nếu dùng tần suất cố định.
4. **Bẫy Look-Ahead Bias & Đa cộng tuyến:** Sử dụng bộ lọc 2 chiều (HPFilter, STLFilter, Kalman làm mượt 2 phía) dẫn tới hiện tượng "vẽ lại" (repainting); sử dụng PCA trên 1 cột gây suy biến ma trận.
5. **Số fix cứng tai hại:** Sử dụng ngưỡng điểm tĩnh (ví dụ: 1.5 hay 3.0 điểm) dễ bị dính bẫy nhiễu giả (whipsaw) khi thị trường biến động mạnh hoặc bị tê liệt tín hiệu khi thị trường đi ngang.

### 1.2. Mục tiêu của Example
Xây dựng một script mẫu hoàn chỉnh `examples/afternoon_session_prediction.py` thực hiện:
1. Nén dữ liệu nến 5m từ `datasets/VN30F1M_5m.csv` thành 2 thanh nến mỗi ngày: Nến Sáng (09:00–11:30) và Nến Chiều (13:00–14:45) với chỉ mục tuần tự (Sequential Trading Index) triệt tiêu hoàn toàn gap thời gian.
2. Trích xuất các đặc trưng nội tại (Causal Features) dạng chuỗi dừng từ phiên sáng lúc 11:30 (Morning Return, Normalized ATR, VWAP Deviation, Volume Ratio).
3. Cấu hình AutoTS chuyên biệt cho phái sinh với Whitelist Transformers 1 chiều an toàn và hệ thống Directional Metrics (`dwae`, `oda`, `spl`).
4. Xây dựng Probabilistic Decision Engine chuyển đổi dải dự báo 90% thành kỳ vọng lợi nhuận ròng trừ phí ma sát ($Friction = 0.8$ điểm), tính Z-score xác suất thắng và ngưỡng Breakout động theo ATR sáng.
5. Hỗ trợ 2 chế độ:
   - `--mode advisor`: Chạy giả lập lúc 11:30 cho phiên sáng gần nhất và in bản tin khuyến nghị lệnh lúc 13:00 (LONG, SHORT hoặc WAIT) kèm SL/TP động.
   - `--mode backtest`: Chạy Walk-Forward backtest trên $N$ phiên gần nhất để kiểm định Win Rate, Net PnL và Max Drawdown.

---

## 2. Kiến trúc & Thiết kế Chi tiết (Detailed Design)

### 2.1. Module 1: Data Resampling & Feature Engineering
- **Đầu vào:** Tệp `datasets/VN30F1M_5m.csv` chứa `Date, Open, High, Low, Close, Volume`.
- **Phân loại phiên giao dịch:**
  - Phiên sáng ($Session = Morning$): Các nến có giờ từ `09:00:00` đến `11:30:00`.
  - Phiên chiều ($Session = Afternoon$): Các nến có giờ từ `13:00:00` đến `14:45:00`.
- **Quy tắc gộp nến:**
  - $Open$: Giá mở cửa nến đầu tiên trong phiên.
  - $High$: Giá cao nhất trong tất cả nến của phiên.
  - $Low$: Giá thấp nhất trong tất cả nến của phiên.
  - $Close$: Giá đóng cửa nến cuối cùng trong phiên.
  - $Volume$: Tổng khối lượng khớp lệnh trong phiên.
- **Tính toán Causal Features cho phiên sáng:**
  - `ret_morning`: $\frac{Close_{11:30} - Open_{09:00}}{Open_{09:00}}$
  - `norm_atr`: $\frac{High_{\text{morning}} - Low_{\text{morning}}}{Close_{11:30}}$
  - `vwap_dev`: $\frac{Close_{11:30} - VWAP_{\text{morning}}}{VWAP_{\text{morning}}}$, trong đó $VWAP_{\text{morning}} = \frac{\sum (Close_i \times Volume_i)}{\sum Volume_i}$.
  - `vol_ratio`: Khối lượng phiên sáng chia cho đường trung bình trượt 10 phiên sáng gần nhất: $\frac{Vol_{\text{morning}}}{SMA_{10}(Vol_{\text{morning}})}$.
- **Định dạng dữ liệu đầu vào cho AutoTS:**
  - Tạo chỉ mục thời gian giả lập tuần tự (Sequential Steps, ví dụ freq='12h' hoặc index liên tục không có ngày trống).
  - Bảng dữ liệu bao gồm: cột target `VN30F1M` (giá Close phiên) và các cột ngoại sinh của phiên sáng.

### 2.2. Module 2: AutoTS Model Configuration
- **Horizon:** `forecast_length = 1` (dự báo chính xác 1 bước thời gian tiếp theo — phiên chiều hôm nay).
- **Prediction Interval:** `prediction_interval = 0.90` (dải phân vị 5% và 95%).
- **Whitelist Transformers:**
  ```python
  financial_transformer_list = {
      "DifferencedTransformer": 0.30,   # Bắt buộc: khử xu hướng bậc 1
      "PctChangeTransformer": 0.20,     # Lợi suất phần trăm
      "RobustScaler": 0.15,             # Kháng nhiễu ngoại lai
      "StandardScaler": 0.10,           # Z-score chuẩn hóa
      "EWMAFilter": 0.15,               # Làm mượt 1 chiều (nhân quả)
      "AlignLastValue": 0.10,           # Neo giá đóng cửa gần nhất
  }
  ```
  *Cấm hoàn toàn:* `PCA`, `FastICA`, `HPFilter`, `STLFilter`, `KalmanSmoothing` 2 chiều.
- **Metric Weightings:**
  - `dwae_weighting = 3.5` (Directional Weighted Absolute Error: phạt nặng khi ngược chiều).
  - `oda_weighting = 2.0` (Origin Direction Accuracy).
  - `spl_weighting = 2.0` (Quantile Loss: tối ưu độ phủ của dải Upper/Lower).
  - `contour_weighting = 1.0`.
  - `mae_weighting = 0.8`, `rmse_weighting = 0.4`, `smape_weighting = 0.2`.
- **Model Candidate Pool:**
  - `RollingRegression` (LightGBM / XGBoost / Ridge)
  - `WindowRegression`
  - `VAR`
  - `VECM`
  - `BallTreeMultivariateMotif`
- **Series Weights:** `{'VN30F1M': 100.0}` và các feature giải thích gán `0.0`.

### 2.3. Module 3: Probabilistic Decision & Risk Engine
- **Đầu vào:** Đối tượng `prediction` từ `model.predict()`, thông tin nến sáng ($Close_{11:30}, High_{\text{morning}}, Low_{\text{morning}}, ATR_{\text{morning}}$).
- **Tham số thị trường:**
  - Phí ma sát ước tính: $Friction = 0.8$ điểm.
  - Ngưỡng xác suất tối thiểu: $min\_prob = 0.60$ (60%).
- **Các bước tính toán:**
  1. Hinge Midpoint: $\hat{P}_{\text{adj}} = \frac{Upper_{0.9} + Lower_{0.9}}{2}$
  2. Độ biến động dự báo: $\hat{\sigma} = \max\left(\frac{Upper_{0.9} - Lower_{0.9}}{2 \times 1.645}, 0.5\right)$
  3. Lợi thế ròng sau phí ma sát:
     $$Edge_{\text{Long}} = (\hat{P}_{\text{adj}} - Close_{11:30}) - Friction$$
     $$Edge_{\text{Short}} = (Close_{11:30} - \hat{P}_{\text{adj}}) - Friction$$
  4. Điểm Z-Score và Xác suất có lãi:
     $$Z_{\text{Long}} = \frac{Edge_{\text{Long}}}{\hat{\sigma}}, \quad P(\text{Long}) = \Phi(Z_{\text{Long}})$$
     $$Z_{\text{Short}} = \frac{Edge_{\text{Short}}}{\hat{\sigma}}, \quad P(\text{Short}) = \Phi(Z_{\text{Short}})$$
  5. Ngưỡng Breakout động:
     $$\text{Threshold}_{\text{dynamic}} = \max(2 \times Friction, \; 0.25 \times ATR_{\text{morning}})$$
  6. Phân loại khuyến nghị:
     - **STRONG LONG:** $P(\text{Long}) \ge min\_prob$ và $(\hat{P}_{\text{adj}} - Close_{11:30}) > \text{Threshold}_{\text{dynamic}}$.
       - Entry: Quanh $Close_{11:30}$ lúc 13:00–13:05.
       - Target (TP): $Upper_{0.9}$.
       - Stop-Loss: $Close_{11:30} - \min(1.5 \times \hat{\sigma}, ATR_{\text{morning}})$.
     - **STRONG SHORT:** $P(\text{Short}) \ge min\_prob$ và $(Close_{11:30} - \hat{P}_{\text{adj}}) > \text{Threshold}_{\text{dynamic}}$.
       - Entry: Quanh $Close_{11:30}$ lúc 13:00–13:05.
       - Target (TP): $Lower_{0.9}$.
       - Stop-Loss: $Close_{11:30} + \min(1.5 \times \hat{\sigma}, ATR_{\text{morning}})$.
     - **WAIT / NO TRADE:** Ngược lại (xác suất lơ lửng, biên độ hẹp hoặc rủi ro bất định cao).

### 2.4. Module 4: CLI Interface & Execution Modes
File script hỗ trợ các tham số dòng lệnh qua `argparse`:
- `--mode`: `advisor` (mặc định) hoặc `backtest`.
- `--days`: Số phiên sử dụng để backtest (mặc định 20 phiên).
- `--date`: Ngày cụ thể để chạy advisor (định dạng `YYYY-MM-DD`, nếu không truyền sẽ lấy ngày cuối cùng trong dataset).
- `--fast`: Bật chế độ chạy nhanh (giảm `max_generations` và `num_validations` để hoàn thành trong 1-2 phút phục vụ demo/kiểm thử).

---

## 3. Quản lý Môi trường & Sửa lỗi Import Codebase (Package Integration)

### 3.1. Vấn đề hiện tại
Trong repository `autots4ps`:
- Tất cả các module trong `src/` sử dụng cú pháp `from autots.xxx import ...`.
- Cấu trúc `src/` hiện tại chưa có thư mục con hoặc package alias `autots`, khiến lệnh `import autots` hoặc `from autots.evaluator...` bị lỗi `ModuleNotFoundError: No module named 'autots'`.
- Thư mục `src/datasets` bị thiếu các hàm loader chuẩn của AutoTS (`load_daily`, `SyntheticDailyGenerator`, v.v.).

### 3.2. Giải pháp khắc phục triệt để
1. **Thiết lập Package Alias trong script và môi trường:**
   - Trong script `examples/afternoon_session_prediction.py`: cấu hình `sys.path` trỏ đúng vào `src/` và tạo alias module `sys.modules['autots'] = src_package` nếu cần, đảm bảo script luôn chạy được trực tiếp độc lập.
   - Bổ sung cấu hình `pyproject.toml` hoặc liên kết gói chuẩn để `uv run` nhận diện đúng package.
2. **Khôi phục `src/datasets`:** Sao chép các tệp loader từ `/home/v002353/Desktop/Git-clone/AutoTS/autots/datasets` vào `src/datasets` để đảm bảo toàn bộ `src/__init__.py` và các test case trong repo không bị văng lỗi.

---

## 4. Kế hoạch Kiểm thử & Tiêu chí Nghiệm thu (Testing & Success Criteria)

1. **Kiểm tra Resampling:**
   - Đọc dữ liệu `datasets/VN30F1M_5m.csv`, kiểm tra nén đủ 2 bar/ngày (Sáng 09:00–11:30 và Chiều 13:00–14:45), không bị NaN ở các giá trị OHLCV.
2. **Kiểm tra Chế độ Advisor (`--mode advisor --fast`):**
   - Chạy thành công và in ra bảng khuyến nghị đầy đủ (Giá Hinge, Dải 90%, Xác suất thắng, Ngưỡng động, Điểm vào lệnh, SL, TP).
3. **Kiểm tra Chế độ Backtest (`--mode backtest --days 10 --fast`):**
   - Chạy Walk-forward qua 10 phiên, tổng hợp bảng số liệu: ODA, Win Rate sau trừ phí ma sát, Net PnL, Max Drawdown.
