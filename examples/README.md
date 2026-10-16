# AutoTS4PS Examples

Thư mục này chứa các ví dụ ứng dụng thực tế của thư viện `autots4ps`:

1. **[afternoon_session_prediction.py](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/afternoon_session_prediction.py)**:  
   Mô hình dự báo giá đóng cửa phiên chiều (Time-Series Forecasting) và hệ thống đánh giá chỉ số định lượng (MAE, RMSE, MAPE, sMAPE, ODA, Coverage 90%) trên tập test cho phái sinh VN30F1M.  
   👉 **Tài liệu hướng dẫn chi tiết:** [docs/afternoon_session_prediction.md](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/docs/afternoon_session_prediction.md)

2. **[best_afternoon_template.json](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/best_afternoon_template.json)**:  
   Cấu hình template mô hình tối ưu đã được huấn luyện trên tập Train (`datasets/train.csv`). Giúp chạy suy luận nhanh (Fast Inference) trong ~0.4 giây/phiên.

3. **[afternoon_test_predictions.csv](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/afternoon_test_predictions.csv)**:  
   Bảng kết quả chi tiết dự báo walk-forward từng phiên trên toàn bộ tập Test (`datasets/test.csv`).

4. **[afternoon_sl_tp_event_risk.py](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/afternoon_sl_tp_event_risk.py)**:  
   **Hướng 1 - Event Risk & First-Passage SL/TP:** Mô phỏng quỹ đạo đa bước trong phiên chiều (13:00–14:45) dựa trên các mẫu hình tương đồng (Nearest-Neighbor Motif Paths). Tính xác suất thực nghiệm chạm Chốt lời (TP) trước hay Cắt lỗ (SL) trước, kỳ vọng lợi nhuận ròng (EV) sau ma sát và đưa ra khuyến nghị vào lệnh có kiểm soát rủi ro.

5. **[afternoon_extremes_volatility.py](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/afternoon_extremes_volatility.py)**:  
   **Hướng 2 - Cực trị High/Low & Chế độ Biến động:** Dự báo bao phủ cực trị Đỉnh (`High`), Đáy (`Low`) và Biên độ (`Range = High - Low`) cùng dải bao phủ 90% (Envelope Coverage). Tự động phân loại chế độ thị trường (`EXPANSION / BREAKOUT` vs `COMPRESSION / SIDEWAY`) và đề xuất chiến lược giao dịch tương ứng (Breakout Stop ngoài hộp sáng vs Mean-Reversion Grid).

