# AutoTS4PS Examples

Thư mục này chứa các ví dụ ứng dụng thực tế của thư viện `autots4ps`:

1. **[afternoon_session_prediction.py](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/afternoon_session_prediction.py)**:  
   Mô hình dự báo giá đóng cửa phiên chiều (Time-Series Forecasting) và hệ thống đánh giá chỉ số định lượng (MAE, RMSE, MAPE, sMAPE, ODA, Coverage 90%) trên tập test cho phái sinh VN30F1M.  
   👉 **Tài liệu hướng dẫn chi tiết:** [docs/afternoon_session_prediction.md](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/docs/afternoon_session_prediction.md)

2. **[best_afternoon_template.json](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/best_afternoon_template.json)**:  
   Cấu hình template mô hình tối ưu đã được huấn luyện trên tập Train (`datasets/train.csv`). Giúp chạy suy luận nhanh (Fast Inference) trong ~0.4 giây/phiên.

3. **[afternoon_test_predictions.csv](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/afternoon_test_predictions.csv)**:  
   Bảng kết quả chi tiết dự báo walk-forward từng phiên trên toàn bộ tập Test (`datasets/test.csv`).
