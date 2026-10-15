# AutoTS4PS Examples

Thư mục này chứa các ví dụ ứng dụng thực tế của thư viện `autots4ps`:

1. **[afternoon_session_prediction.py](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/afternoon_session_prediction.py)**:  
   Mô hình dự báo phiên chiều và hệ thống khuyến nghị giao dịch giờ nghỉ trưa (Lunchtime Advisor) cho phái sinh VN30F1M, kết hợp phân phối xác suất động sau phí ma sát (0.8 điểm), dải breakout ATR và SL/TP.  
   👉 **Tài liệu hướng dẫn chi tiết:** [docs/afternoon_session_prediction.md](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/docs/afternoon_session_prediction.md)

2. **[best_afternoon_template.json](file:///home/v002353/Desktop/SN/04.%20TPO/autots4ps/examples/best_afternoon_template.json)**:  
   Cấu hình template mô hình tối ưu đã được huấn luyện. Giúp chạy suy luận (Fast Inference) trong ~1 giây mà không cần train lại.
