# CameraAI - Hệ thống An ninh Giám sát

Hệ thống giám sát an ninh sử dụng AI để phát hiện việc sử dụng thiết bị di động trong trường học.

## Tính năng chính

- **Phát hiện điện thoại**: Sử dụng YOLOv11 để detect điện thoại di động
- **Đa camera**: Hỗ trợ nhiều nguồn camera (Webcam, RTSP, HTTP, Video file)
- **Hai chế độ hoạt động**:
  - Mode A: Phát hiện hành vi CALL/VIEW/TEXT (sử dụng MediaPipe Pose)
  - Mode B: Chỉ phát hiện sự hiện diện của điện thoại
- **Giao diện web**: Sử dụng Streamlit với dashboard trực quan
- **Lưu trữ vi phạm**: Tự động lưu ảnh và log khi phát hiện vi phạm

## Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/naktranthuan-hub/CameraAI.git
cd CameraAI
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng
```bash
streamlit run app_final.py
```

## Cấu trúc dự án

```
CameraAI/
├── app_final.py          # Ứng dụng chính
├── app.py               # Phiên bản đơn giản
├── requirements.txt     # Danh sách thư viện cần thiết
├── yolo11n.pt          # Model YOLO
├── lythuongkiet.jpg    # Logo trường
├── nguyenanninh.jpg    # Logo trường
└── vipham/             # Thư mục lưu vi phạm
    └── log.csv         # File log
```

## Hướng dẫn sử dụng

1. **Thêm camera**: Sử dụng sidebar để thêm nguồn camera mới
2. **Cấu hình**: Chọn chế độ hoạt động và các thông số
3. **Khởi động**: Click "Start ALL" để bắt đầu giám sát
4. **Xem kết quả**: Theo dõi dashboard và check thư mục `vipham/` để xem các vi phạm được lưu

## Yêu cầu hệ thống

- Python 3.9+
- Camera hoặc nguồn video
- RAM: tối thiểu 4GB
- GPU (tùy chọn, để tăng hiệu suất)

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo pull request hoặc báo cáo lỗi qua Issues.

## Giấy phép

MIT License