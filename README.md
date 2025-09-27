<<<<<<< HEAD
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
=======
# 📷 Dashboard Camera AI – Phát hiện sử dụng điện thoại trong trường học

## Giới thiệu
Dự án xây dựng **hệ thống giám sát thông minh** sử dụng mô hình học sâu YOLOv11 kết hợp với MediaPipe Pose để:
- Phát hiện học sinh có mang điện thoại.
- Phân loại hành vi **CALL (nghe/gọi)**, **VIEW (xem màn hình)**, **TEXT (nhắn tin)**.
- Ghi lại hình ảnh vi phạm và lưu nhật ký CSV.
- Hỗ trợ giám sát đa camera (Webcam, camera IP, stream RTSP/HTTP, file video).

Ứng dụng triển khai trên **Streamlit Dashboard**, dễ dàng cấu hình và mở rộng.

## Mục tiêu
- Xây dựng công cụ hỗ trợ giáo viên quản lý lớp học.
- Ngăn chặn việc **sử dụng điện thoại trái phép** trong giờ học.
- Đem lại môi trường học tập tập trung, nghiêm túc.
- Ứng dụng thử nghiệm cho các hội thi khoa học kỹ thuật học sinh.

---

## 🚩 Phát hiện điện thoại (YOLOv11)
YOLOv11 nhận diện nhãn `cell phone` từ khung hình. Khi có điện thoại trong ảnh:
- Vẽ bounding box quanh điện thoại.
- Ghi nhận và lưu khung hình vào thư mục `vipham/`.

Ví dụ đoạn mã phát hiện điện thoại:

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
frame = cv2.imread("test.jpg")
results = model.predict(source=frame, conf=0.3, iou=0.5)[0]
vis = results.plot()   # vẽ bounding box
cv2.imwrite("output.jpg", vis)
```

---

## 🤳 Phát hiện hành vi CALL / VIEW / TEXT
Ngoài việc chỉ phát hiện điện thoại, hệ thống còn phân loại hành vi sử dụng:

- **CALL**: điện thoại gần tai/má → hành vi nghe/gọi.  
- **VIEW**: điện thoại đặt trước mặt, đầu cúi xuống → hành vi xem màn hình.  
- **TEXT**: giống VIEW nhưng điện thoại gần cổ tay → hành vi nhắn tin.  

Thuật toán kết hợp:
- YOLOv11 để phát hiện vị trí điện thoại.  
- MediaPipe Pose để lấy keypoints (mũi, tai, vai, cổ tay).  
- Quy tắc hình học (vùng má–tai, khoảng cách, góc cúi đầu) để phân loại.  

---

## 🖥️ Dashboard giám sát
- Giao diện Streamlit, hỗ trợ **nhiều camera cùng lúc**.  
- Cho phép **Add/Remove camera** trực tiếp từ sidebar.  
- Mỗi camera hiển thị:
  - Video real-time với bounding box.
  - Thông tin FPS, số lượng đối tượng, trạng thái vi phạm.
  - Tuỳ chỉnh tham số YOLO và tham số hành vi (ngưỡng pitch, khoảng cách...).  
- Header hỗ trợ hiển thị **2 logo trường học** và tiêu đề hệ thống.

---

## 🔗 Kết nối nguồn
Hệ thống hỗ trợ nhiều loại nguồn video:

- **Webcam (nội/USB):**
  - Chọn chỉ số webcam (0,1,2...).
- **RTSP (H264/H265):**
  - Ví dụ:  
    ```
    rtsp://username:password@192.168.1.100:554/Streaming/Channels/101
    ```
- **HTTP MJPEG:**
  - Ví dụ:  
    ```
    http://192.168.1.101:8080/video
    ```
- **HTTP Snapshot (.jpg):**
  - Ví dụ:  
    ```
    http://192.168.1.102/jpg/image.jpg
    ```
- **Video file:**  
  - Hỗ trợ `.mp4`, `.avi`, `.mov`, `.mkv`.

---

## 📂 Cấu trúc dữ liệu
- Thư mục `vipham/`: chứa các ảnh vi phạm.  
- File `vipham/log.csv`: ghi log theo định dạng:  
  - **Chế độ A (CALL/VIEW/TEXT):** `[timestamp, camera_name, image_path, intent]`  
  - **Chế độ B (chỉ phát hiện điện thoại):** `[timestamp, camera_name, image_path]`

---

## 🚀 Chạy ứng dụng
```bash
streamlit run app.py
```

---

## 📌 Ghi chú
- Sử dụng `yolo11n.pt` (phiên bản nhỏ) để chạy trên CPU.  
- Có thể đổi sang `yolo11s.pt` hoặc lớn hơn nếu có GPU.  
- Tham số mặc định có thể tinh chỉnh ngay trên Dashboard.
>>>>>>> a950caa497aff718b014f2451f0039091d63521b
