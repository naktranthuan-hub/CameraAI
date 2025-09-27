# 🎯 CameraAI - Hệ thống An ninh Giám sát

> Hệ thống giám sát thông minh sử dụng AI để phát hiện việc sử dụng thiết bị di động trong trường học

[![GitHub](https://img.shields.io/badge/GitHub-CameraAI-blue?logo=github)](https://github.com/naktranthuan-hub/CameraAI)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)

---

## ✨ Tính năng chính

🔍 **Phát hiện điện thoại**: Sử dụng YOLOv11 để detect điện thoại di động  
📹 **Đa camera**: Hỗ trợ Webcam, RTSP, HTTP, Video file  
🤖 **AI thông minh**: Hai chế độ hoạt động:
   - **Mode A**: Phát hiện hành vi CALL/VIEW/TEXT (YOLOv11 + MediaPipe Pose)
   - **Mode B**: Chỉ phát hiện sự hiện diện điện thoại
🖥️ **Dashboard trực quan**: Giao diện Streamlit với real-time monitoring  
📊 **Lưu trữ vi phạm**: Tự động capture và log vi phạm

---

## 🚀 Cài đặt nhanh

### Bước 1: Clone repository
```bash
git clone https://github.com/naktranthuan-hub/CameraAI.git
cd CameraAI
```

### Bước 2: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng
```bash
streamlit run app_final.py
```

🎉 **Thành công!** Truy cập `http://localhost:8501` để sử dụng

---

## 📁 Cấu trúc dự án

```
CameraAI/
├── 📄 app_final.py          # Ứng dụng chính
├── 📄 app.py               # Phiên bản đơn giản  
├── 📋 requirements.txt     # Dependencies
├── 🤖 yolo11n.pt          # Model YOLOv11
├── 🖼️ lythuongkiet.jpg    # Logo trường
├── 🖼️ nguyenanninh.jpg    # Logo trường
└── 📁 vipham/             # Thư mục vi phạm
    └── 📊 log.csv         # File log
```

---

## 🛠️ Hướng dẫn sử dụng

1. **➕ Thêm camera**: Sử dụng sidebar để add nguồn camera
2. **⚙️ Cấu hình**: Chọn chế độ và điều chỉnh thông số  
3. **▶️ Khởi động**: Click "Start ALL" để bắt đầu giám sát
4. **👁️ Giám sát**: Theo dõi dashboard real-time
5. **📸 Kết quả**: Check thư mục `vipham/` để xem vi phạm
- GPU (tùy chọn, để tăng hiệu suất)

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

## 🤳 Phát hiện hành vi CALL / VIEW / TEXT
Ngoài việc chỉ phát hiện điện thoại, hệ thống còn phân loại hành vi sử dụng:

- **CALL**: điện thoại gần tai/má → hành vi nghe/gọi.  
- **VIEW**: điện thoại đặt trước mặt, đầu cúi xuống → hành vi xem màn hình.  
- **TEXT**: giống VIEW nhưng điện thoại gần cổ tay → hành vi nhắn tin.  

Thuật toán kết hợp:
- YOLOv11 để phát hiện vị trí điện thoại.  
- MediaPipe Pose để lấy keypoints (mũi, tai, vai, cổ tay).  
- Quy tắc hình học (vùng má–tai, khoảng cách, góc cúi đầu) để phân loại.

## 🖥️ Dashboard giám sát
- Giao diện Streamlit, hỗ trợ **nhiều camera cùng lúc**.  
- Cho phép **Add/Remove camera** trực tiếp từ sidebar.  
- Mỗi camera hiển thị:
  - Video real-time với bounding box.
  - Thông tin FPS, số lượng đối tượng, trạng thái vi phạm.
  - Tuỳ chỉnh tham số YOLO và tham số hành vi (ngưỡng pitch, khoảng cách...).  
- Header hỗ trợ hiển thị **2 logo trường học** và tiêu đề hệ thống.

## 🔗 Kết nối nguồn
Hệ thống hỗ trợ nhiều loại nguồn video:

- **Webcam (nội/USB):** Chọn chỉ số webcam (0,1,2...).
- **RTSP (H264/H265):** `rtsp://username:password@192.168.1.100:554/Streaming/Channels/101`
- **HTTP MJPEG:** `http://192.168.1.101:8080/video`
- **HTTP Snapshot (.jpg):** `http://192.168.1.102/jpg/image.jpg`
- **Video file:** Hỗ trợ `.mp4`, `.avi`, `.mov`, `.mkv`.

## 📂 Cấu trúc dữ liệu
- Thư mục `vipham/`: chứa các ảnh vi phạm.  
- File `vipham/log.csv`: ghi log theo định dạng:  
  - **Chế độ A (CALL/VIEW/TEXT):** `[timestamp, camera_name, image_path, intent]`  
  - **Chế độ B (chỉ phát hiện điện thoại):** `[timestamp, camera_name, image_path]`

## 🚀 Chạy ứng dụng
```bash
streamlit run app_final.py
```

## 🧪 Kiểm tra Dependencies (trước khi push)
```bash
# Trên Linux/Mac
bash test-dependencies.sh

# Trên Windows (PowerShell)
python -c "import streamlit, cv2, ultralytics, numpy, mediapipe; print('✅ All dependencies OK')"
```

## 🔧 Troubleshooting

### Lỗi OpenCV version không khớp
Nếu gặp lỗi version mismatch:

**Vấn đề thường gặp:** OpenCV package 4.10.0.82 có thể report runtime version là 4.10.0, 4.11.0, hoặc khác tùy môi trường.

**Giải pháp:**
```bash
# 1. Clean install OpenCV
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python==4.10.0.82

# 2. Test functionality (quan trọng hơn version number)
python -c "
import cv2, numpy as np
img = np.zeros((100,100,3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'✅ OpenCV {cv2.__version__} works correctly')
"

# 3. Nếu functionality OK, version number không quan trọng
```

### Lỗi MediaPipe
```bash
pip install mediapipe --upgrade
```

### Test GitHub Actions locally
Sử dụng workflow `simple-test.yml` để test nhanh:
```bash
# Trigger manual workflow trên GitHub
# hoặc test local với Docker:
docker run --rm -v $(pwd):/workspace -w /workspace python:3.10 bash -c "
  pip install -r requirements.txt && 
  python -m py_compile app_final.py && 
  python -c 'import streamlit, cv2, ultralytics'
"
```

## 📌 Ghi chú
- Sử dụng `yolo11n.pt` (phiên bản nhỏ) để chạy trên CPU.  
- Có thể đổi sang `yolo11s.pt` hoặc lớn hơn nếu có GPU.  
- Tham số mặc định có thể tinh chỉnh ngay trên Dashboard.

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo pull request hoặc báo cáo lỗi qua Issues.

## Giấy phép

MIT License