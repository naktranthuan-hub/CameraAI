# Hướng dẫn cài đặt môi trường CameraAI

## Giới thiệu
File `environment.yml` này chứa tất cả các thư viện cần thiết để chạy ứng dụng Camera AI, bao gồm:

- OpenCV 4.10.0.82 (như yêu cầu)
- Ultralytics (cho YOLOv11)
- Streamlit (cho giao diện web)
- MediaPipe (cho phát hiện pose)
- Và các thư viện hỗ trợ khác

## Cách sử dụng

### 1. Cài đặt Conda/Miniconda
Đầu tiên, đảm bảo bạn đã cài đặt Conda hoặc Miniconda trên hệ thống.

### 2. Tạo môi trường từ file environment.yml
```bash
# Tạo môi trường mới từ file environment.yml
conda env create -f environment.yml

# Kích hoạt môi trường
conda activate cameraai
```

### 3. Kiểm tra cài đặt
```bash
# Kiểm tra các thư viện đã được cài đặt
python -c "import cv2, numpy, streamlit, ultralytics; print('Cài đặt thành công!')"
```

### 4. Chạy ứng dụng
```bash
# Chạy ứng dụng Camera AI
streamlit run app.py
```

## Vị trí file
File `environment.yml` được đặt tại thư mục gốc của dự án:
```
CameraAI/
├── environment.yml  ← File môi trường conda
├── app.py
├── app_final.py
├── README.md
└── ...
```

## Các thư viện chính được cài đặt

### Từ Conda
- `python=3.9`: Phiên bản Python
- `numpy`: Thư viện tính toán số
- `opencv=4.10.0.82`: OpenCV cho xử lý ảnh/video
- `pillow`: Thư viện xử lý ảnh PIL

### Từ pip
- `ultralytics`: Framework YOLOv11
- `streamlit`: Framework web app
- `mediapipe`: Thư viện phát hiện pose của Google

## Troubleshooting

### Nếu gặp lỗi khi tạo environment:
1. Cập nhật conda: `conda update conda`
2. Thử tạo lại environment: `conda env remove -n cameraai && conda env create -f environment.yml`

### Nếu OpenCV không hoạt động:
```bash
# Thử cài đặt lại OpenCV
conda install -c conda-forge opencv=4.10.0.82
```

### Nếu thiếu các thư viện khác:
```bash
# Cài đặt thêm thư viện nếu cần
pip install <package-name>
```