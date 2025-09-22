# -*- coding: utf-8 -*-
"""
Camera AI – Phát hiện học sinh sử dụng điện thoại (YOLOv11 + Streamlit)
- Khi phát hiện 'cell phone' trong khung hình, chụp ảnh vào vipham/
- Ghi CSV: timestamp, camera_location, image_path -> vipham/log.csv
- Nguồn video: Webcam / RTSP-HTTP URL / Video file
"""

import os, csv, time
from datetime import datetime
from pathlib import Path
from collections import Counter

import cv2
import streamlit as st
from ultralytics import YOLO

# -----------------------------
# Cấu hình app & thư mục output
# -----------------------------
st.set_page_config(page_title="Camera AI - Phát hiện sử dụng điện thoại", layout="wide")

OUTPUT_DIR = Path("D:\\ThuNghiem\\PythonApplication1\\vipham")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "log.csv"

if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "camera_location", "image_path"])

# -----------------------------
# Cache model để tải 1 lần
# -----------------------------
#@st.cache_resource
def load_model():
    # Nhẹ và nhanh cho máy cá nhân/Colab
    return YOLO("D:\ThuNghiem\PythonApplication1\yolo11n.pt")

model = load_model()
st.success("Đã tải mô hình YOLOv11 (yolo11n.pt).")

# -----------------------------
# Sidebar cấu hình
# -----------------------------
st.sidebar.header("Cấu hình")
camera_location = st.sidebar.text_input("Vị trí camera", value="Lớp 9A4 - góc trái")

source_type = st.sidebar.selectbox(
    "Nguồn video",
    ["Webcam", "RTSP/HTTP URL", "Video file (.mp4, .avi, ...)"]
)

source_value = None
if source_type == "Webcam":
    cam_index = st.sidebar.number_input("Chỉ số webcam", min_value=0, value=0, step=1)
    source_value = cam_index
elif source_type == "RTSP/HTTP URL":
    source_value = st.sidebar.text_input("Nhập URL stream", value="")
else:
    uploaded = st.sidebar.file_uploader("Tải video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        tmp_path = Path(f"temp_{uploaded.name}")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        source_value = str(tmp_path)

conf_thres = st.sidebar.slider("Confidence", 0.1, 0.9, 0.35, 0.05)
iou_thres  = st.sidebar.slider("IoU",        0.1, 0.9, 0.50, 0.05)
imgsz      = st.sidebar.select_slider("Image size", options=[320, 480, 640, 768, 960, 1280], value=640)
save_cooldown = st.sidebar.slider("Khoảng cách mỗi lần lưu (giây)", 0, 30, 5, 1)

st.sidebar.caption("Mẹo: giảm imgsz nếu bị giật; giảm conf nếu bỏ sót điện thoại.")

# -----------------------------
# Tiện ích
# -----------------------------
def open_capture(src):
    cap = cv2.VideoCapture(src)
    # Giảm trễ với stream
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap

def save_violation_frame(frame_bgr, location_text):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = OUTPUT_DIR / f"vipham_{ts}.jpg"
    cv2.imwrite(str(img_path), frame_bgr)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), location_text, str(img_path)])

    return img_path

def ensure_session_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "last_saved" not in st.session_state:
        st.session_state.last_saved = 0.0

ensure_session_state()

# -----------------------------
# Giao diện chính
# -----------------------------
st.title("📷 Camera AI – Phát hiện học sinh sử dụng điện thoại (YOLOv11)")
st.write("Phát hiện **cell phone** → chụp ảnh vào **vipham/** và ghi log vào **vipham/log.csv**.")

start_col, stop_col = st.columns(2)
start_clicked = start_col.button("▶️ Bắt đầu giám sát", use_container_width=True, key="start_btn")
stop_clicked  = stop_col.button("⏹️ Dừng", use_container_width=True, key="stop_btn")

frame_holder = st.empty()
status_holder = st.empty()

if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

# -----------------------------
# Vòng lặp giám sát
# -----------------------------
if st.session_state.running:
    if source_type != "Webcam" and not source_value:
        st.error("Vui lòng cung cấp URL hoặc tải video.")
        st.session_state.running = False
    else:
        cap = open_capture(source_value if source_type != "Webcam" else cam_index)
        if not cap or not cap.isOpened():
            st.error("Không mở được nguồn video. Kiểm tra webcam/URL/file.")
            st.session_state.running = False
        else:
            try:
                while st.session_state.running and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        status_holder.warning("Hết video hoặc mất kết nối.")
                        break

                    # Suy luận YOLO
                    r = model.predict(
                        source=frame, conf=conf_thres, iou=iou_thres,
                        imgsz=imgsz, verbose=False
                    )[0]

                    vis = r.plot()  #BGR (đã vẽ bbox)
                    names = r.names
                    ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
                    label_counts = Counter([names[i] for i in ids]) if len(ids) else Counter()
                    has_phone = label_counts.get("cell phone", 0) > 0

                    # Hiển thị
                    frame_holder.image(
                        cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                        caption=f"Đếm theo lớp: {dict(label_counts)}",
                        use_column_width=True
                    )

                    # Lưu vi phạm theo cooldown
                    now = time.time()
                    if has_phone and (now - st.session_state.last_saved >= save_cooldown):
                        saved = save_violation_frame(vis, camera_location)
                        st.session_state.last_saved = now
                        status_holder.success(f"📸 Đã lưu vi phạm: {saved}")

                    # Nhường quyền cho UI cập nhật (tránh chặn sự kiện nút Dừng)
                    time.sleep(0.01)

            finally:
                cap.release()
                status_holder.info("Đã dừng giám sát.")
else:
    st.info("Nhấn **Bắt đầu giám sát** để khởi động.")
