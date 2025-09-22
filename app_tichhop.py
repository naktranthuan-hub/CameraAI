# -*- coding: utf-8 -*-
"""
Camera AI — 2 chế độ:
(A) Khi sử dụng (CALL/VIEW/TEXT) — YOLOv11 + MediaPipe Pose + rule + cửa sổ thời gian
(B) Chỉ cần thấy điện thoại — dùng YOLOv11, lưu theo cooldown (theo đúng script đơn giản)

Ảnh -> vipham/vipham_YYYYMMDD_hhmmss_ms.jpg
CSV:
  - Chế độ (A): 4 cột  [timestamp, camera_location, image_path, intent]
  - Chế độ (B): 3 cột  [timestamp, camera_location, image_path]  (đúng format script cũ)
"""
import csv, time
from datetime import datetime
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_unicode_text(img_bgr, text, position, font_path="arial.ttf", font_size=32, color=(0, 0, 255)):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # Load a TTF font that supports Vietnamese (e.g., Arial, Roboto, etc.)
    font = ImageFont.truetype(font_path, font_size)
    # Draw text
    draw.text(position, text, font=font, fill=color)
    # Convert back to OpenCV image (BGR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ================== MediaPipe Pose (CPU) ==================
POSE = None
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    POSE = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        enable_segmentation=False, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
except Exception:
    POSE = None  # fallback cho chế độ A nếu thiếu mediapipe

# ================== App / IO ==================
st.set_page_config(page_title="Camera AI - Phát hiện điện thoại / sử dụng", layout="wide")

pathVisual="d:\\ThuNghiem\\PythonApplication1"

OUTPUT_DIR = Path(f"{pathVisual}\\vipham"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "log.csv"

@st.cache_resource
def load_model():
    return YOLO(f"{pathVisual}\\yolo11n.pt")  # nhẹ, chạy CPU ổn
model = load_model()
st.success("Đã tải YOLOv11 (yolo11n.pt).")

# ================== Sidebar ==================
st.sidebar.header("Chế độ")
mode = st.sidebar.radio(
    "Chọn chế độ giám sát",
    ["Khi sử dụng (CALL/VIEW/TEXT)", "Chỉ cần thấy điện thoại"],
    index=0
)

st.sidebar.header("Nguồn video & YOLO")
camera_location = st.sidebar.text_input("Vị trí camera", value="Lớp 9A4 - góc trái")

source_type = st.sidebar.selectbox("Nguồn video", ["Webcam","RTSP/HTTP URL","Video file (.mp4, .avi, ...)"])
source_value = None
cam_index = 0
if source_type == "Webcam":
    cam_index = st.sidebar.number_input("Chỉ số webcam", min_value=0, value=0, step=1)
    source_value = cam_index
elif source_type == "RTSP/HTTP URL":
    source_value = st.sidebar.text_input("Nhập URL stream", value="")
else:
    up = st.sidebar.file_uploader("Tải video", type=["mp4","avi","mov","mkv"])
    if up is not None:
        tmp = Path(f"temp_{up.name}"); tmp.write_bytes(up.read())
        source_value = str(tmp)

conf_thres = st.sidebar.slider("YOLO Confidence", 0.10, 0.90, 0.30, 0.05)
iou_thres  = st.sidebar.slider("YOLO IoU",        0.10, 0.90, 0.50, 0.05)
imgsz      = st.sidebar.select_slider("Image size", options=[320,480,640,768,960,1280], value=640)

# Tham số riêng theo chế độ
if mode.startswith("Khi sử dụng"):
    st.sidebar.divider()
    st.sidebar.subheader("Ngưỡng Ý ĐỊNH (chuẩn theo head_size)")
    pitch_deg_thr = st.sidebar.slider("Đầu cúi xuống ≥ (°)", 5, 45, 20, 1)
    near_ear_k    = st.sidebar.slider("CALL: gần tai (× head_size)", 0.10, 1.00, 0.65, 0.05)
    k_lateral     = st.sidebar.slider("Bộ lọc lệch ngang (× head_size)", 0.8, 3.0, 1.5, 0.1)
    k_wrist_head  = st.sidebar.slider("TEXT: gần cổ tay (× head_size)", 0.6, 3.0, 1.4, 0.1)
    intent_seconds= st.sidebar.slider("Duy trì Ý ĐỊNH (giây)", 0.5, 5.0, 1.2, 0.1)
    smooth_center = st.sidebar.checkbox("Làm mượt vị trí điện thoại", True)
    debug_overlay = st.sidebar.checkbox("Overlay debug (tai/má…)", False)
    show_debug    = st.sidebar.checkbox("Hiển thị bảng debug", True)
else:
    st.sidebar.divider()
    st.sidebar.subheader("Thiết lập lưu (điện thoại)")
    save_cooldown = st.sidebar.slider("Khoảng cách mỗi lần lưu (giây)", 0, 30, 5, 1)
    smooth_center = st.sidebar.checkbox("Làm mượt vị trí điện thoại", True)
    show_debug    = st.sidebar.checkbox("Hiển thị bảng debug", True)
    st.sidebar.caption("Mẹo: giảm imgsz nếu bị giật; giảm conf nếu bỏ sót điện thoại.")

# ================== Utils ==================
def ensure_csv_header_if_needed():
    """Tạo header nếu file chưa có. Header sẽ theo chế độ hiện tại:
       - Chế độ A: 4 cột
       - Chế độ B: 3 cột
       (Nếu đã tồn tại, giữ nguyên; hệ thống vẫn có thể ghi thêm dòng khác số cột — CSV viewer vẫn đọc được)"""
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            if mode.startswith("Khi sử dụng"):
                csv.writer(f).writerow(["timestamp", "camera_location", "image_path", "intent"])
            else:
                csv.writer(f).writerow(["timestamp", "camera_location", "image_path"])

def open_capture(src):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # giảm trễ stream
    return cap

def save_frame_simple(frame_bgr, location_text):
    """Ghi ảnh + CSV 3 cột như script tối giản."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = OUTPUT_DIR / f"vipham_{ts}.jpg"
    cv2.imwrite(str(img_path), frame_bgr)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), location_text, str(img_path)])
    return img_path

def save_frame_intent(frame_bgr, location_text, intent_label):
    """Ghi ảnh + CSV 4 cột cho chế độ Ý ĐỊNH."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = OUTPUT_DIR / f"vipham_{ts}.jpg"
    cv2.imwrite(str(img_path), frame_bgr)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"), location_text, str(img_path), intent_label
        ])
    return img_path

def ensure_state():
    ss = st.session_state
    ss.setdefault("running", False)
    ss.setdefault("last_saved", 0.0)     # cho chế độ B
    ss.setdefault("window", deque(maxlen=120))  # cho chế độ A (cửa sổ thời gian)
    ss.setdefault("center_smooth", deque(maxlen=7))
    ss.setdefault("fps_hist", deque(maxlen=30))
    ss.setdefault("prev_ts", time.time())
ensure_state()
ensure_csv_header_if_needed()

# ---- Hình học / Pose helpers (chế độ A) ----
def _to_xy(lm, w, h): return np.array([lm.x*w, lm.y*h], dtype=np.float32)
def _dist(a, b): return float(np.linalg.norm(a-b))
def _angle_deg(v1, v2):
    denom = (np.linalg.norm(v1)*np.linalg.norm(v2))+1e-6
    cosv = np.clip(np.dot(v1, v2)/denom, -1, 1)
    return float(np.degrees(np.arccos(cosv)))
def _dist_point_to_segment(p, a, b):
    if a is None or b is None or p is None: return 1e9
    ap = p - a; ab = b - a
    t = float(np.clip(np.dot(ap, ab)/(np.dot(ab,ab)+1e-6), 0, 1))
    proj = a + t*ab
    return _dist(p, proj)

def estimate_geometry(lms, w, h, draw=None):
    from mediapipe.python.solutions.pose import PoseLandmark as P
    def g(idx):
        lm = lms[idx]
        return _to_xy(lm, w, h) if getattr(lm, 'visibility', 1.0) > 0.35 else None

    nose = g(P.NOSE); le = g(P.LEFT_EAR); re = g(P.RIGHT_EAR)
    lw   = g(P.LEFT_WRIST); rw = g(P.RIGHT_WRIST)
    ls   = g(P.LEFT_SHOULDER); rs = g(P.RIGHT_SHOULDER)

    shoulder_width = _dist(ls, rs) if (ls and rs) else None
    head_size = _dist(le, re) if (le is not None and re is not None) else (shoulder_width*0.35 if shoulder_width else None)

    pitch = None
    if nose is not None and ls is not None and rs is not None:
        mid = (ls + rs)/2
        v = nose - mid
        pitch = _angle_deg(v/np.linalg.norm(v+1e-6), np.array([0,-1],dtype=np.float32))

    if head_size is None and shoulder_width:
        head_size = shoulder_width*0.35
    if le is None and nose is not None and ls is not None and head_size:
        dir_face = nose - ls; dir_face = dir_face/(np.linalg.norm(dir_face)+1e-6)
        ortho = np.array([-dir_face[1], dir_face[0]], dtype=np.float32)
        le = nose + ortho * (0.45*head_size)
    if re is None and nose is not None and rs is not None and head_size:
        dir_face = nose - rs; dir_face = dir_face/(np.linalg.norm(dir_face)+1e-6)
        ortho = np.array([ dir_face[1],-dir_face[0]], dtype=np.float32)
        re = nose + ortho * (0.45*head_size)

    if mode.startswith("Khi sử dụng") and draw is not None:
        for p,c in [(nose,(255,255,255)),(le,(0,200,0)),(re,(0,200,0)),
                    (lw,(255,0,0)),(rw,(255,0,0)),(ls,(0,160,255)),(rs,(0,160,255))]:
            if p is not None: cv2.circle(draw, tuple(p.astype(int)), 4, c, -1)

    return dict(nose=nose, le=le, re=re, lw=lw, rw=rw, ls=ls, rs=rs,
                head_size=head_size, shoulder_width=shoulder_width, pitch=pitch)

def _safe(v, default=0.0):
    return float(v) if v is not None else float(default)

def classify_intent_with_pose(
    phone_xy, geom,
    k_wrist_head=1.4,     # TEXT: gần cổ tay
    pitch_thr=20,         # VIEW/TEXT cần cúi đầu
    # ==== Các trọng số/threshold mới ====
    w_ear=2.0,            # điểm cho gần tai/má
    w_lateral=1.1,        # điểm cho lệch ngang (CALL thường lệch rõ)
    w_height=1.0,         # điểm cho độ cao tương đối (mũi→cằm)
    w_wrist=1.0,          # điểm cho cổ tay nằm giữa ear–phone
    call_region_xk=1.9,   # bề rộng vùng má theo head_size (từ mũi sang 2 bên)
    call_region_y_up=0.50,# vùng má bắt đầu từ mũi đi lên/ xuống  (± theo hd)
    call_region_y_down=0.80,
    tall_ratio=1.20       # bbox phone “dựng dọc”: h/w >= tall_ratio
):
    """
    Phân loại: 'CALL' | 'VIEW' | 'TEXT' | None
    - Ưu tiên CALL khi phone nằm trong Cheek Region (má–tai mở rộng) hoặc gần tai.
    - VIEW/TEXT yêu cầu pitch >= pitch_thr.
    """
    if phone_xy is None:
        return None

    nose = geom.get("nose"); le=geom.get("le"); re=geom.get("re")
    lw=geom.get("lw"); rw=geom.get("rw"); ls=geom.get("ls"); rs=geom.get("rs")
    pitch = float(geom.get("pitch") or 0.0)

    # ===== head_size ổn định =====
    hd = geom.get("head_size")
    if hd is None or hd <= 0:
        sw = geom.get("shoulder_width") or 0
        hd = sw*0.35 if sw>0 else 0.08 * 720
    hd = max(hd, 1.0)

    # Ước lượng cằm từ mũi→tâm vai
    chin = None
    if nose is not None and ls is not None and rs is not None:
        mid = (ls + rs) / 2.0
        v = mid - nose
        v = v / (np.linalg.norm(v) + 1e-6)
        chin = nose + v * (1.1 * hd)

    # ===== 1) Cheek Region “đè ngưỡng” CALL =====
    # Khung chữ nhật 2 bên mặt: từ mũi sang trái/phải call_region_xk*hd;
    # theo trục dọc: [nose_y - call_region_y_up*hd, nose_y + call_region_y_down*hd]
    def in_cheek_region(p):
        if nose is None:
            return False
        x, y = p[0], p[1]
        y1 = nose[1] - call_region_y_up*hd
        y2 = nose[1] + call_region_y_down*hd
        # 2 nửa mặt: trái và phải
        return (y1 <= y <= y2) and (abs(x - nose[0]) <= call_region_xk*hd)

    # Nếu phone nằm trong Cheek Region -> gán “ứng viên CALL mạnh”
    in_cheek = in_cheek_region(phone_xy)

    # ===== 2) Tín hiệu phụ trợ (điểm) =====
    def _dist(a,b): 
        return float(np.linalg.norm(a-b)) if (a is not None and b is not None) else 1e9
    def _segdist(p,a,b):
        if a is None or b is None or p is None: return 1e9
        ap = p - a; ab = b - a
        t = float(np.clip(np.dot(ap,ab)/(np.dot(ab,ab)+1e-6), 0, 1))
        proj = a + t*ab
        return float(np.linalg.norm(p-proj))

    d_ear_min = min(_dist(phone_xy, le), _dist(phone_xy, re))
    d_cheek_seg = min(_segdist(phone_xy, nose, ls), _segdist(phone_xy, nose, rs)) if nose is not None else 1e9
    lateral = abs(phone_xy[0] - (nose[0] if nose is not None else phone_xy[0]))

    # Cao độ cho CALL/VĐ
    h_call = 0.0; h_view = 0.0
    if nose is not None:
        nose_y = nose[1]
        chin_y = chin[1] if chin is not None else (nose_y + 1.1*hd)
        cy = phone_xy[1]
        if nose_y <= cy <= chin_y:  # nằm giữa mũi và cằm → phù hợp CALL
            mid = (nose_y + chin_y) * 0.5
            h_call = 1.0 - abs(cy - mid) / (0.5*(chin_y - nose_y) + 1e-6)
        if cy > nose_y + 0.3*hd:     # thấp xuống → ưu tiên VIEW
            h_view = min(1.0, (cy - (nose_y + 0.3*hd)) / (0.8*hd))

    # Cổ tay nằm giữa ear–phone → ưu tiên CALL
    def wrist_between(wrist, ear):
        if wrist is None or ear is None: return 0.0
        d = _segdist(wrist, ear, phone_xy)
        return float(np.exp(-d/(0.6*hd)))
    wrist_mid = max(wrist_between(lw, le), wrist_between(lw, re),
                    wrist_between(rw, le), wrist_between(rw, re))

    # Gần cổ tay → TEXT
    near_wrist = any(_dist(phone_xy, w) < k_wrist_head*hd for w in (lw, rw) if w is not None)

    # ===== 3) Điểm CALL và VIEW/TEXT =====
    # CALL
    call_score  = 0.0
    # (a) nếu rơi Cheek Region → boost mạnh
    if in_cheek:
        call_score += 2.6
    # (b) gần tai + gần “đoạn mũi→vai”
    call_score += w_ear*np.exp(-d_ear_min/(0.9*hd)) + w_ear*np.exp(-d_cheek_seg/(0.7*hd))
    # (c) lệch ngang tăng điểm CALL
    call_score += w_lateral*(1.0 - np.exp(-lateral/(1.1*hd)))
    # (d) tay giữa ear–phone
    call_score += w_wrist*wrist_mid
    # (e) cao độ hợp lý
    call_score += w_height*h_call

    # VIEW/TEXT
    vt_score = 0.0
    if pitch >= pitch_thr and nose is not None and phone_xy[1] > nose[1]:
        vt_score += 1.1*np.exp(-lateral/(1.0*hd))
        vt_score += w_height*(0.6*h_view + 0.4)
        if near_wrist:
            vt_score += 0.9

    # ===== 4) Quyết định =====
    if call_score > max(vt_score, 0.35):
        return u"Gọi điện"
    if vt_score > 0.6:
        return u"Nhắn tin" if near_wrist else u"Xem điện thoại"
    return None


def classify_intent_fallback(phone_box, W, H):
    if phone_box is None:
        return None
    x1,y1,x2,y2 = phone_box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w = max(1, x2-x1); h = max(1, y2-y1)

    head_top = 0.05*H
    head_bot = 0.58*H   # dải đầu/má rộng hơn
    near_edge_x = (cx < 0.24*W) or (cx > 0.76*W)
    tall_phone = (h / float(w)) >= 1.20

    # CALL thô: phone cao dọc, ở dải đầu/má, lệch về rìa
    if head_top <= cy <= head_bot and (near_edge_x or tall_phone):
        return u"Gọi điện"

    # VIEW thô: thấp dưới mũi/đầu
    if cy > 0.45*H:
        return u"Xem điện thoại"

    return None

# ================== UI ==================
title = "📷 Camera AI — Phát hiện **sử dụng** / **điện thoại** (CPU)"
st.title(title)
st.caption("Chế độ hiện tại: **SỬ DỤNG**" if mode.startswith("Khi sử dụng") else "Chế độ hiện tại: **ĐIỆN THOẠI**")

c1,c2 = st.columns(2)
if c1.button("▶️ Bắt đầu giám sát", use_container_width=True): st.session_state.running = True
if c2.button("⏹️ Dừng", use_container_width=True):            st.session_state.running = False
frame_holder = st.empty(); debug_holder = st.empty(); status_holder = st.empty()

# ================== Loop ==================
if st.session_state.running:
    if source_type!="Webcam" and not source_value:
        st.error("Vui lòng cung cấp URL hoặc tải video."); st.session_state.running=False
    else:
        cap = open_capture(source_value if source_type!="Webcam" else cam_index)
        if not cap or not cap.isOpened():
            st.error("Không mở được nguồn video."); st.session_state.running=False
        else:
            try:
                need_frames = 6
                while st.session_state.running and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        status_holder.warning("Hết video hoặc mất kết nối."); break
                    H, W = frame.shape[:2]

                    # FPS thực
                    now = time.time()
                    dt = max(1e-6, now - st.session_state.prev_ts)
                    st.session_state.prev_ts = now
                    fps_inst = 1.0/dt
                    st.session_state.fps_hist.append(fps_inst)
                    fps = float(np.mean(st.session_state.fps_hist)) if st.session_state.fps_hist else 25.0

                    # YOLO
                    r = model.predict(source=frame, conf=conf_thres, iou=iou_thres, imgsz=imgsz, verbose=False)[0]
                    names = r.names
                    ids   = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
                    label_counts = Counter([names[i] for i in ids]) if len(ids) else Counter()

                    # ================== CHẾ ĐỘ B: CHỈ CẦN THẤY ĐIỆN THOẠI ==================
                    if mode.startswith("Chỉ cần"):
                        vis = r.plot()  # BGR có bbox như script gốc
                        has_phone = label_counts.get("cell phone", 0) > 0

                        # Hiển thị
                        frame_holder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                           caption=f"Đếm theo lớp: {dict(label_counts)}",
                                           use_container_width=True)

                        # Lưu theo cooldown (đúng script)
                        now_t = time.time()
                        if has_phone and (now_t - st.session_state.last_saved >= save_cooldown):
                            save_path = save_frame_simple(vis, camera_location)
                            st.session_state.last_saved = now_t
                            status_holder.success(f"📸 Đã lưu vi phạm: {save_path}")

                        time.sleep(0.01)
                        continue  # sang khung hình kế

                    # ================== CHẾ ĐỘ A: KHI SỬ DỤNG ==================
                    vis = frame.copy()
                    phone_boxes = []
                    if r.boxes is not None:
                        for b, cls in zip(r.boxes.xyxy.cpu().numpy(), ids):
                            if names[cls] in ("cell phone","mobile phone","phone"):
                                phone_boxes.append(b.astype(int))

                    phone_center, chosen_box = None, None
                    if phone_boxes:
                        phone_boxes.sort(key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
                        chosen_box = phone_boxes[0]
                        x1,y1,x2,y2 = chosen_box
                        c_now = np.array([(x1+x2)/2, (y1+y2)/2], dtype=np.float32)
                        if smooth_center:
                            st.session_state.center_smooth.append(c_now)
                            phone_center = np.mean(np.stack(st.session_state.center_smooth,0), axis=0).astype(np.float32)
                        else:
                            phone_center = c_now
                        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,200,255),2)
                        cv2.circle(vis, tuple(phone_center.astype(int)), 4, (0,255,255), -1)

                    # Ý định
                    trigger_label, reason = None, "No phone"
                    if chosen_box is not None:
                        if POSE is not None:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            res = POSE.process(rgb)
                            if res and res.pose_landmarks:
                                geom = estimate_geometry(res.pose_landmarks.landmark, W, H, draw=vis if debug_overlay else None)
                                trigger_label = classify_intent_with_pose(
                                    phone_center, geom,
                                    k_wrist_head=k_wrist_head, k_ear=near_ear_k,
                                    k_lateral=k_lateral, pitch_thr=pitch_deg_thr
                                )
                                reason = "Pose OK" if trigger_label else "Pose but no intent"
                            else:
                                trigger_label = classify_intent_fallback(chosen_box, W, H)
                                reason = "No pose -> fallback"
                        else:
                            trigger_label = classify_intent_fallback(chosen_box, W, H)
                            reason = "No pose module"

                    if trigger_label:
                        # cv2.putText(vis, f"Ý định: {trigger_label}", (10, 34),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                        vis=draw_unicode_text(vis, f"Ý định: {trigger_label}", (10, 34))

                    # Cửa sổ thời gian
                    need_frames = max(3, int(intent_seconds * max(5.0, fps)))
                    st.session_state.window.append(1 if trigger_label else 0)
                    stable = sum(st.session_state.window) >= need_frames

                    # Show
                    frame_holder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                       caption=f"Đếm theo lớp: {dict(label_counts)} | FPS ≈ {fps:.1f}",
                                       use_container_width=True)
                    if show_debug:
                        debug_holder.info(
                            f"MODE=USAGE • Trigger: **{trigger_label or 'None'}** | "
                            f"window_sum: **{sum(st.session_state.window)}** / need_frames: **{need_frames}** | "
                            f"reason: {reason}"
                        )

                    # Lưu
                    if stable and trigger_label:
                        save_path = save_frame_intent(vis, camera_location, trigger_label)
                        status_holder.success(f"📸 Lưu (ý định {trigger_label}): {save_path}")
                        st.session_state.window.clear()

                    time.sleep(0.005)
            finally:
                cap.release(); status_holder.info("Đã dừng giám sát.")
else:
    st.info("Nhấn **Bắt đầu giám sát** để khởi động.")
