# -*- coding: utf-8 -*-
"""
Camera AI — Ghi nhận Ý ĐỊNH dùng điện thoại (YOLOv11 + MediaPipe Pose + Streamlit, CPU)
- Không lưu nếu chỉ "cầm".
- Lưu khi CALL (áp tai) / VIEW (nhìn màn hình trước mặt) / TEXT (nhắn – máy gần cổ tay).
- Ảnh -> vipham/vipham_YYYYMMDD_hhmmss_ms.jpg
- CSV -> vipham/log.csv  (timestamp, camera_location, image_path, intent)
"""
import csv, time
from datetime import datetime
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ===== MediaPipe Pose (CPU) =====
POSE = None
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    POSE = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        enable_segmentation=False, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
except Exception:
    POSE = None  # Fallback sẽ hoạt động nếu không có mediapipe

# ===== IO / App =====
st.set_page_config(page_title="Camera AI - Ý định dùng điện thoại", layout="wide")

pathVisual="d:\\ThuNghiem\\PythonApplication1"

OUTPUT_DIR = Path(f"{pathVisual}\\vipham"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "log.csv"
if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "camera_location", "image_path", "intent"])

@st.cache_resource
def load_model():
    return YOLO(f"{pathVisual}\\yolo11n.pt")  # đổi sang yolo11s.pt nếu CPU khỏe
model = load_model()
st.success("Đã tải YOLOv11 (yolo11n.pt).")

# ===== Sidebar =====
st.sidebar.header("Nguồn video & YOLO")
camera_location = st.sidebar.text_input("Vị trí camera", value="Lớp 9A4 - góc trái")

source_type = st.sidebar.selectbox("Nguồn video", ["Webcam","RTSP/HTTP URL","Video file (.mp4, .avi, ...)", "Stream"])
source_value = None
cam_index = 0
if source_type == "Webcam":
    cam_index = st.sidebar.number_input("Chỉ số webcam", min_value=0, value=0, step=1)
    source_value = cam_index
elif source_type == "RTSP/HTTP URL":
    source_value = st.sidebar.text_input("Nhập URL stream", value="")
elif source_type == "Stream":
    source_value = st.sidebar.text_input("Nhập URL stream (MJPEG/HTTP)", value="")
else:
    up = st.sidebar.file_uploader("Tải video", type=["mp4","avi","mov","mkv"])
    if up is not None:
        tmp = Path(f"temp_{up.name}"); tmp.write_bytes(up.read())
        source_value = str(tmp)

conf_thres = st.sidebar.slider("YOLO Confidence", 0.10, 0.90, 0.30, 0.05)
iou_thres  = st.sidebar.slider("YOLO IoU",        0.10, 0.90, 0.50, 0.05)
imgsz      = st.sidebar.select_slider("Image size", options=[320,480,640,768,960,1280], value=640)

st.sidebar.divider()
st.sidebar.subheader("Ngưỡng Ý ĐỊNH (theo kích thước đầu)")
pitch_deg_thr   = st.sidebar.slider("Đầu cúi xuống ≥ (°)", 5, 45, 20, 1)
near_ear_k      = st.sidebar.slider("CALL: gần tai (× head_size)",   0.10, 1.00, 0.65, 0.05)
k_lateral       = st.sidebar.slider("Bộ lọc lệch ngang (× head_size)", 0.8, 3.0, 1.5, 0.1)
k_wrist_head    = st.sidebar.slider("TEXT: gần cổ tay (× head_size)", 0.6, 3.0, 1.4, 0.1)
intent_seconds  = st.sidebar.slider("Duy trì ý định (giây)", 0.5, 5.0, 1.2, 0.1)
cooldown_sec    = st.sidebar.slider("Khoảng cách mỗi lần lưu (giây)", 0, 30, 3, 1)

st.sidebar.divider()
smooth_center   = st.sidebar.checkbox("Làm mượt vị trí điện thoại", True)
debug_overlay   = st.sidebar.checkbox("Overlay debug (tai/má…)", False)
show_debug_panel= st.sidebar.checkbox("Hiển thị bảng debug", True)

# ===== Utils =====
def open_capture(src):
    cap = cv2.VideoCapture(src); cap.set(cv2.CAP_PROP_BUFFERSIZE, 2); return cap

def save_violation_frame(frame_bgr, location_text, intent_label):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = OUTPUT_DIR / f"vipham_{ts}.jpg"
    cv2.imwrite(str(img_path), frame_bgr)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), location_text, str(img_path), intent_label])
    return img_path

def ensure_state():
    ss = st.session_state
    ss.setdefault("running", False)
    ss.setdefault("last_saved", 0.0)
    ss.setdefault("intent_window", deque(maxlen=120))  # ~4s @30FPS
    ss.setdefault("center_smooth", deque(maxlen=7))
    ss.setdefault("fps_hist", deque(maxlen=30))
    ss.setdefault("prev_ts", time.time())
ensure_state()

# ---- Geometry helpers ----
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
    """Lấy nose, ear L/R (proxy), wrist L/R, shoulder L/R; head_size & pitch."""
    from mediapipe.python.solutions.pose import PoseLandmark as P
    def g(idx):
        lm = lms[idx]
        return _to_xy(lm, w, h) if getattr(lm, 'visibility', 1.0) > 0.35 else None

    nose = g(P.NOSE); le = g(P.LEFT_EAR); re = g(P.RIGHT_EAR)
    lw   = g(P.LEFT_WRIST); rw = g(P.RIGHT_WRIST)
    ls   = g(P.LEFT_SHOULDER); rs = g(P.RIGHT_SHOULDER)

    shoulder_width = _dist(ls, rs) if (ls is not None and rs is not None) else None
    head_size = _dist(le, re) if (le is not None and re is not None) else (shoulder_width*0.35 if shoulder_width else None)

    pitch = None
    if nose is not None and ls is not None and rs is not None:
        mid = (ls + rs)/2
        v = nose - mid
        pitch = _angle_deg(v/np.linalg.norm(v+1e-6), np.array([0,-1],dtype=np.float32))

    # Tai proxy (góc nghiêng/tóc che)
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

    if debug_overlay and draw is not None:
        for p,c in [(nose,(255,255,255)),(le,(0,200,0)),(re,(0,200,0)),
                    (lw,(255,0,0)),(rw,(255,0,0)),(ls,(0,160,255)),(rs,(0,160,255))]:
            if p is not None: cv2.circle(draw, tuple(p.astype(int)), 4, c, -1)

    return dict(nose=nose, le=le, re=re, lw=lw, rw=rw, ls=ls, rs=rs,
                head_size=head_size, shoulder_width=shoulder_width, pitch=pitch)

# ===== Phân loại Ý ĐỊNH (KHÔNG dùng shoulder_width) =====
def classify_intent_with_pose(phone_xy, geom,
                              k_wrist_head=1.4,  # TEXT: gần cổ tay (< 1.4 * head_size)
                              k_ear=0.65,        # CALL: gần tai (< 0.65 * head_size)
                              k_lateral=1.5,     # lọc lệch ngang: |dx| <= 1.5 * head_size
                              pitch_thr=20):
    if phone_xy is None: return None
    nose = geom.get("nose"); le=geom.get("le"); re=geom.get("re")
    lw=geom.get("lw"); rw=geom.get("rw"); ls=geom.get("ls"); rs=geom.get("rs")
    pitch = geom.get("pitch") or 0.0

    # Ước lượng head_size nếu thiếu
    hd = geom.get("head_size")
    if hd is None or hd <= 0:
        # backup: 8% chiều cao ảnh hoặc 0.35*shoulder_width nếu có
        sw = geom.get("shoulder_width") or 0
        hd = sw*0.35 if sw>0 else 0.08 * 720  # 720 chỉ là tỉ lệ tham chiếu; hd chỉ dùng tương đối

    # 1) CALL: gần tai (kể cả proxy)
    if le is not None and _dist(phone_xy, le) < k_ear * hd: return "CALL"
    if re is not None and _dist(phone_xy, re) < k_ear * hd: return "CALL"

    # CALL fallback: vùng má (dọc theo nose->shoulder, cao ngang mũi)
    if nose is not None:
        dL = _dist_point_to_segment(phone_xy, nose, ls)
        dR = _dist_point_to_segment(phone_xy, nose, rs)
        if (dL < 0.35*hd and abs(phone_xy[1]-nose[1]) < 0.5*hd) or \
           (dR < 0.35*hd and abs(phone_xy[1]-nose[1]) < 0.5*hd):
            return "CALL"

    # 2) Bộ lọc lệch ngang (trừ CALL)
    if nose is not None:
        lateral = abs(phone_xy[0] - nose[0])
        if lateral > k_lateral * hd:
            return None  # cầm lệch ngang xa -> không phải VIEW/TEXT

    # 3) TEXT / VIEW (đầu cúi + máy ở phía trước đầu)
    if pitch >= pitch_thr and nose is not None and phone_xy[1] > nose[1]:
        near_wrist = False
        if lw is not None and _dist(phone_xy, lw) < k_wrist_head * hd: near_wrist = True
        if rw is not None and _dist(phone_xy, rw) < k_wrist_head * hd: near_wrist = True

        lateral = abs(phone_xy[0] - nose[0])
        if lateral <= 1.2 * hd:  # chặt hơn để thật sự "trước mặt"
            return "TEXT" if near_wrist else "VIEW"
    return None

# ===== Fallback khi không có Pose =====
def classify_intent_fallback(phone_box, W, H):
    if phone_box is None: return None
    x1,y1,x2,y2 = phone_box; cx,cy = (x1+x2)/2, (y1+y2)/2
    # CALL thô: ở nửa trên & lệch mạnh về 2 biên (áp tai)
    if cy < H*0.35 and (cx < W*0.25 or cx > W*0.75): return "CALL"
    # VIEW thô: bên dưới phần đầu
    if cy >= H*0.35: return "VIEW"
    return None

# ===== UI =====
st.title("📷 Camera AI — Ghi nhận **Ý ĐỊNH** dùng điện thoại (CPU)")
st.write("Không lưu nếu **chỉ cầm**; chỉ lưu khi **CALL / VIEW / TEXT** giữ ≥ ngưỡng thời gian.")
c1,c2 = st.columns(2)
if c1.button("▶️ Bắt đầu giám sát", use_container_width=True): st.session_state.running = True
if c2.button("⏹️ Dừng", use_container_width=True):            st.session_state.running = False
frame_holder = st.empty(); debug_holder = st.empty(); status_holder = st.empty()

# ===== Loop =====
if st.session_state.running:
    if source_type not in ["Webcam", "RTSP/HTTP URL", "Video file (.mp4, .avi, ...)", "Stream"] or not source_value:
        st.error("Vui lòng cung cấp URL hoặc tải video."); st.session_state.running=False
    else:
        cap = open_capture(source_value if source_type!="Webcam" else cam_index)
        if not cap or not cap.isOpened():
            st.error("Không mở được nguồn video."); st.session_state.running=False
        else:
            need_frames = 6  # sẽ cập nhật từ FPS thực
            try:
                while st.session_state.running and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok: status_holder.warning("Hết video hoặc mất kết nối."); break
                    H, W = frame.shape[:2]

                    # FPS thực
                    now = time.time()
                    dt = max(1e-6, now - st.session_state.prev_ts)
                    st.session_state.prev_ts = now
                    fps_inst = 1.0/dt
                    st.session_state.fps_hist.append(fps_inst)
                    fps = float(np.mean(st.session_state.fps_hist)) if st.session_state.fps_hist else 25.0
                    need_frames = max(3, int(intent_seconds * max(5.0, fps)))

                    # YOLO
                    r = model.predict(source=frame, conf=conf_thres, iou=iou_thres,
                                      imgsz=imgsz, verbose=False)[0]
                    names = r.names
                    ids   = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
                    label_counts = Counter([names[i] for i in ids]) if len(ids) else Counter()

                    phone_boxes = []
                    if r.boxes is not None:
                        for b, cls in zip(r.boxes.xyxy.cpu().numpy(), ids):
                            if names[cls] in ("cell phone","mobile phone","phone"):
                                phone_boxes.append(b.astype(int))

                    vis = frame.copy()
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
                    intent_label, reason = None, "No phone"
                    if chosen_box is not None:
                        if POSE is not None:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            res = POSE.process(rgb)
                            if res and res.pose_landmarks:
                                geom = estimate_geometry(res.pose_landmarks.landmark, W, H, draw=vis if debug_overlay else None)
                                intent_label = classify_intent_with_pose(
                                    phone_center, geom,
                                    k_wrist_head=k_wrist_head, k_ear=near_ear_k,
                                    k_lateral=k_lateral, pitch_thr=pitch_deg_thr
                                )
                                reason = "Pose OK" if intent_label else "Pose but no intent"
                            else:
                                intent_label = classify_intent_fallback(chosen_box, W, H)
                                reason = "No pose -> fallback"
                        else:
                            intent_label = classify_intent_fallback(chosen_box, W, H)
                            reason = "No pose module"

                    if intent_label:
                        cv2.putText(vis, f"INTENT: {intent_label}", (10, 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

                    # Cửa sổ thời gian
                    win = st.session_state.intent_window
                    win.append(1 if intent_label else 0)
                    stable = sum(win) >= need_frames

                    # Show
                    frame_holder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                       caption=f"Đếm theo lớp: {dict(label_counts)} | FPS ≈ {fps:.1f}",
                                       use_container_width=True)
                    if show_debug_panel:
                        debug_holder.info(
                            f"Intent: **{intent_label or 'None'}** | "
                            f"window_sum: **{sum(win)}** / need_frames: **{need_frames}** | "
                            f"reason: {reason}"
                        )

                    # Lưu
                    tnow = time.time()
                    if stable and intent_label and (tnow - st.session_state.last_saved >= cooldown_sec):
                        saved = save_violation_frame(vis, camera_location, intent_label)
                        st.session_state.last_saved = tnow
                        status_holder.success(f"📸 Lưu (ý định {intent_label}): {saved}")
                        win.clear()

                    time.sleep(0.005)
            finally:
                cap.release(); status_holder.info("Đã dừng giám sát.")
else:
    st.info("Nhấn **Bắt đầu giám sát** để khởi động.")
