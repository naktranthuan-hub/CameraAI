# -*- coding: utf-8 -*-
"""
Dashboard Camera AI (multi-cam) + Header 2 logo
- Add/Remove camera; đặt tên
- Webcam (nội/USB), RTSP, HTTP MJPEG, HTTP Snapshot, Video file
- Mode A: CALL/VIEW/TEXT; Mode B: chỉ cần thấy điện thoại
- Threaded reader + auto-reconnect
- Header: 2 logo (lythuongkiet.jpg & nguyenanninh.jpg) + tiêu đề mới
"""

import os, csv, time, threading, queue, urllib.request, platform, uuid
from datetime import datetime
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ---------------- RERUN helper (compat) ----------------
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ================== (Optional) MediaPipe Pose ==================
POSE = None
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    POSE = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        enable_segmentation=False, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
except Exception:
    POSE = None  # nếu thiếu mediapipe, Mode A sẽ fallback heuristic

# ================== App / IO ==================

st.set_page_config(page_title="Hệ thống An ninh Giám sát", layout="wide")
APP_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
OUTPUT_DIR = Path("vipham"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "log.csv"

# ---------- Header với 2 logo + tiêu đề ----------
def render_header():
    left_logo  = APP_DIR / "lythuongkiet.jpg"
    right_logo = APP_DIR / "nguyenanninh.jpg"
    c1, c2, c3 = st.columns([1, 6, 1], gap="small")

    with c1:
        if left_logo.exists():
            st.image(str(left_logo), width='stretch')
        else:
            st.write("")

    with c2:
        st.markdown(
            """
            <div style="text-align:center; padding-top:6px;">
                <h1 style="margin-bottom:4px;">Hệ thống An ninh Giám sát sử dụng thiết bị di động trong trường học</h1>
                <div style="color:#5c6b7a; font-size:16px;">
                    Camera AI — Phát hiện CALL / VIEW / TEXT (YOLOv11 + Pose)
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    with c3:
        if right_logo.exists():
            st.image(str(right_logo), width='stretch')
        else:
            st.write("")

render_header()

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")  # nhẹ, chạy CPU ổn
model = load_model()
#st.success("YOLOv11 (yolo11n.pt) đã sẵn sàng.")

# ================== Session State ==================
def ss_init():
    ss = st.session_state
    ss.setdefault("cams", {})          # id -> config (metadata/UI)
    ss.setdefault("rt", {})            # id -> runtime (reader/cap, caches)
    ss.setdefault("running_all", False)
    ss.setdefault("grid_cols", 2)
    ss.setdefault("global_conf", 0.30)
    ss.setdefault("global_iou", 0.50)
    ss.setdefault("global_imgsz", 640)
ss_init()

# ================== CSV utils ==================
def ensure_csv_header_if_needed():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "camera_name", "image_path", "intent_or_blank"])
ensure_csv_header_if_needed()

def save_frame_simple(frame_bgr, camera_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = OUTPUT_DIR / f"vipham_{ts}.jpg"
    cv2.imwrite(str(img_path), frame_bgr)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), camera_name, str(img_path), ""])
    return img_path

def save_frame_intent(frame_bgr, camera_name, intent_label):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = OUTPUT_DIR / f"vipham_{ts}.jpg"
    cv2.imwrite(str(img_path), frame_bgr)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), camera_name, str(img_path), intent_label])
    return img_path

# ================== Threaded Reader for IP ==================
class FrameReader:
    """Đọc khung hình trong thread nền, giữ 1 khung mới nhất; auto-reconnect."""
    def __init__(self, capture_fn, name="reader", max_queue=1, fps_limit=None):
        self.capture_fn = capture_fn
        self.name = name
        self.max_queue = max_queue
        self.fps_limit = fps_limit
        self.q = queue.Queue(maxsize=max_queue)
        self.stop_event = threading.Event()
        self.thread = None
        self.last_err = None

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self.thread.start()

    def _loop(self):
        backoff = 0.5
        while not self.stop_event.is_set():
            cap = None
            try:
                cap = self.capture_fn()
                if not cap or not cap.isOpened():
                    raise RuntimeError("Không mở được stream")
                t_prev = 0.0
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok: raise RuntimeError("Mất khung hình / kết nối")
                    if self.fps_limit:
                        now = time.time()
                        period = 1.0 / float(self.fps_limit)
                        if now - t_prev < period: continue
                        t_prev = now
                    while not self.q.empty():
                        try: self.q.get_nowait()
                        except queue.Empty: break
                    self.q.put(frame, timeout=0.01)
                backoff = 0.5
            except Exception as e:
                self.last_err = str(e)
                time.sleep(backoff)
                backoff = min(backoff*2, 8.0)
            finally:
                if cap is not None:
                    try: cap.release()
                    except: pass

    def get(self, timeout=0.02):
        try: return self.q.get(timeout=timeout)
        except queue.Empty: return None

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=1.0)

# ============ Builders for different sources ============
def build_rtsp_capture(url, transport="tcp", max_delay_ms=500, drop_late=True):
    def _open():
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
        opts = []
        if drop_late:     opts += ["-fflags","nobuffer","-flags","low_delay","-rtbufsize","0"]
        if transport:     opts += ["-rtsp_transport", "udp"]
        if max_delay_ms:  opts += ["-max_delay", str(max_delay_ms*1000)]
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = " ".join(opts)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 0)
        return cap
    return _open

def build_mjpeg_capture(url):
    def _open():
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return _open

def build_snapshot_reader(url, interval_ms=250):
    def _open_snapshot():
        class _SnapCap:
            def __init__(self): self.opened = True
            def isOpened(self): return self.opened
            def read(self):
                try:
                    with urllib.request.urlopen(url, timeout=2.0) as resp:
                        data = resp.read()
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None: return False, None
                    time.sleep(interval_ms/1000.0)
                    return True, img
                except Exception:
                    time.sleep(0.5)
                    return False, None
            def release(self): self.opened = False
        return _SnapCap()
    return FrameReader(_open_snapshot, name="snapshot", max_queue=1, fps_limit=None)

# ================== Pose/Geometry helpers (Mode A) ==================
def _to_xy(lm, w, h): return np.array([lm.x*w, lm.y*h], dtype=np.float32) if lm is not None else None
def _dist(a, b): 
    if a is None or b is None: return 1e9
    return float(np.linalg.norm(a-b))
def _angle_deg(v1, v2):
    denom = (np.linalg.norm(v1)*np.linalg.norm(v2))+1e-6
    cosv = np.clip(np.dot(v1, v2)/denom, -1, 1)
    return float(np.degrees(np.arccos(cosv)))

def estimate_geometry(lms, w, h, draw=None):
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
        v = v/(np.linalg.norm(v)+1e-6)
        pitch = _angle_deg(v, np.array([0,-1],dtype=np.float32))

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

    if draw is not None:
        for p,c in [(nose,(255,255,255)),(le,(0,200,0)),(re,(0,200,0)),
                    (lw,(255,0,0)),(rw,(255,0,0)),(ls,(0,160,255)),(rs,(0,160,255))]:
            if p is not None: cv2.circle(draw, tuple(p.astype(int)), 4, c, -1)

    return dict(nose=nose, le=le, re=re, lw=lw, rw=rw, ls=ls, rs=rs,
                head_size=head_size, shoulder_width=shoulder_width, pitch=pitch)

def classify_intent_with_pose(phone_xy, geom,
                              k_wrist_head=1.4, pitch_thr=20,
                              k_ear=0.65, k_lateral=1.5):
    if phone_xy is None:
        return None
    w_ear = max(0.5, float(k_ear) * 3.0)
    nose = geom.get("nose"); le=geom.get("le"); re=geom.get("re")
    lw=geom.get("lw"); rw=geom.get("rw"); ls=geom.get("ls"); rs=geom.get("rs")
    pitch = float(geom.get("pitch") or 0.0)
    hd = geom.get("head_size")
    if hd is None or hd <= 0:
        sw = geom.get("shoulder_width") or 0
        hd = sw*0.35 if sw>0 else 0.08*720
    hd = max(hd, 1.0)

    def in_cheek_region(p):
        if nose is None: return False
        x, y = p[0], p[1]
        y1 = nose[1] - 0.50*hd
        y2 = nose[1] + 0.80*hd
        return (y1 <= y <= y2) and (abs(x - nose[0]) <= 1.9*hd)
    in_cheek = in_cheek_region(phone_xy)

    def _segdist(p,a,b):
        if a is None or b is None or p is None: return 1e9
        ap = p - a; ab = b - a
        t = float(np.clip(np.dot(ap,ab)/(np.dot(ab,ab)+1e-6), 0, 1))
        proj = a + t*ab
        return float(np.linalg.norm(p-proj))

    d_ear_min = min(_dist(phone_xy, le), _dist(phone_xy, re))
    d_cheek_seg = min(_segdist(phone_xy, nose, ls), _segdist(phone_xy, nose, rs)) if nose is not None else 1e9
    lateral = abs(phone_xy[0] - (nose[0] if nose is not None else phone_xy[0]))

    h_call = 0.0; h_view = 0.0
    if nose is not None:
        chin_y = (nose[1] + 1.1*hd)
        cy = phone_xy[1]
        if nose[1] <= cy <= chin_y:
            mid = (nose[1] + chin_y)/2
            h_call = 1.0 - abs(cy-mid)/(0.5*(chin_y-nose[1])+1e-6)
        if cy > nose[1] + 0.3*hd:
            h_view = min(1.0, (cy - (nose[1] + 0.3*hd)) / (0.8*hd))

    def wrist_between(wrist, ear):
        if wrist is None or ear is None: return 0.0
        d = _segdist(wrist, ear, phone_xy)
        return float(np.exp(-d/(0.6*hd)))
    wrist_mid = max(wrist_between(lw, le), wrist_between(lw, re),
                    wrist_between(rw, le), wrist_between(rw, re))
    near_wrist = any(_dist(phone_xy, w) < k_wrist_head*hd for w in (lw, rw) if w is not None)

    call_score = 0.0
    if in_cheek: call_score += 2.6
    call_score += w_ear*np.exp(-d_ear_min/(0.9*hd)) + w_ear*np.exp(-d_cheek_seg/(0.7*hd))
    call_score += 1.1*(1.0 - np.exp(-lateral/(max(1e-3, k_lateral)*hd)))
    call_score += 1.0*wrist_mid + 1.0*h_call

    vt_score = 0.0
    if pitch >= pitch_thr and nose is not None and phone_xy[1] > nose[1]:
        vt_score += 1.1*np.exp(-lateral/(1.0*hd))
        vt_score += 1.0*(0.6*h_view + 0.4)
        if near_wrist: vt_score += 0.9

    if call_score > max(vt_score, 0.35): return "CALL"
    if vt_score  > 0.6: return "TEXT" if near_wrist else "VIEW"
    return None

def classify_intent_fallback(phone_box, W, H):
    if phone_box is None:
        return None
    x1,y1,x2,y2 = phone_box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w = max(1, x2-x1); h = max(1, y2-y1)
    head_top = 0.05*H
    head_bot = 0.58*H
    near_edge_x = (cx < 0.24*W) or (cx > 0.76*W)
    tall_phone = (h / float(w)) >= 1.20
    if head_top <= cy <= head_bot and (near_edge_x or tall_phone):
        return "CALL"
    if cy > 0.45*H:
        return "VIEW"
    return None

# ================== Camera Config & Runtime ==================
CAM_TYPES = ["Webcam", "RTSP (H264/H265)", "HTTP MJPEG", "HTTP Snapshot (.jpg)", "Video file (.mp4, .avi, ...)"]
MODES = ["Sử dụng (CALL/VIEW/TEXT)", "Chỉ cần thấy điện thoại"]

def add_camera(name, cam_type, src, mode, params):
    print(f"[INFO] Thêm camera: {name} | {cam_type} | {src} | mode={mode} | params={params}")
    import uuid as _uuid
    cid = str(_uuid.uuid4())[:8]
    st.session_state.cams[cid] = {
        "name": name.strip() or f"Camera {cid}",
        "type": cam_type,
        "src": src,
        "mode": mode,             # 0: A, 1: B
        "params": params,
        "enabled": True,
        "smooth_center": True,
        "intent_seconds": 1.2,
        "pitch_deg_thr": 20,
        "near_ear_k": 0.65,
        "k_lateral": 1.5,
        "k_wrist_head": 1.4,
        "save_cooldown": 5,
        "debug_overlay": False,
        "show_debug": True
    }
    st.session_state.rt[cid] = {
        "reader": None, "cap": None,
        "center_smooth": deque(maxlen=7),
        "fps_hist": deque(maxlen=30),
        "prev_ts": time.time(),
        "window": deque(maxlen=120),
        "last_saved": 0.0,
        "running": False
    }

def remove_camera(cid):
    rt = st.session_state.rt.get(cid)
    if rt:
        if rt["reader"] is not None: 
            try: rt["reader"].stop()
            except: pass
        if rt["cap"] is not None:
            try: rt["cap"].release()
            except: pass
    st.session_state.rt.pop(cid, None)
    st.session_state.cams.pop(cid, None)

def start_camera(cid):
    cam = st.session_state.cams[cid]; rt = st.session_state.rt[cid]
    if rt["running"]: return
    t = cam["type"]; src = cam["src"]; p = cam["params"]
    reader, cap = None, None
    if t == "RTSP (H264/H265)":
        open_fn = build_rtsp_capture(src, transport=p.get("rtsp_transport","tcp"),
                                     max_delay_ms=p.get("rtsp_max_delay_ms",500),
                                     drop_late=p.get("drop_late_frames", True))
        reader = FrameReader(open_fn, name=f"rtsp_{cid}", max_queue=1)
        reader.start()
    elif t == "HTTP MJPEG":
        open_fn = build_mjpeg_capture(src)
        reader = FrameReader(open_fn, name=f"mjpeg_{cid}", max_queue=1)
        reader.start()
    elif t == "HTTP Snapshot (.jpg)":
        reader = build_snapshot_reader(src, interval_ms=p.get("snapshot_interval",250))
        reader.start()
    elif t == "Webcam":
        os_name = platform.system().lower()
        backend = 0
        if "darwin" in os_name or "mac" in os_name: backend = cv2.CAP_AVFOUNDATION
        elif "windows" in os_name: backend = cv2.CAP_DSHOW
        cap = cv2.VideoCapture(int(src), backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:  # Video file
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    rt["reader"], rt["cap"] = reader, cap
    rt["running"] = True

def stop_camera(cid):
    rt = st.session_state.rt.get(cid)
    if not rt: return
    if rt["reader"] is not None:
        try: rt["reader"].stop()
        except: pass
        rt["reader"] = None
    if rt["cap"] is not None:
        try: rt["cap"].release()
        except: pass
        rt["cap"] = None
    rt["running"] = False
    rt["window"].clear()
    rt["center_smooth"].clear()
    rt["fps_hist"].clear()

# ================== Sidebar: Global Controls & Add Camera ==================
with st.sidebar:
    st.header("⚙️ Cấu hình chung")
    st.session_state.grid_cols = st.slider("Số cột dashboard", 1, 4, st.session_state.grid_cols)
    st.session_state.global_conf = st.slider("YOLO Confidence", 0.10, 0.90, float(st.session_state.global_conf), 0.05)
    st.session_state.global_iou  = st.slider("YOLO IoU",        0.10, 0.90, float(st.session_state.global_iou), 0.05)
    st.session_state.global_imgsz= st.select_slider("YOLO Image size", [320,480,640,768,960,1280], value=int(st.session_state.global_imgsz))

    c1, c2 = st.columns(2)
    if c1.button("▶️ Start ALL", width='stretch'):
        for cid in list(st.session_state.cams.keys()):
            try: start_camera(cid)
            except: pass
        st.session_state.running_all = True
    if c2.button("⏹️ Stop ALL", width='stretch'):
        for cid in list(st.session_state.cams.keys()):
            try: stop_camera(cid)
            except: pass
        st.session_state.running_all = False

    st.divider()
    st.subheader("➕ Thêm camera")
    name = st.text_input("Tên camera", value="Lớp 9A4 - góc trái")
    cam_type = st.selectbox("Loại nguồn", CAM_TYPES, index=0, key="add_type")
    src = ""
    params = {}
    if cam_type == "Webcam":
        src = st.number_input("Chỉ số webcam (0,1,2...)", min_value=0, value=0, step=1, key="wb_idx")
        src = str(int(src))
    elif cam_type == "RTSP (H264/H265)":
        src = st.text_input("RTSP URL", value="", key="rtsp_url")
        params["rtsp_transport"] = st.selectbox("RTSP transport", ["tcp","udp"], index=0, key="rtsp_tr")
        params["rtsp_max_delay_ms"] = st.slider("Max delay (ms)", 0, 3000, 500, 50, key="rtsp_delay")
        params["drop_late_frames"] = st.checkbox("Drop late frames (low-latency)", True, key="rtsp_drop")
    elif cam_type == "HTTP MJPEG":
        src = st.text_input("MJPEG URL", value="http://...", key="mjpeg_url")
    elif cam_type == "HTTP Snapshot (.jpg)":
        src = st.text_input("Snapshot URL (.jpg)", value="http://...", key="snap_url")
        params["snapshot_interval"] = st.slider("Khoảng làm mới (ms)", 100, 2000, 250, 50, key="snap_itv")
    else:
        up = st.file_uploader("Tải video", type=["mp4","avi","mov","mkv"], key="vid_up")
        if up is not None:
            tmp = Path(f"temp_{up.name}"); tmp.write_bytes(up.read())
            src = str(tmp)

    mode = st.selectbox("Chế độ", MODES, index=0, key="add_mode")
    if st.button("Thêm vào dashboard", width='stretch', type="primary", disabled=(not src and cam_type!="Webcam")):
        add_camera(name, cam_type, src, 0 if mode.startswith("Sử dụng") else 1, params)
        st.success("Đã thêm camera.")

# ================== Main Grid: Cards per Camera ==================
cams = st.session_state.cams
if not cams:
    st.info("Chưa có camera nào. Thêm camera ở panel bên trái nhé.")
else:
    cols = st.columns(st.session_state.grid_cols)
    card_placeholders = {}  # cid -> dict of placeholders
    i = 0
    for cid, cam in list(cams.items()):
        with cols[i % st.session_state.grid_cols]:
            st.markdown(f"### 🎥 {cam['name']}")
            cc1, cc2, cc3 = st.columns(3)
            running = st.toggle("Running", value=st.session_state.rt[cid]["running"], key=f"run_{cid}")
            if running and not st.session_state.rt[cid]["running"]:
                start_camera(cid)
            if (not running) and st.session_state.rt[cid]["running"]:
                stop_camera(cid)
            if cc1.button("Start", key=f"start_{cid}", width='stretch'): start_camera(cid)
            if cc2.button("Stop",  key=f"stop_{cid}",  width='stretch'): stop_camera(cid)
            if cc3.button("❌ Remove", key=f"rm_{cid}", width='stretch'):
                remove_camera(cid)
                _rerun()

            with st.expander("Thiết lập nhanh", expanded=False):
                mode_idx = 0 if cam["mode"]==0 else 1
                new_mode = st.selectbox("Chế độ", MODES, index=mode_idx, key=f"mode_{cid}")
                cam["mode"] = 0 if new_mode.startswith("Sử dụng") else 1
                cam["smooth_center"] = st.checkbox("Làm mượt tâm phone", cam["smooth_center"], key=f"smooth_{cid}")
                if cam["mode"]==0:
                    cam["intent_seconds"] = st.slider("Duy trì Ý Định (s)", 0.5, 5.0, float(cam["intent_seconds"]), 0.1, key=f"intwin_{cid}")
                    cam["pitch_deg_thr"] = st.slider("Pitch ≥ (°)", 5, 45, int(cam["pitch_deg_thr"]), 1, key=f"pitch_{cid}")
                    cam["near_ear_k"] = st.slider("CALL: gần tai (×hd)", 0.10, 1.00, float(cam["near_ear_k"]), 0.05, key=f"ear_{cid}")
                    cam["k_lateral"] = st.slider("Lệch ngang (×hd)", 0.8, 3.0, float(cam["k_lateral"]), 0.1, key=f"lat_{cid}")
                    cam["k_wrist_head"] = st.slider("TEXT: gần cổ tay (×hd)", 0.6, 3.0, float(cam["k_wrist_head"]), 0.1, key=f"wr_{cid}")
                    cam["debug_overlay"] = st.checkbox("Overlay debug", cam["debug_overlay"], key=f"dbg_{cid}")
                else:
                    cam["save_cooldown"] = st.slider("Cooldown lưu (s)", 0, 30, int(cam["save_cooldown"]), 1, key=f"cool_{cid}")
                cam["show_debug"] = st.checkbox("Hiện bảng debug", cam["show_debug"], key=f"showdbg_{cid}")

            img_ph = st.empty()
            dbg_ph = st.empty()
            card_placeholders[cid] = {"img": img_ph, "dbg": dbg_ph}
        i += 1

    # ================== Main Update Loop ==================
    any_running = any(st.session_state.rt[cid]["running"] for cid in st.session_state.rt)
    loop_start = time.time()
    max_loop_seconds = 3600
    while any_running and (time.time() - loop_start < max_loop_seconds):
        any_running = False
        for cid, cam in list(st.session_state.cams.items()):
            rt = st.session_state.rt.get(cid)
            if not rt or not rt["running"]: 
                continue
            any_running = True
            reader, cap = rt["reader"], rt["cap"]

            # Lấy frame
            frame, ok = None, False
            if reader is not None:
                frame = reader.get(timeout=0.01)
                ok = frame is not None
            else:
                if cap is None or (not cap.isOpened()):
                    ok = False
                else:
                    ok, frame = cap.read()

            if not ok or frame is None:
                card_placeholders[cid]["dbg"].warning("⏳ Chờ khung hình...")
                continue

            H, W = frame.shape[:2]
            now = time.time()
            dt = max(1e-6, now - rt["prev_ts"]); rt["prev_ts"] = now
            fps_inst = 1.0/dt
            rt["fps_hist"].append(fps_inst)
            fps = float(np.mean(rt["fps_hist"])) if rt["fps_hist"] else 25.0

            r = model.predict(source=frame,
                              conf=float(st.session_state.global_conf),
                              iou=float(st.session_state.global_iou),
                              imgsz=int(st.session_state.global_imgsz),
                              verbose=False)[0]
            names = r.names
            ids   = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
            label_counts = Counter([names[i] for i in ids]) if len(ids) else Counter()

            if cam["mode"] == 1:
                # Mode B
                vis = r.plot()
                has_phone = label_counts.get("cell phone", 0) > 0
                card_placeholders[cid]["img"].image(
                    cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                    caption=f"{cam['name']} • {dict(label_counts)} | FPS≈{fps:.1f}",
                    width='stretch'
                )
                if has_phone and (now - rt["last_saved"] >= cam["save_cooldown"]):
                    save_path = save_frame_simple(vis, cam["name"])
                    rt["last_saved"] = now
                    card_placeholders[cid]["dbg"].success(f"📸 Lưu vi phạm: {save_path}")
                else:
                    if cam["show_debug"]:
                        card_placeholders[cid]["dbg"].info(
                            f"MODE=B • Cooldown: {max(0, cam['save_cooldown'] - (now-rt['last_saved'])):.1f}s"
                        )
                continue

            # Mode A
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
                if cam["smooth_center"]:
                    rt["center_smooth"].append(c_now)
                    phone_center = np.mean(np.stack(rt["center_smooth"],0), axis=0).astype(np.float32)
                else:
                    phone_center = c_now
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,200,255),2)
                cv2.circle(vis, tuple(phone_center.astype(int)), 4, (0,255,255), -1)

            trigger_label, reason = None, "No phone"
            if chosen_box is not None:
                if POSE is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = POSE.process(rgb)
                    if res and res.pose_landmarks:
                        geom = estimate_geometry(res.pose_landmarks.landmark, W, H,
                                                 draw=vis if cam["debug_overlay"] else None)
                        trigger_label = classify_intent_with_pose(
                            phone_center, geom,
                            k_wrist_head=cam["k_wrist_head"],
                            pitch_thr=cam["pitch_deg_thr"],
                            k_ear=cam["near_ear_k"],
                            k_lateral=cam["k_lateral"]
                        )
                        reason = "Pose OK" if trigger_label else "Pose but no intent"
                    else:
                        trigger_label = classify_intent_fallback(chosen_box, W, H)
                        reason = "No pose -> fallback"
                else:
                    trigger_label = classify_intent_fallback(chosen_box, W, H)
                    reason = "No pose module"

            if trigger_label:
                cv2.putText(vis, f"INTENT: {trigger_label}", (10, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

            need_frames = max(3, int(cam["intent_seconds"] * max(5.0, fps)))
            rt["window"].append(1 if trigger_label else 0)
            stable = sum(rt["window"]) >= need_frames

            card_placeholders[cid]["img"].image(
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                caption=f"{cam['name']} • {dict(label_counts)} | FPS≈{fps:.1f}",
                width='stretch'
            )
            if cam["show_debug"]:
                card_placeholders[cid]["dbg"].info(
                    f"MODE=A • Trigger: **{trigger_label or 'None'}** | "
                    f"window_sum: **{sum(rt['window'])}** / need: **{need_frames}** | {reason}"
                )

            if stable and trigger_label:
                save_path = save_frame_intent(vis, cam["name"], trigger_label)
                card_placeholders[cid]["dbg"].success(f"📸 Lưu (ý định {trigger_label}): {save_path}")
                rt["window"].clear()

        time.sleep(0.01)

    if not any_running:
        st.caption("⏹️ Tất cả camera đang ở trạng thái dừng hoặc chờ khung hình.")
