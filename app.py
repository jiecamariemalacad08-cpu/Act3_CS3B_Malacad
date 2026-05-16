import subprocess
import sys

try:
    import tornado
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tornado"])

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import av
import cv2
import glob
import io
import zipfile
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(
    page_title="📹 Live Object Detection & Tracing",
    layout="wide"
)

SAVE_DIR = "detection_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

if "gallery_mode" not in st.session_state:
    st.session_state.gallery_mode = False

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
CLASS_NAMES = list(model.names.values())

st.markdown("""
<style>
html, body, [class*="css"] {font-family: 'Poppins', sans-serif;}
.stApp {background: linear-gradient(135deg, #0f172a, #111827, #1e293b); color: #f8fafc;}
.title {text-align: center; font-size: clamp(35px, 5vw, 60px); font-weight: 900; color: #38bdf8; text-shadow: 0px 0px 20px rgba(56,189,248,0.7); margin-bottom: 5px;}
.subtitle {text-align: center; color: #cbd5e1; font-size: 18px; margin-bottom: 30px;}
.panel {background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56,189,248,0.3); padding: 25px; border-radius: 20px; box-shadow: 0 0 25px rgba(56,189,248,0.15); backdrop-filter: blur(12px);}
.stat-box {background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid rgba(56,189,248,0.2); padding: 20px; border-radius: 18px; text-align: center; margin-bottom: 20px; transition: 0.3s ease;}
.stat-box:hover {transform: translateY(-4px); box-shadow: 0 0 20px rgba(56,189,248,0.35);}
.stat-title {color: #94a3b8; font-size: 14px; margin-bottom: 6px;}
.stat-value {font-size: 34px; font-weight: 800; color: #38bdf8;}
.alert-box {background: linear-gradient(135deg, #ef4444, #b91c1c); color: white; padding: 18px; border-radius: 18px; text-align: center; font-size: 20px; font-weight: bold; box-shadow: 0 0 20px rgba(239,68,68,0.4);}
.stButton > button {background: linear-gradient(135deg, #0ea5e9, #2563eb); color: white; border: none; border-radius: 14px; padding: 12px 24px; font-weight: 700; transition: all 0.3s ease; box-shadow: 0 0 15px rgba(14,165,233,0.35);}
.stButton > button:hover {transform: scale(1.05); background: linear-gradient(135deg, #0284c7, #1d4ed8);}
.stDownloadButton > button {background: linear-gradient(135deg, #22c55e, #15803d) !important;}
.stDownloadButton > button:hover {background: linear-gradient(135deg, #16a34a, #166534) !important;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #111827, #0f172a); border-right: 1px solid rgba(56,189,248,0.2);}
.stSlider label, .stSelectbox label, .stToggle label {color: #e2e8f0 !important; font-weight: 600;}
img {border-radius: 18px; border: 2px solid rgba(56,189,248,0.25); box-shadow: 0 0 25px rgba(56,189,248,0.15);}
.block-container {padding-top: 2rem;}
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title">📹 Live Object Detection & Tracing</div>
<div class="subtitle">Real-Time AI Detection using YOLOv8 + Streamlit</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    target_object = st.selectbox("🚨 Alert Target", CLASS_NAMES)
    save_images = st.toggle("📸 Save Detection", value=True)
    show_boxes = st.toggle("🟦 Show Bounding Boxes", value=True)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_objects = set()
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = model.predict(img, conf=confidence, imgsz=480, verbose=False)
        detected_counts = {}
        current_objects = set()
        alert_detected = False
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, "unknown")
                detected_counts[label] = detected_counts.get(label, 0) + 1
                current_objects.add(label)
                if label.lower() == "person":
                    alert_detected = True
                if show_boxes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (56, 189, 248), 2)
                    cv2.putText(img, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if save_images and current_objects != self.prev_objects:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, img)
            self.prev_objects = current_objects
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown('<div class="stat-title">👥 People Detected</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{detected_counts.get("person", 0)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if alert_detected:
            st.markdown('<div class="alert-box">🚨 Person Detected!</div>', unsafe_allow_html=True)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown('<div class="panel">', unsafe_allow_html=True)
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("## 📂 Detection Gallery")
image_files = glob.glob(os.path.join(SAVE_DIR, "*.jpg"))
if image_files:
    cols = st.columns(3)
    for index, img_path in enumerate(reversed(image_files[-9:])):
        with cols[index % 3]:
            st.image(img_path, use_container_width=True)
            st.caption(os.path.basename(img_path))
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file in image_files:
            zip_file.write(file, os.path.basename(file))
    st.download_button(
        label="⬇ Download Detection Logs",
        data=zip_buffer.getvalue(),
        file_name="detection_logs.zip",
        mime="application/zip"
    )
else:
    st.info("No saved detections yet.")
