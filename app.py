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
    page_title="📹Live Object Detection & Tracing",
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
html, body, [class*="css"] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
    color: #1e293b;
}

.title {
    text-align: center;
    font-size: clamp(32px, 4vw, 52px);
    font-weight: 800;
    background: linear-gradient(135deg, #475569, #334155);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 25px;
    font-size: 18px;
    font-weight: 500;
}

.panel {
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(12px);
    border: 1px solid #e2e8f0;
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
}

.stat-box {
    background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
    padding: 20px;
    border-radius: 16px;
    border-left: 4px solid #475569;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}

.stat-title {
    color: #64748b;
    font-size: 15px;
    font-weight: 500;
    margin-bottom: 8px;
}

.stat-value {
    font-size: 32px;
    font-weight: 800;
    color: #1e293b;
    margin: 0;
}

.alert-box {
    background: linear-gradient(135deg, #f87171, #ef4444);
    color: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
    margin-bottom: 20px;
}

img {
    border-radius: 14px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.block-container {
    padding-top: 2rem;
}
.stButton > button {
    background: linear-gradient(135deg, #64748b, #475569);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 12px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(103, 116, 139, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #475569, #334155);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(103, 116, 139, 0.4);
}

.element-container .stDownloadButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
}

.element-container .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #059669, #047857) !important;
    transform: translateY(-2px);
}

button[kind="secondary"] {
    background: linear-gradient(135deg, #f87171, #ef4444) !important;
}

button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
}

.stMarkdown {
    color: #475569 !important;
    font-weight: 500 !important;
}

.stAlert {
    background: rgba(248, 250, 252, 0.9);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    color: #1e293b;
}

.css-1d391kg {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title">📹Live Object Detection & Tracing</div>

""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    image_size = st.selectbox("📏 Resolution", [320, 480, 640], index=1)
    target_object = st.selectbox("🚨 Alert Target", CLASS_NAMES)
    save_images = st.toggle(" Save Detection", value=True)
    show_boxes = st.toggle(" Show Bounding Boxes", value=True)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_objects = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        results = model.predict(img, conf=confidence, imgsz=image_size, verbose=False)
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

                if label.lower() == target_object.lower():
                    color = (0, 0, 255)
                    alert_detected = True
                else:
                    color = (0, 255, 0)

                if show_boxes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if alert_detected:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), -1)
            alpha = 0.25
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            overlay_height = 90
            cv2.rectangle(img, (0, img.shape[0] - overlay_height), (img.shape[1], img.shape[0]), (0, 0, 255), -1)
            cv2.putText(img, "ALERT DETECTED!", (40, img.shape[0] - 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

        new_objects = current_objects - self.prev_objects
        if save_images and new_objects:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for obj in new_objects:
                filename = f"{SAVE_DIR}/{obj}_{timestamp}.jpg"
                cv2.imwrite(filename, img)
        self.prev_objects = current_objects

        y = 35
        for obj, cnt in detected_counts.items():
            cv2.putText(img, f"{obj}: {cnt}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 30

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def get_image_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def create_zip():
    images = glob.glob(f"{SAVE_DIR}/*.jpg")
    if not images:
        return None
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in images:
            zf.write(img_path, os.path.basename(img_path))
    memory_file.seek(0)
    return memory_file.getvalue()

if st.session_state.gallery_mode:
    st.markdown("## 🖼️ Detection Gallery")

    images = sorted(glob.glob(f"{SAVE_DIR}/*.jpg"), reverse=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.gallery_mode = False
            st.rerun()

    with col2:
        zip_data = create_zip()
        if zip_data:
            st.download_button("⬇️ Download ZIP", zip_data, "detections.zip", "application/zip", use_container_width=True)

    if images:
        cols = st.columns(3)
        for idx, img_path in enumerate(images):
            with cols[idx % 3]:
                st.image(img_path, use_container_width=True)
                filename = os.path.basename(img_path)
                st.caption(filename)
                st.download_button("⬇️ Download", get_image_bytes(img_path), filename, "image/jpeg", key=f"dl_{img_path}", use_container_width=True)
                if st.button("🗑️ Delete", key=f"del_{img_path}", use_container_width=True):
                    os.remove(img_path)
                    st.rerun()
    else:
        st.info("No detections saved.")

else:
    left, right = st.columns([2.2, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        images = sorted(glob.glob(f"{SAVE_DIR}/*.jpg"), reverse=True)

        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-title">📸 Saved Detections</div>
            <div class="stat-value">{len(images)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="alert-box">
            🚨 ALERT TARGET<br>
            {target_object.upper()}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🖼️ Recent Captures")

        if images:
            for img_path in images[:4]:
                st.image(img_path, use_container_width=True)
                filename = os.path.basename(img_path)
                st.caption(filename)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("⬇️", get_image_bytes(img_path), filename, "image/jpeg", key=f"recent_dl_{img_path}", use_container_width=True)
                with col2:
                    if st.button("🗑️", key=f"recent_del_{img_path}", use_container_width=True):
                        os.remove(img_path)
                        st.rerun()

            if len(images) > 4:
                if st.button("🔍 View Full Gallery", use_container_width=True):
                    st.session_state.gallery_mode = True
                    st.rerun()
        else:
            st.info("No saved detections yet.")

st.markdown("---")
st.caption("👨‍💻 Developed by Joan M. Aduvas | YOLOv8 + Streamlit + WebRTC")
