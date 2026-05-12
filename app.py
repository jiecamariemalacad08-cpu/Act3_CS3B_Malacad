import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import av
import cv2
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="SmartVision AI", layout="wide")

SAVE_DIR = "detection_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
CLASS_NAMES = list(model.names.values())

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #f1f5f9, #dbeafe);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 2px solid #1e293b;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: white !important;
}
div[data-baseweb="select"] > div {
    color: black !important;
    font-weight: 600;
    border-radius: 12px;
}
ul, li {
    color: black !important;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    border: none;
    padding: 12px;
    font-weight: bold;
    transition: 0.3s;
}
.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #dc2626, #991b1b);
}
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 10px;
}
.sub-title {
    text-align: center;
    color: #334155;
    font-size: 18px;
    margin-bottom: 30px;
}
.tip-box {
    background: rgba(255,255,255,0.7);
    padding: 15px;
    border-radius: 15px;
    color: #0f172a;
    font-weight: 600;
    border-left: 5px solid #ef4444;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">📹 Live Object Detection & Tracing</div>
<div class="sub-title">Real-Time Object Detection and Tracking using YOLOv8</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎛️ Control Panel")

    confidence = st.slider("🎯 Detection Confidence", 0.1, 1.0, 0.5, 0.05)

    target_object = st.selectbox("🚨 Alert Object", CLASS_NAMES)

    save_images = st.toggle("📸 Save Detection Images", value=True)

    show_boxes = st.toggle("🟦 Show Bounding Boxes", value=True)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        results = model.predict(img, conf=confidence, imgsz=480, verbose=False)

        detected_counts = {}

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, "Unknown")

                detected_counts[label] = detected_counts.get(label, 0) + 1

                if show_boxes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

        if target_object in detected_counts:
            cv2.putText(img,
                        f"ALERT: {target_object.upper()} DETECTED!",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        if save_images and len(detected_counts) > 0:
            filename = os.path.join(
                SAVE_DIR,
                f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(filename, img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="smartvision",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
