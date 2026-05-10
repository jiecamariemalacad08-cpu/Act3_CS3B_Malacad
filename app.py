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
    page_title="✨ SmartVision AI Monitor",
    layout="wide"
)

SAVE_DIR = "smartvision_captures"
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
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #ecfeff 0%, #dbeafe 50%, #ede9fe 100%);
    color: #1e293b;
}

.title {
    text-align: center;
    font-size: clamp(34px, 4vw, 54px);
    font-weight: 800;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #475569;
    margin-bottom: 25px;
    font-size: 18px;
    font-weight: 500;
}

.panel {
    background: rgba(255,255,255,0.95);
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}

.alert-box {
    background: linear-gradient(135deg, #fb7185, #e11d48);
    color: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 20px;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: 600;
    padding: 10px 18px;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title">Live object Detection & Tracing</div>
<div class="subtitle">
Point your camera at objects to identify them in real-time
</div>
""", unsafe_allow_html=True)

st.caption("Developed by Jieca Marie J. Malacad • BSCS 3B")

with st.sidebar:

    st.markdown("## 🎛️ Control Center")

    confidence = st.slider(
        "🎯 Detection Accuracy",
        0.1,
        1.0,
        0.5,
        0.05
    )

    image_size = st.selectbox(
        "🖥️ Camera Quality",
        [320, 480, 640],
        index=1
    )

    target_object = st.selectbox(
        "🔔 Watch Object",
        CLASS_NAMES
    )

    save_images = st.toggle(
        "📸 Auto Save Frames",
        value=True
    )

    show_boxes = st.toggle(
        "🧩 Show Detection Boxes",
        value=True
    )

class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.prev_objects = set()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        results = model.predict(
            img,
            conf=confidence,
            imgsz=image_size,
            verbose=False
        )

        detected_counts = {}

        if results and results[0].boxes is not None:

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls_id = int(box.cls[0])

                label = model.names.get(cls_id, "unknown")

                detected_counts[label] = detected_counts.get(label, 0) + 1

                if show_boxes:

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        y = 30

        for obj, count in detected_counts.items():

            cv2.putText(
                img,
                f"{obj}: {count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            y += 30

        if target_object in detected_counts:

            cv2.putText(
                img,
                f"ALERT: {target_object.upper()} DETECTED!",
                (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        if save_images and len(detected_counts) > 0:

            filename = os.path.join(
                SAVE_DIR,
                f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )

            cv2.imwrite(filename, img)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

webrtc_streamer(
    key="smartvision",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

st.markdown("---")

st.subheader("🖼️ Saved Detection Frames")

saved_images = glob.glob(f"{SAVE_DIR}/*.jpg")

if saved_images:

    cols = st.columns(3)

    for i, image_path in enumerate(saved_images[-6:]):

        cols[i % 3].image(
            image_path,
            use_column_width=True
        )

else:
    st.info("No saved detection frames yet.")
