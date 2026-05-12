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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="SmartVision AI",
    layout="wide"
)

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
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #f1f5f9 0%,
        #dbeafe 50%,
        #e0f2fe 100%
    );
    color: #0f172a;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 5px;
    animation: fadeIn 1s ease;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #475569;
    font-size: 18px;
    margin-bottom: 30px;
    animation: fadeIn 1.5s ease;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #1e293b 100%
    );
    border-right: 3px solid #38bdf8;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        135deg,
        #38bdf8,
        #0ea5e9
    );
    color: white;
    font-weight: 700;
    padding: 12px;
    transition: 0.3s ease;
    box-shadow: 0px 5px 15px rgba(14,165,233,0.4);
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(
        135deg,
        #0ea5e9,
        #0284c7
    );
}

/* Toggle */
[data-baseweb="switch"] {
    background-color: #38bdf8 !important;
}

/* Webcam */
video {
    border-radius: 20px !important;
    border: 4px solid #38bdf8;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
}

/* Saved Images */
img {
    border-radius: 16px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
}

/* Animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
✨ SmartVision AI
</div>

<div class="sub-title">
Live object Detection & Tracing
</div>
""", unsafe_allow_html=True)

with st.sidebar:

    st.markdown("## 🎛️ Control Panel")

    confidence = st.slider(
        "🎯 Detection Confidence",
        0.1,
        1.0,
        0.5,
        0.05
    )

    target_object = st.selectbox(
        "🚨 Alert Object",
        CLASS_NAMES
    )

    save_images = st.toggle(
        "📸 Save Detection Images",
        value=True
    )

    show_boxes = st.toggle(
        "🟦 Show Bounding Boxes",
        value=True
    )

    st.markdown("---")

    st.info(
        "📌 Tip: Use proper lighting for better detection accuracy."
    )

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        results = model.predict(
            img,
            conf=confidence,
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
                        0.7,
                        (0, 255, 0),
                        2
                    )

        y_position = 30

        for obj, count in detected_counts.items():

            cv2.putText(
                img,
                f"{obj}: {count}",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            y_position += 30

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

        if save_images and detected_counts:

            filename = os.path.join(
                SAVE_DIR,
                f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )

            cv2.imwrite(filename, img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="smartvision",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

st.markdown("## 🖼️ Saved Detection Frames")

saved_images = glob.glob(f"{SAVE_DIR}/*.jpg")

if saved_images:

    cols = st.columns(3)

    for i, image in enumerate(saved_images[-6:]):

        cols[i % 3].image(
            image,
            use_column_width=True
        )

else:
    st.info("No saved detection images yet.")
