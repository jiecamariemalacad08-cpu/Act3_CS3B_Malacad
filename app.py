import subprocess
import sys

try:
    import tornado
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tornado"])

try:
    from twilio.rest import Client
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "twilio"])
    from twilio.rest import Client

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


TWILIO_ACCOUNT_SID = "ACcd3c04d2fc8d40b43f6921f4d08b9403"
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_PHONE_NUMBER = "+1234567890"
ALERT_PHONE_NUMBER = "+639123456789"

twilio_client = Client(
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN
)

if "gallery_mode" not in st.session_state:
    st.session_state.gallery_mode = False

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
CLASS_NAMES = list(model.names.values())

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

.stApp {
    background:
        linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.82)),
        url('https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?q=80&w=1920&auto=format&fit=crop');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #f8fafc;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
    background-size: 35px 35px;
    pointer-events: none;
    z-index: 0;
}

.title {
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(42px, 6vw, 75px);
    font-weight: 800;
    color: #00f5ff;
    text-transform: uppercase;
    letter-spacing: 4px;
    margin-top: 10px;
    margin-bottom: 10px;
    text-shadow:
        0 0 10px #00f5ff,
        0 0 20px #00f5ff,
        0 0 40px #00f5ff;
    animation: flicker 2s infinite alternate;
}

@keyframes flicker {
    0% {
        opacity: 1;
    }
    100% {
        opacity: 0.92;
    }
}

.subtitle {
    text-align: center;
    color: #d1d5db;
    font-size: 20px;
    font-weight: 500;
    margin-bottom: 35px;
    letter-spacing: 1px;
}

.panel {
    background: rgba(0, 0, 0, 0.55);
    border: 1px solid rgba(0,255,255,0.2);
    border-radius: 28px;
    padding: 25px;
    backdrop-filter: blur(16px);
    box-shadow:
        0 0 20px rgba(0,255,255,0.12),
        inset 0 0 15px rgba(255,255,255,0.03);
    transition: all 0.3s ease;
}

.panel:hover {
    border: 1px solid rgba(0,255,255,0.5);
    box-shadow:
        0 0 35px rgba(0,255,255,0.25);
}


section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg,
        rgba(0,0,0,0.96),
        rgba(15,23,42,0.96));
    border-right: 1px solid rgba(0,255,255,0.15);
}


section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
    font-family: 'Rajdhani', sans-serif;
}


.stButton > button,
.stDownloadButton > button {
    width: 100%;
    border: none;
    border-radius: 16px;
    padding: 14px 22px;
    background: linear-gradient(135deg, #00f5ff, #0066ff);
    color: white;
    font-size: 16px;
    font-weight: 700;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
    transition: all 0.3s ease;
    box-shadow:
        0 0 15px rgba(0,245,255,0.3);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow:
        0 0 30px rgba(0,245,255,0.6);
}

.stSlider label {
    color: #00f5ff !important;
    font-weight: 700;
}

.stSlider > div > div {
    color: #00f5ff !important;
}

.stSelectbox label {
    color: #00f5ff !important;
    font-weight: 700;
}

.stSelectbox div[data-baseweb="select"] {
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(0,255,255,0.2);
    border-radius: 15px;
}

.stToggle label {
    color: #ffffff !important;
    font-weight: 600;
}

img {
    border-radius: 24px;
    border: 2px solid rgba(0,255,255,0.15);
    box-shadow:
        0 0 20px rgba(0,255,255,0.15);
    transition: all 0.3s ease;
}

img:hover {
    transform: scale(1.03);
    border: 2px solid rgba(0,255,255,0.5);
    box-shadow:
        0 0 35px rgba(0,255,255,0.4);
}


h2 {
    color: #00f5ff !important;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 2px;
}


.stInfo {
    background: rgba(0,255,255,0.08) !important;
    border: 1px solid rgba(0,255,255,0.2) !important;
    border-radius: 16px !important;
}


.dev-footer {
    text-align: center;
    margin-top: 45px;
    padding: 18px;
    border-radius: 18px;
    background: rgba(0,0,0,0.55);
    border: 1px solid rgba(0,255,255,0.15);
    color: #d1d5db;
    font-size: 16px;
    letter-spacing: 1px;
    backdrop-filter: blur(12px);
    box-shadow:
        0 0 20px rgba(0,255,255,0.1);
}

.dev-footer span {
    color: #00f5ff;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
}


#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* SCROLLBAR */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #000000;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(#00f5ff, #0066ff);
    border-radius: 20px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(#00d9ff, #0051ff);
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="title">📹 Live Object Detection & Tracing</div>

<div class="subtitle">
Real-Time AI Detection using YOLOv8 + Streamlit
</div>
""", unsafe_allow_html=True)

with st.sidebar:

    st.markdown("## ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        0.5,
        0.05
    )

    target_object = st.selectbox(
        "🚨 Alert Target",
        CLASS_NAMES
    )

    save_images = st.toggle(
        "📸 Save Detection",
        value=True
    )

    show_boxes = st.toggle(
        "🟦 Show Bounding Boxes",
        value=True
    )

class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.prev_objects = set()
        self.alert_sent = False

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        results = model.predict(
            img,
            conf=confidence,
            imgsz=480,
            verbose=False
        )

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

                if label == target_object:
                    alert_detected = True

                if show_boxes:

                    color = (56, 189, 248)

                    if label == target_object:
                        color = (0, 0, 255)

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3
                    )

                    cv2.putText(
                        img,
                        f"{label}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

        # =========================
        # SEND TWILIO SMS ALERT
        # =========================

        if alert_detected and not self.alert_sent:

            try:
                twilio_client.messages.create(
                    body=f"🚨 ALERT: {target_object.upper()} detected by YOLOv8 system.",
                    from_=TWILIO_PHONE_NUMBER,
                    to=ALERT_PHONE_NUMBER
                )

                print("SMS Alert Sent!")

            except Exception as e:
                print("Twilio Error:", e)

            self.alert_sent = True

        if not alert_detected:
            self.alert_sent = False

        total_objects = sum(detected_counts.values())

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (350, 140),
            (15, 23, 42),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.7,
            img,
            0.3,
            0,
            img
        )

        cv2.putText(
            img,
            f"Total Objects: {total_objects}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (56, 189, 248),
            2
        )

        y_position = 80

        for obj, count in detected_counts.items():

            cv2.putText(
                img,
                f"{obj}: {count}",
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y_position += 30

        if alert_detected:

            alert_text = f"ALERT: {target_object.upper()} DETECTED"

            (text_width, text_height), _ = cv2.getTextSize(
                alert_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            cv2.rectangle(
                img,
                (15, 15),
                (text_width + 35, 55),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                img,
                alert_text,
                (25, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            print("\a")

        if save_images and current_objects != self.prev_objects:

            filename = datetime.now().strftime(
                "%Y%m%d_%H%M%S.jpg"
            )

            filepath = os.path.join(
                SAVE_DIR,
                filename
            )

            cv2.imwrite(filepath, img)

            self.prev_objects = current_objects

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

st.markdown('<div class="panel">', unsafe_allow_html=True)

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("## 📂 Detection Gallery")

image_files = glob.glob(os.path.join(SAVE_DIR, "*.jpg"))

if image_files:

    cols = st.columns(3)

    for index, img_path in enumerate(reversed(image_files[-9:])):

        with cols[index % 3]:

            st.image(
                img_path,
                use_container_width=True
            )

            st.caption(
                os.path.basename(img_path)
            )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w"
    ) as zip_file:

        for file in image_files:

            zip_file.write(
                file,
                os.path.basename(file)
            )

    st.download_button(
        label="⬇ Download Detection Logs",
        data=zip_buffer.getvalue(),
        file_name="detection_logs.zip",
        mime="application/zip"
    )

else:
    st.info("No saved detections yet.")
