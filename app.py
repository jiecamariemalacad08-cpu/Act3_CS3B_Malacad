import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import os
import time
from datetime import datetime

CONFIDENCE = 0.4
ALERT_OBJECT = "person"
SAVE_INTERVAL = 3

os.makedirs("saved_frames", exist_ok=True)

@st.cache_resource
def load_model():

    return YOLO("yolov8n.pt")

model = load_model()

st.set_page_config(page_title="Object Detection", layout="wide")

st.title("🎥 Live Object Detection & Tracking")
st.write("Real-time object detection using YOLOv8 + Streamlit")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_save_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.resize(img, (640, 480))

        results = model.predict(
            img,
            conf=CONFIDENCE,
            verbose=False
        )

        annotated_frame = results[0].plot()

        counts = {}

        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                label = model.names[int(cls)]
                counts[label] = counts.get(label, 0) + 1

        y = 30
        for obj, count in counts.items():
            cv2.putText(
                annotated_frame,
                f"{obj}: {count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y += 30

        if ALERT_OBJECT in counts:
            cv2.putText(
                annotated_frame,
                f"ALERT: {ALERT_OBJECT} DETECTED!",
                (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        current_time = time.time()

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            if current_time - self.last_save_time > SAVE_INTERVAL:

                filename = (
                    f"saved_frames/frame_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )

                cv2.imwrite(filename, annotated_frame)

                self.last_save_time = current_time

                print("Saved:", filename)

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )

webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
