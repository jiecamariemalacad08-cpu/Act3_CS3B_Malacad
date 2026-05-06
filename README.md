🎥 Live Object Detection & Tracing

📌 Overview

This project is a real-time object detection system built using YOLOv8, OpenCV, and Streamlit WebRTC. It detects and tracks objects from a live camera feed and automatically saves frames as images whenever objects are detected.

The system is optimized for smooth performance and includes object counting, alerts, and intelligent frame saving.

🚀 Features

🎯 Real-Time Object Detection & Tracking
🔢 Object Counting (per frame)
🚨 Alert System for Specific Objects
📸 Automatic Frame Saving
⚡ Optimized for Smooth Performance (Reduced Lag)
💻 Web-based Interface (Streamlit)
🛠️ Technologies Used

Python
Streamlit
streamlit-webrtc
OpenCV (cv2)
Ultralytics YOLOv8
AV (PyAV)
📂 Project Structure

project_folder/ │ ├── main.py # Main Streamlit application ├── saved_frames/ # Automatically saved images └── README.md # Project documentation

📸 How It Works

The system accesses your webcam using WebRTC.
Each video frame is processed using YOLOv8.
Detected objects are:
Labeled and tracked
Counted in real-time
If objects are detected:
The frame is automatically saved every few seconds
Alerts are triggered for specific objects (e.g., "person")
🧠 Key Functionalities

✔ Object Counting

Displays the number of detected objects per class (e.g., person, phone).

✔ Alert System

Shows alert text when a specific object is detected:

ALERT_OBJECT = "person"

✔ Automatic Frame Saving

Saves frames only when objects are detected
Uses time interval to avoid duplicate images
Saved images location:

saved_frames/

⚡ Performance Optimization

Frame resizing (640x480) to reduce processing load
Lightweight YOLO model ("yolov8n.pt")
Controlled frame saving using time intervals
⚠️ Limitations

On Streamlit Cloud, saved images are temporary and may be deleted after app restart
Requires a working webcam
Detection accuracy depends on lighting and camera quality
🎯 Future Enhancements

📂 Image gallery viewer in UI
📥 Download saved images
☁️ Cloud storage integration (Google Drive / Firebase)
🔊 Sound alert system
📊 Real-time analytics dashboard
👨‍💻 Author

Jieca Marie J. Malacad BSCS 3B
