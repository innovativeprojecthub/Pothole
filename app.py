import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from streamlit_webrtc import webrtc_streamer
import av

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Smart Road Monitoring System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== ADVANCED CSS ==================
st.markdown("""
<style>
html, body {
    background: linear-gradient(145deg, #020617, #020617);
    color: #e5e7eb;
}
.dashboard-title {
    font-size: 42px;
    font-weight: 900;
    text-align: left;
    color: #22c55e;
}
.dashboard-subtitle {
    font-size: 16px;
    color: #9ca3af;
}
.kpi-card {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 0 25px rgba(34,197,94,0.15);
}
.section-card {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
}
.status-ok {
    color: #22c55e;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='dashboard-title'>SMART ROAD DAMAGE MONITORING</div>", unsafe_allow_html=True)
    st.markdown("<div class='dashboard-subtitle'>AI-powered pothole detection using deep learning</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<br><span class='status-ok'>● SYSTEM ONLINE</span>", unsafe_allow_html=True)

st.markdown("---")

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    if not os.path.exists(model_path):
        st.error("Model not found")
        st.stop()
    return YOLO(model_path)

model = load_model()

# ================== KPI BAR ==================
k1, k2, k3, k4 = st.columns(4)
k1.markdown("<div class='kpi-card'>🧠 AI Model<br><b>YOLOv8</b></div>", unsafe_allow_html=True)
k2.markdown("<div class='kpi-card'>🎯 Accuracy<br><b>High Precision</b></div>", unsafe_allow_html=True)
k3.markdown("<div class='kpi-card'>⚡ Inference<br><b>Real-Time</b></div>", unsafe_allow_html=True)
k4.markdown("<div class='kpi-card'>🌐 Deployment<br><b>Cloud Ready</b></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.title("CONTROL CENTER")
st.sidebar.markdown("Select operational mode")

mode = st.sidebar.radio(
    "Detection Mode",
    ["Live Surveillance Camera", "Media File Analysis"]
)

confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.45)

st.sidebar.markdown("---")
st.sidebar.markdown("🟢 Model Status: **ACTIVE**")

# ================== DRAW FUNCTION ==================
def draw_boxes(frame, results):
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (34,197,94), 3)
                cv2.putText(
                    frame,
                    f"POTHOLE",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (34,197,94),
                    2
                )
    return frame

# ================== MAIN SECTION ==================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)

if mode == "Live Surveillance Camera":
    st.subheader("LIVE ROAD SURVEILLANCE")
    st.caption("Browser-based secure camera feed")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, stream=True)
        img = draw_boxes(img, results)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}
    )

else:
    st.subheader("MEDIA FILE ANALYSIS")
    st.caption("Upload images or videos captured by inspection vehicles")

    uploaded_file = st.file_uploader(
        "Upload road footage",
        type=["jpg", "png", "jpeg", "mp4", "avi", "mov"]
    )

    if uploaded_file:
        if "image" in uploaded_file.type:
            img = np.array(Image.open(uploaded_file))
            results = model(img, stream=True)
            img = draw_boxes(img, results)
            st.image(img, use_column_width=True)

        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.image([])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, stream=True)
                frame = draw_boxes(frame, results)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()
            os.unlink(tfile.name)

st.markdown("</div>", unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    "<div class='footer'>© 2026 Smart Infrastructure AI Platform | YOLO + Streamlit</div>",
    unsafe_allow_html=True
)
