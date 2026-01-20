import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from streamlit_webrtc import webrtc_streamer
import av

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Vision Playground",
    page_icon="✨",
    layout="wide"
)

# ================= ULTRA CUSTOM CSS =================
st.markdown("""
<style>
@keyframes float {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
  100% { transform: translateY(0px); }
}

@keyframes gradient {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

.hero-text {
  font-size: 64px;
  font-weight: 900;
  background: linear-gradient(270deg, #00f5a0, #00d9f5, #a855f7);
  background-size: 600% 600%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: gradient 6s ease infinite;
}

.hero-sub {
  font-size: 20px;
  color: #475569;
}

.glass {
  background: rgba(255,255,255,0.55);
  backdrop-filter: blur(14px);
  border-radius: 25px;
  padding: 30px;
  box-shadow: 0 30px 60px rgba(0,0,0,0.1);
  transition: all 0.4s ease;
}

.glass:hover {
  transform: translateY(-12px) scale(1.02);
  box-shadow: 0 40px 80px rgba(0,0,0,0.18);
}

.mode-btn {
  border-radius: 50px !important;
  font-size: 18px !important;
  padding: 12px 30px !important;
}

.footer {
  text-align: center;
  font-size: 14px;
  color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# ================= HERO SECTION =================
col1, col2 = st.columns([3,2])

with col1:
    st.markdown("<div class='hero-text'>AI Vision Playground</div>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Real-time pothole detection powered by deep learning</p>", unsafe_allow_html=True)
    st.markdown("🚀 **See. Detect. Improve Roads.**")

with col2:
    st.image("PotholeGIF.gif", use_column_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    if not os.path.exists(model_path):
        st.error("Model file missing")
        st.stop()
    return YOLO(model_path)

model = load_model()

# ================= MODE SELECTION =================
st.markdown("## 🔀 Choose How You Want to Detect")

mode_col1, mode_col2 = st.columns(2)

with mode_col1:
    live_mode = st.button("🎥 Live Camera", use_container_width=True)

with mode_col2:
    upload_mode = st.button("📤 Upload Media", use_container_width=True)

if "mode" not in st.session_state:
    st.session_state.mode = None

if live_mode:
    st.session_state.mode = "live"

if upload_mode:
    st.session_state.mode = "upload"

confidence = st.slider("🎯 Detection Sensitivity", 0.1, 1.0, 0.45)

# ================= DRAW FUNCTION =================
def draw_boxes(frame, results):
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 3)
                cv2.putText(frame, "POTHOLE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
    return frame

# ================= LIVE MODE =================
if st.session_state.mode == "live":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🎥 Live Browser Camera Detection")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, stream=True)
        img = draw_boxes(img, results)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live-playground",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ================= UPLOAD MODE =================
if st.session_state.mode == "upload":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Image or Video")

    file = st.file_uploader("Drop your road image or video here",
                            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if file:
        if "image" in file.type:
            img = np.array(Image.open(file))
            results = model(img, stream=True)
            img = draw_boxes(img, results)
            st.image(img, use_column_width=True)

        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
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

# ================= FOOTER =================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div class='footer'>✨ Built with YOLO & Streamlit | AI Vision Playground</div>", unsafe_allow_html=True)
