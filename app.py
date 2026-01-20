import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from streamlit_webrtc import webrtc_streamer
import av

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🕳️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* -------- BACKGROUND -------- */
body {
    background-color: #87CEEB;
}
.stApp {
    background-color: #87CEEB;
}

/* -------- HEADERS -------- */
.main-title {
    font-size: 45px;
    font-weight: 800;
    color: #0f172a;
}
.subtitle {
    color: #1e293b;
    font-size: 18px;
}

/* -------- CARD STYLE + HOVER -------- */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    transition: all 0.35s ease-in-out;
}
.card:hover {
    transform: translateY(-10px) scale(1.02);
    box-shadow: 0px 25px 45px rgba(0,0,0,0.18);
}

/* -------- SLIDER ANIMATION -------- */
div[data-baseweb="slider"] {
    transition: all 0.3s ease-in-out;
}
div[data-baseweb="slider"]:hover {
    transform: scale(1.03);
}
div[data-baseweb="slider"] span {
    background-color: #0ea5e9 !important;
}

/* -------- FOOTER -------- */
.footer {
    text-align: center;
    color: #334155;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER (TEXT LEFT, GIF RIGHT) ----------------
header_col1, header_col2 = st.columns([3, 2])

with header_col1:
    st.markdown("<div class='main-title'>🕳️ Pothole Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>YOLO-based Intelligent Road Damage Detection</div>", unsafe_allow_html=True)

with header_col2:
    st.image("PotholeGIF.gif", use_column_width=True)

st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    if not os.path.exists(model_path):
        st.error("❌ Model file (best.pt) not found")
        st.stop()
    return YOLO(model_path)

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Configure detection settings")

mode = st.sidebar.radio(
    "🎥 Select Detection Mode",
    ("📷 Live Camera", "📤 Upload Image / Video")
)

confidence = st.sidebar.slider(
    "🎯 Confidence Threshold",
    0.1, 1.0, 0.4
)

st.sidebar.markdown("---")
st.sidebar.success("🟢 Model Loaded Successfully")

# ---------------- UTILITY FUNCTION ----------------
def draw_boxes(frame, results):
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"POTHOLE {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
    return frame

# ---------------- MAIN CONTENT ----------------
if "Live Camera" in mode:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📷 Real-Time Pothole Detection (Browser Camera)")
    st.info("Uses browser camera – works on Streamlit Cloud & Mobile")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, stream=True)
        img = draw_boxes(img, results)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="pothole-live",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Image or Video")
    st.info("Supported formats: JPG, PNG, MP4, AVI")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )

    if uploaded_file:
        if "image" in uploaded_file.type:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            results = model(image_np, stream=True)
            image_np = draw_boxes(image_np, results)

            st.image(image_np, caption="✅ Detected Potholes", use_column_width=True)

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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame)

            cap.release()
            os.unlink(tfile.name)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div class='footer'>🚀 Developed for Smart Road Monitoring | YOLO + Streamlit</div>",
    unsafe_allow_html=True
)
