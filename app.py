import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🕳️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #87CEEB;
}
.stApp {
    background-color: #87CEEB;
}

/* Header */
.main-title {
    font-size: 45px;
    font-weight: 800;
    color: #0f172a;
}
.subtitle {
    color: #1e293b;
    font-size: 18px;
}

/* Card Hover */
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

/* Slider animation */
div[data-baseweb="slider"] {
    transition: all 0.3s ease-in-out;
}
div[data-baseweb="slider"]:hover {
    transform: scale(1.03);
}
div[data-baseweb="slider"] span {
    background-color: #0ea5e9 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #334155;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='main-title'>🕳️ Pothole Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>YOLO-based Intelligent Road Damage Detection</div>", unsafe_allow_html=True)

with col2:
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

mode = st.sidebar.radio(
    "🎥 Select Detection Mode",
    ("📤 Upload Image / Video",)
)

confidence = st.sidebar.slider(
    "🎯 Confidence Threshold",
    0.1, 1.0, 0.4
)

st.sidebar.success("🟢 Model Loaded Successfully")

# ---------------- DRAW FUNCTION ----------------
def draw_boxes(frame, results):
    try:
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
    except Exception as e:
        st.error(f"Drawing error: {e}")

    return frame

# ---------------- MAIN ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📤 Upload Image or Video")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file:

    # ---------- IMAGE ----------
    if "image" in uploaded_file.type:
        image = Image.open(uploaded_file)

        # Fix 1: convert to RGB
        image = image.convert("RGB")

        image_np = np.array(image)

        # Fix 2: dtype
        image_np = image_np.astype(np.uint8)

        # Fix 3: validate
        if image_np is None or len(image_np.shape) != 3:
            st.error("Invalid image format")
            st.stop()

        try:
            results = model(image_np)   # FIXED (no stream=True)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

        image_np = draw_boxes(image_np, results)

        st.image(image_np, caption="✅ Detected Potholes", use_column_width=True)

    # ---------- VIDEO ----------
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = model(frame)
                frame = draw_boxes(frame, results)
            except Exception as e:
                st.error(f"Video processing error: {e}")
                break

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
