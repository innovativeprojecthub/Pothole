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

/* Card */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-10px);
}

/* Slider */
div[data-baseweb="slider"]:hover {
    transform: scale(1.03);
}

/* Footer */
.footer {
    text-align: center;
    color: #334155;
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

confidence = st.sidebar.slider(
    "🎯 Confidence Threshold",
    0.1, 1.0, 0.4
)

st.sidebar.success("🟢 Model Loaded Successfully")

# ---------------- DRAW FUNCTION ----------------
def draw_boxes(frame, results):
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"POTHOLE {conf:.2f}",
                            (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0), 2)
    return frame

# ---------------- MAIN ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📸 Capture or Upload")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
camera_image = st.camera_input("Take Photo")
video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

# ---------- IMAGE FROM CAMERA OR UPLOAD ----------
input_image = None

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")

elif camera_image:
    input_image = Image.open(camera_image).convert("RGB")

if input_image:
    st.image(input_image, caption="📷 Input Image", use_column_width=True)

    img_np = np.array(input_image).astype(np.uint8)

    try:
        results = model(img_np)
        img_np = draw_boxes(img_np, results)
    except Exception as e:
        st.error(f"Detection error: {e}")
        st.stop()

    st.image(img_np, caption="✅ Detected Potholes", use_column_width=True)

# ---------- VIDEO ----------
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

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
            st.error(f"Video error: {e}")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

    cap.release()
    os.unlink(tfile.name)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<div class='footer'>🚀 Smart Road Monitoring System</div>", unsafe_allow_html=True)
