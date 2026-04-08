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

# ---------------- CSS ----------------
st.markdown("""
<style>
body { background-color: #87CEEB; }
.stApp { background-color: #87CEEB; }

.main-title {
    font-size: 45px;
    font-weight: 800;
    color: #0f172a;
}
.subtitle {
    color: #1e293b;
    font-size: 18px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-10px);
}

.footer {
    text-align: center;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
col1, col2 = st.columns([3,2])

with col1:
    st.markdown("<div class='main-title'>🕳️ Pothole Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>YOLO-based Intelligent Detection</div>", unsafe_allow_html=True)

with col2:
    st.image("PotholeGIF.gif", use_column_width=True)

st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

mode = st.sidebar.radio(
    "Select Mode",
    ["📷 Live Camera", "📤 Upload Image", "🎥 Upload Video"]
)

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)

st.sidebar.success("🟢 Model Loaded")

# ---------------- DRAW FUNCTION ----------------
def draw_boxes(frame, results):
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"POTHOLE {conf:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)
    return frame

# ================= LIVE CAMERA =================
if mode == "📷 Live Camera":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📷 Live Camera Detection (Local Only)")

    run = st.checkbox("Start Camera")

    col1, col2 = st.columns(2)
    live = col1.image([])
    detected = col2.image([])

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not accessible")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    break

                live.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_small = cv2.resize(frame, (640,480))
                results = model(frame_small)
                frame_small = draw_boxes(frame_small, results)

                detected.image(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

        cap.release()

    st.markdown("</div>", unsafe_allow_html=True)

# ================= IMAGE =================
elif mode == "📤 Upload Image":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    camera = st.camera_input("Or Take Photo")

    img = None

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
    elif camera:
        img = Image.open(camera).convert("RGB")

    if img:
        st.image(img, caption="Input")

        img_np = np.array(img).astype(np.uint8)

        results = model(img_np)
        img_np = draw_boxes(img_np, results)

        st.image(img_np, caption="Detected")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= VIDEO =================
elif mode == "🎥 Upload Video":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if video_file:
        st.warning("Processing video...")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        stframe = st.image([])

        count = 0
        skip = 3  # speed optimization

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % skip != 0:
                continue

            frame = cv2.resize(frame, (640,480))

            results = model(frame)
            frame = draw_boxes(frame, results)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            progress.progress(min(int((count/total)*100), 100))

        cap.release()
        os.unlink(tfile.name)

        st.success("Video completed")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<div class='footer'>🚀 Smart Road Monitoring System</div>", unsafe_allow_html=True)
