import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import requests

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

# ---------------- MODEL DOWNLOAD ----------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"   # 🔥 replace

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("✅ Model downloaded")
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

mode = st.sidebar.radio(
    "Select Mode",
    ["📷 Live Photo", "📤 Upload Image", "🎥 Upload Video"]
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

# ================= LIVE PHOTO =================
if mode == "📷 Live Photo":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📷 Capture Live Photo")

    camera_image = st.camera_input("Take Photo")

    if camera_image:
        img = Image.open(camera_image).convert("RGB")
        st.image(img, caption="Captured Image")

        img_np = np.array(img).astype(np.uint8)

        results = model(img_np)
        img_np = draw_boxes(img_np, results)

        st.image(img_np, caption="Detected Potholes")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= IMAGE =================
elif mode == "📤 Upload Image":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image")

        img_np = np.array(img).astype(np.uint8)

        results = model(img_np)
        img_np = draw_boxes(img_np, results)

        st.image(img_np, caption="Detected Potholes")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= VIDEO =================
elif mode == "🎥 Upload Video":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if video_file:
        st.warning("⏳ Processing video... Please wait")

        input_path = "input_video.mp4"
        with open(input_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = "output_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        count = 0
        skip = 3

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            frame = cv2.resize(frame, (640,480))

            if count % skip == 0:
                results = model(frame)
                frame = draw_boxes(frame, results)

            out.write(frame)

            progress.progress(min(int((count/total)*100), 100))

        cap.release()
        out.release()

        st.success("✅ Video Processing Completed")

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download Processed Video",
                data=f,
                file_name="pothole_detected.mp4",
                mime="video/mp4"
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<div class='footer'>🚀 Smart Road Monitoring System</div>", unsafe_allow_html=True)
