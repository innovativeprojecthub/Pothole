import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Pothole Detection", layout="wide")
st.title("🕳️ Pothole Detection System using YOLO")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Detection Mode")
mode = st.sidebar.radio(
    "Select Detection Type",
    ("Live Camera", "Upload Image / Video")
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4
)

# ---------------- FUNCTIONS ----------------
def draw_boxes(frame, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Pothole {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
    return frame

# ---------------- LIVE CAMERA MODE ----------------
if mode == "Live Camera":
    st.subheader("📷 Live Camera Pothole Detection")

    run = st.checkbox("Start Camera")

    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        results = model(frame, stream=True)
        frame = draw_boxes(frame, results)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cap.release()

# ---------------- UPLOAD IMAGE / VIDEO MODE ----------------
else:
    st.subheader("📤 Upload Image or Video")

    uploaded_file = st.file_uploader(
        "Upload Image or Video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )

    if uploaded_file is not None:

        file_type = uploaded_file.type

        # ---------- IMAGE ----------
        if "image" in file_type:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            results = model(image_np, stream=True)
            image_np = draw_boxes(image_np, results)

            st.image(image_np, caption="Detected Potholes", use_column_width=True)

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

                results = model(frame, stream=True)
                frame = draw_boxes(frame, results)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame)

            cap.release()
            os.unlink(tfile.name)

