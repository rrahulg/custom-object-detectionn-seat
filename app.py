import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model.yolo_model import load_yolo_model, predict_yolo


@st.cache_resource
def get_model():
    return load_yolo_model(weights="runs/train/exp/weights/best.pt")


model = get_model()

st.title("🦾 YOLO Seat Detection App")
st.write("Upload an image to detect empty and occupied seats.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image)
    results = predict_yolo(model, image_np, conf_threshold=0.25)
    annotated_image = results[0].plot()

    st.image(annotated_image, caption="Annotated Image", use_column_width=True)
