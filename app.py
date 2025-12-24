import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("🚦 Real-Time Accident Detection System")

model = load_model("model.h5")
IMG_SIZE = 224

run = st.checkbox("Start Camera")
frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.error("🚨 Accident Detected! Alert Sent")
    else:
        st.success("✅ Normal Traffic")

    frame_window.image(frame, channels="BGR")

cap.release()
