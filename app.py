import streamlit as st
import cv2
import numpy as np
import joblib
import os
import gdown
from skimage.feature import hog

# -----------------------------
# Load model & scaler (from Google Drive)
# -----------------------------
MODEL_PATH = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"
GDRIVE_ID = "16VjPLyDMwe4La9aHuCZcFaijbvLhgklL"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Constants
# -----------------------------
IMG_SIZE = (128, 128)
CATEGORIES = ["Healthy Rice Leaf", "Bacterial Leaf"]

# -----------------------------
# HOG Feature Extraction
# -----------------------------
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AgriScan AI", layout="centered")

st.title("AgriScan AI")
st.subheader("Rice Leaf Disease Classification")
st.write("Upload a rice leaf image to detect disease")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR → RGB for Streamlit
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image_resized = cv2.resize(image, IMG_SIZE)
    hog_features = extract_hog_features(image_resized)
    hog_features = scaler.transform(hog_features.reshape(1, -1))

    # Prediction
    prediction = model.predict(hog_features)[0]
    confidence = abs(model.decision_function(hog_features)[0])

    st.success(f"Prediction: **{CATEGORIES[prediction]}**")
    st.info(f"Confidence Score: **{confidence:.2f}**")
