import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# Page Configuration
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Waste Classifier")
st.markdown(
    "<h3 style='text-align: center; color: green;'>Classify your waste and recycle smartly!</h3>",
    unsafe_allow_html=True
)

# =========================
# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# =========================
# Classes
classes = ["plastic", "metal", "glass", "organic"]  # عدليها حسب مودلك

# =========================
# Upload Image
uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # =========================
    # Preprocess Image
    IMG_HEIGHT, IMG_WIDTH = 224, 224  # عدلي حسب مودلك
    img_array = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # Predict
    predictions = model.predict(img_array)
    pred_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f}%)")

