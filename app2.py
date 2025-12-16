import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# =========================
# Page Configuration
st.set_page_config(page_title="Smart Waste Classifier", layout="wide")

st.title("♻️ Smart Waste Classifier")
st.markdown("<h3 style='text-align: center; color: green;'>Upload an image to classify waste type</h3>", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_mymodel():
    model = load_model("my_mobnet_model.h5")
    return model

model = load_mymodel()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Selected Image', use_column_width=True)

    # Prepare image for model
    img = image.resize((224, 224))  # MobileNet input size
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    classes = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]  # Replace with your actual classes
    st.success(f"✅ Predicted Class: {classes[class_index]}")


