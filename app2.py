import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="♻️ Smart Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

# =========================
# Custom CSS for Styling
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #d0f0c0, #a0e0a0);
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2a7f2a;
        text-align: center;
    }
    .prediction {
        font-size: 30px;
        font-weight: bold;
        color: #1f5f1f;
    }
    </style>
    """, unsafe_allow_html=True
)

# =========================
# Load TFLite Model
# =========================
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# Prediction Function
# =========================
def predict(image: Image.Image):
    IMG_HEIGHT = input_details[0]['shape'][1]
    IMG_WIDTH = input_details[0]['shape'][2]
    
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    return prediction

# =========================
# Streamlit Interface
# =========================
st.markdown('<div class="title">♻️ Smart Waste Classifier ♻️</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing...", unsafe_allow_html=True)
    prediction = predict(image)
    
    # Mapping example classes, عدّلي حسب مودلك
    classes = {0: "Plastic", 1: "Metal", 2: "Paper", 3: "Organic", 4: "Glass"}
    st.markdown(f'<div class="prediction">Prediction: {classes.get(prediction, "Unknown")}</div>', unsafe_allow_html=True)
