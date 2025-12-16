import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# =========================
# Page Configuration
st.set_page_config(page_title="Smart Waste Classifier", layout="wide")

st.title("♻️ Smart Waste Classifier")
st.write("ضع الصورة لتصنيف نوع النفايات")

# تحميل المودل
@st.cache_resource
def load_mymodel():
    model = load_model("my_mobnet_model.h5")
    return model

model = load_mymodel()

# رفع الصورة
uploaded_file = st.file_uploader("اختر صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='الصورة المختارة', use_column_width=True)

    # تجهيز الصورة للمودل
    img = image.resize((224, 224))  # حجم المودل MobileNet
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    classes = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]  # عدلي حسب الكلاسات الحقيقية
    st.success(f"✅ النوع المتوقع: {classes[class_index]}")

