import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to classify tumor type")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/brain_tumor_model.h5")

model = load_model()

# IMPORTANT: match training order
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# ------------------------------
# PREPROCESS FUNCTION
# ------------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)
    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction")
    st.success(f"{predicted_class.upper()} ({confidence*100:.2f}%)")

    st.subheader("Probabilities")

    probs = prediction[0]
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    st.bar_chart(prob_dict)
