import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Brain Tumor Detection App")

# Load model
model = load_model("model.h5")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        st.error("Tumor detected")
    else:
        st.success("No tumor detected")
