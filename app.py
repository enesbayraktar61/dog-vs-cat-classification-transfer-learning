import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Configuration 
st.set_page_config(page_title="Dog vs Cat Classifier", layout="centered")
st.title("🐶 Dog vs. 🐱 Cat Classifier")
st.write("Upload an image, and the model will predict whether it's a **Dog** or a **Cat**.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dog_cat_efficientnetb0.keras")

@st.cache_resource
def load_my_model():
    # compile=False avoids errors related to optimizers during inference
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

# Image Processing 
def prepare_image(img):
    # Ensure image is in RGB mode
    img = img.convert("RGB")
    # Resize to the input size required by EfficientNetB0
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32")
    # Crucial: Apply EfficientNet specific scaling
    img_array = preprocess_input(img_array)
    # Add batch dimension: (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

# User Interface 
st.subheader("Upload Section")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner('Analyzing...'):
        # Prediction
        X = prepare_image(image)
        prediction = model.predict(X)[0][0] # Extract the single sigmoid value
        
        # Classification Logic
        # Sigmoid output: 0.0 to 1.0 (threshold 0.5)
        is_dog = prediction >= 0.5
        confidence = prediction if is_dog else (1 - prediction)
        label = "Dog 🐶" if is_dog else "Cat 🐱"
        
        st.divider()
        st.subheader(f"Result: {label}")
        
        # Visual feedback with a progress bar for confidence
        st.write(f"Confidence: **{confidence:.2%}**")
        st.progress(float(confidence))

        if is_dog:
            st.success("The model is confident this is a Dog!")
        else:
            st.info("The model is confident this is a Cat!")

st.divider()
st.caption("Model: EfficientNetB0 | Framework: TensorFlow/Keras | Task: Binary Classification")