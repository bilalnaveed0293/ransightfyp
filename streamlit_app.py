import streamlit as st
import tensorflow as st_tf # We'll use a local alias
import numpy as np
from PIL import Image
import os

# --- 1. Page Config ---
st.set_page_config(page_title="Ransomware Detector", page_icon="🛡️")
st.title("🛡️ Ransomware Binary Classifier")
st.write("Upload an `.exe` file to visualize it as a Natraj grayscale image and predict its status.")

# --- 2. Load Model (Cached so it doesn't reload every time) ---
@st.cache_resource
def load_my_model():
    # Use tensorflow-cpu to stay under the 1GB RAM limit
    import tensorflow as tf
    return tf.keras.models.load_model("cnn.keras")

model = load_my_model()

# --- 3. Natraj Logic ---
def get_natraj_width(file_size_kb):
    if file_size_kb < 10: return 32
    elif file_size_kb < 30: return 64
    elif file_size_kb < 60: return 128
    elif file_size_kb < 100: return 256
    elif file_size_kb < 200: return 384
    elif file_size_kb < 500: return 512
    elif file_size_kb < 1000: return 768
    else: return 1024

# --- 4. The UI ---
uploaded_file = st.file_uploader("Choose an executable file", type=["exe", "bin"])

if uploaded_file is not None:
    # Read bytes
    bytes_data = uploaded_file.getvalue()
    file_size_kb = len(bytes_data) / 1024
    width = get_natraj_width(file_size_kb)
    
    # Natraj conversion
    img_array = np.frombuffer(bytes_data, dtype=np.uint8)
    height = len(img_array) // width
    img_array = img_array[:height * width].reshape((height, width))
    pil_img = Image.fromarray(img_array)

    # Layout: Two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Binary Visualization")
        st.image(pil_img, use_container_width=True, caption=f"Width: {width}px")

    with col2:
        st.subheader("Model Prediction")
        # Resize for model
        input_img = pil_img.resize((224, 224))
        final_array = np.array(input_img).astype(np.float32)
        final_array = np.expand_dims(final_array, axis=(0, -1))
        
        # Predict
        prediction = model.predict(final_array)
        prob = float(prediction[0][0])
        
        if prob > 0.5:
            st.error(f"⚠️ RANSOMWARE DETECTED ({prob*100:.2f}%)")
        else:
            st.success(f"✅ BENIGN FILE ({ (1-prob)*100:.2f}%)")
