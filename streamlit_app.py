import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

# --- 1. Page Config ---
st.set_page_config(page_title="Ransomware Detector XAI", page_icon="🔎", layout="wide")
st.title("🔎 Ransomware Classifier with Grad-CAM")
st.write("Upload an `.exe` file to classify it and visualize the exact byte regions the CNN focused on.")

# --- 2. Load Model ---
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("cnn1.keras")

model = load_my_model()

# --- 3. Helper Functions ---
def get_standard_width(file_size_bytes):
    if file_size_bytes < 10 * 1024: return 32
    elif file_size_bytes < 30 * 1024: return 64
    elif file_size_bytes < 60 * 1024: return 128
    elif file_size_bytes < 100 * 1024: return 256
    elif file_size_bytes < 200 * 1024: return 384
    elif file_size_bytes < 500 * 1024: return 512
    elif file_size_bytes < 1000 * 1024: return 768
    else: return 1024

def get_last_conv_layer_name(keras_model):
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, keras_model, last_conv_layer_name, pred_index=None):
    # Create a clean input tensor
    grad_model_input = tf.keras.Input(shape=(128, 128, 1))
    x = grad_model_input
    last_conv_output = None

    # Re-connect the layers manually
    for layer in keras_model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x

    grad_model = tf.keras.models.Model(
        inputs=grad_model_input,
        outputs=[last_conv_output, x]
    )

    # Compute Gradient
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_overlay(pil_img, heatmap):
    # Resize heatmap to match image
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((128, 128), resample=Image.BICUBIC)
    heatmap_resized = np.array(heatmap_resized)
    
    # Apply Jet Colormap using Matplotlib instead of OpenCV
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    
    # Convert original image to RGB
    img_rgb = np.stack((np.array(pil_img),)*3, axis=-1) / 255.0
    
    # Superimpose
    overlay = jet_heatmap * 0.4 + img_rgb * 0.6
    overlay = np.uint8(255 * overlay)
    return Image.fromarray(overlay)

# --- 4. Main UI Flow ---
uploaded_file = st.file_uploader("Choose an executable file (.exe)", type=["exe", "bin"])

if uploaded_file is not None:
    with st.spinner("Analyzing binary structure..."):
        # 1. Process Bytes
        bytes_data = uploaded_file.getvalue()
        width = get_standard_width(len(bytes_data))
        height = int(np.ceil(len(bytes_data) / width))
        
        pad_len = (width * height) - len(bytes_data)
        if pad_len > 0:
            bytes_data += b'\x00' * pad_len
            
        img_array = np.frombuffer(bytes_data, dtype=np.uint8).reshape((height, width))
        pil_img = Image.fromarray(img_array, 'L')
        
        # 2. Prepare for Model
        img_resized = pil_img.resize((128, 128))
        model_input = np.array(img_resized).astype(np.float32) / 255.0
        model_input = np.expand_dims(model_input, axis=(0, -1))
        
        # 3. Predict
        prediction = model.predict(model_input)
        prob = float(prediction[0][0])
        
        # 4. Generate Heatmap
        last_conv_name = get_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(model_input, model, last_conv_name)
        overlay_img = create_overlay(img_resized, heatmap)

    # --- Display Results ---
    st.divider()
    
    if prob > 0.5:
        st.error(f"### ⚠️ RANSOMWARE DETECTED (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"### ✅ BENIGN FILE (Confidence: {(1-prob)*100:.2f}%)")
        
    st.write(f"**Target Layer Analyzed:** `{last_conv_name}`")

    # Display Images in 3 Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_resized, caption="Raw Binary Image (128x128)", use_container_width=True)
    with col2:
        # Display standalone heatmap
        jet = cm.get_cmap("jet")
        heatmap_colored = jet(heatmap)[:, :, :3]
        st.image(heatmap_colored, caption="Grad-CAM Heatmap", use_container_width=True)
    with col3:
        st.image(overlay_img, caption="Feature Overlay", use_container_width=True)
