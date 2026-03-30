import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
import base64
from io import BytesIO
from groq import Groq

# --- 1. Page Config & Styling ---
st.set_page_config(page_title="Ransight AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Ransight: CNN Static Analysis + AI Forensics")
st.markdown("---")

# --- 2. Initialize Clients & Models ---
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("cnn1.keras")
    # Groq Key from Streamlit Secrets
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return model, client

model, groq_client = load_resources()

# --- 3. Core Logic: Grad-CAM & Processing ---
def get_standard_width(file_size):
    if file_size < 10240: return 32
    elif file_size < 30720: return 64
    elif file_size < 61440: return 128
    elif file_size < 102400: return 256
    elif file_size < 204800: return 384
    elif file_size < 512000: return 512
    elif file_size < 1024000: return 768
    else: return 1024

def make_gradcam_heatmap(img_array, keras_model):
    # Find last conv layer automatically
    last_conv_layer_name = None
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    grad_model_input = tf.keras.Input(shape=(128, 128, 1))
    x = grad_model_input
    last_conv_output = None
    for layer in keras_model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x

    grad_model = tf.keras.models.Model(inputs=grad_model_input, outputs=[last_conv_output, x])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_overlay(pil_img, heatmap):
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((128, 128), resample=Image.BICUBIC)
    jet = cm.get_cmap("jet")
    jet_heatmap = (jet(np.array(heatmap_img))[:, :, :3] * 255).astype(np.uint8)
    img_rgb = np.stack((np.array(pil_img),)*3, axis=-1)
    overlay = (jet_heatmap * 0.4 + img_rgb * 0.6).astype(np.uint8)
    return Image.fromarray(overlay)

# --- 4. Groq AI Integration ---
def get_ai_explanation(overlay_image, verdict, confidence):
    # 1. Ensure the image is in RGB mode (required for PNG/JPEG conversion)
    if overlay_image.mode != 'RGB':
        overlay_image = overlay_image.convert('RGB')
        
    # 2. Encode to PNG
    buffered = BytesIO()
    overlay_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 3. Use the 90B model (usually more robust for vision tasks)
    # Note: Ensure the model ID is exactly as supported by Groq
    MODEL_ID = "llama-3.2-90b-vision-preview" 

    prompt = f"""Analyze this Grad-CAM heatmap of an executable file. 
    The CNN model classified it as {verdict} with {confidence:.2f}% confidence.
    Red regions are high-importance byte clusters. 
    Explain in 3 technical bullet points what these patterns usually represent in ransomware (e.g., PE headers, encrypted overlays, or resource sections)."""

    try:
        chat_completion = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}" # Explicitly match PNG here
                        }
                    }
                ]
            }],
            temperature=0.2, # Lower temperature for more factual analysis
            max_tokens=512
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Analysis Error: {str(e)}"
# --- 5. Main UI ---
uploaded_file = st.file_uploader("Upload suspicious .exe file", type=["exe"])

if uploaded_file:
    with st.spinner("Converting binary to image and analyzing..."):
        # Process file
        data = uploaded_file.read()
        width = get_standard_width(len(data))
        height = int(np.ceil(len(data) / width))
        img_raw = np.frombuffer(data, dtype=np.uint8)
        img_raw = np.pad(img_raw, (0, (width * height) - len(data)))
        pil_img = Image.fromarray(img_raw.reshape((height, width)), 'L').resize((128, 128))
        
        # Predict
        input_arr = np.array(pil_img).astype('float32') / 255.0
        input_arr = np.expand_dims(input_arr, axis=(0, -1))
        prediction = model.predict(input_arr)
        prob = float(prediction[0][0])
        verdict = "RANSOMWARE" if prob > 0.5 else "BENIGN"
        conf = prob if prob > 0.5 else (1 - prob)

        # Heatmap
        heatmap = make_gradcam_heatmap(input_arr, model)
        overlay = create_overlay(pil_img, heatmap)

    # UI Display
    col1, col2 = st.columns([1, 1])
    with col1:
        if verdict == "RANSOMWARE":
            st.error(f"### Result: {verdict} ({conf*100:.1f}%)")
        else:
            st.success(f"### Result: {verdict} ({conf*100:.1f}%)")
        st.image(overlay, caption="Grad-CAM Focus Overlay", use_container_width=True)

    with col2:
        st.subheader("🤖 AI Forensic Report")
        if st.button("Generate AI Explanation"):
            with st.spinner("Groq is decoding patterns..."):
                report = get_ai_explanation(overlay, verdict, conf*100)
                st.markdown(report)
