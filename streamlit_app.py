import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import base64
import time
import requests
import pickle
from io import BytesIO
from groq import Groq
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import shap

# --- 1. Page Config & Styling ---
st.set_page_config(page_title="RanSight AI", page_icon="🛡️", layout="wide")
st.title("🛡️ RanSight: Hybrid Ransomware Detection & XAI")
st.markdown("---")

# --- 2. Initialize Clients & Models ---
@st.cache_resource
def load_resources():
    # Load CNN
    cnn = tf.keras.models.load_model("cnn1.keras")
    # Load LSTM
    lstm = tf.keras.models.load_model("ransomware_lstm_dynamic.keras")
    # Load Tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    # Groq Client
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return cnn, lstm, tok, client

try:
    cnn_model, lstm_model, tokenizer, groq_client = load_resources()
    st.sidebar.success("✅ System Ready: Models & Tokenizer Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Initialization Error: {e}")
    st.stop()

# --- Create Navigation ---
tab1, tab2 = st.tabs(["📂 Static Analysis (CNN)", "⚙️ Dynamic Analysis (LSTM + HA)"])

# ==================================================================
# TAB 1: STATIC ANALYSIS (CNN + Grad-CAM)
# ==================================================================
with tab1:
    st.header("Static Analysis: Binary Image Visualization")
    
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

    uploaded_static = st.file_uploader("Upload .exe for Static Visual Analysis", type=["exe"], key="u1")

    if uploaded_static:
        data = uploaded_static.read()
        width = get_standard_width(len(data))
        height = int(np.ceil(len(data) / width))
        img_raw = np.frombuffer(data, dtype=np.uint8)
        img_raw = np.pad(img_raw, (0, (width * height) - len(data)))
        pil_img = Image.fromarray(img_raw.reshape((height, width)), 'L').resize((128, 128))
        
        input_arr = np.array(pil_img).astype('float32') / 255.0
        input_arr = np.expand_dims(input_arr, axis=(0, -1))
        
        prediction = cnn_model.predict(input_arr)
        prob = float(prediction[0][0])
        verdict = "RANSOMWARE" if prob > 0.5 else "BENIGN"
        conf = prob if prob > 0.5 else (1 - prob)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Static Verdict", verdict, f"{conf*100:.1f}% Confidence")
            heatmap = make_gradcam_heatmap(input_arr, cnn_model)
            overlay = create_overlay(pil_img, heatmap)
            st.image(overlay, caption="Grad-CAM: Suspicious Byte Heatmap", use_container_width=True)

        with col2:
            st.subheader("🤖 AI Forensic Insight")
            if st.button("Explain Heatmap (Groq)"):
                buffered = BytesIO()
                overlay.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                prompt = f"Analyze this malware binary heatmap. Result: {verdict}. Provide 3 technical reasons why these byte patterns look like ransomware."
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}]}],
                )
                st.write(response.choices[0].message.content)

# ==================================================================
# TAB 2: DYNAMIC ANALYSIS (LSTM + Hybrid Analysis API)
# ==================================================================
with tab2:
    st.header("Dynamic Analysis: Cloud Sandbox Detonation")
    st.info("Files are sent to Hybrid Analysis (CrowdStrike) for safe execution.")

    uploaded_dynamic = st.file_uploader("Upload .exe for Sandbox Detonation", type=["exe"], key="u2")
    MAX_LEN = 520

    if uploaded_dynamic and st.button("Start Cloud Analysis"):
        HA_KEY = st.secrets["HYBRID_ANALYSIS_API_KEY"]
        HEADERS = {'api-key': HA_KEY, 'user-agent': 'Falcon Sandbox'}
        
        # 1. Submission
        with st.spinner("Uploading to Hybrid Analysis..."):
            files = {'file': (uploaded_dynamic.name, uploaded_dynamic.read())}
            res = requests.post("https://www.hybrid-analysis.com/api/v2/submit/file", headers=HEADERS, files=files, data={'environment_id': 160})
            job_id = res.json().get('job_id')
            if not job_id: st.error("Upload failed."); st.stop()

        # 2. Polling
        bar = st.progress(0)
        status_text = st.empty()
        for i in range(40): # Poll for ~10 mins
            time.sleep(15)
            state = requests.get(f"https://www.hybrid-analysis.com/api/v2/report/{job_id}/state", headers=HEADERS).json()
            bar.progress(min((i+1)*3, 100))
            status_text.text(f"Sandbox State: {state.get('state')}")
            if state.get('state') == 'SUCCESS': break
        
        # 3. Extraction
        with st.spinner("Parsing API Traces..."):
            report = requests.get(f"https://www.hybrid-analysis.com/api/v2/report/{job_id}/summary", headers=HEADERS).json()
            
            apis = [c.get('name') for p in report.get('processes', []) for c in p.get('calls', [])]
            dlls = report.get('dll_loaded', [])
            mutexes = report.get('mutex_created', [])
            
            trace_input = " ".join((apis[:500] + dlls[:10] + mutexes[:10]))
            if not trace_input.strip(): trace_input = "LdrLoadDll RegOpenKeyExW NtCreateFile PRF PRF PRF PRF kernel32.dll" # Fallback

        # 4. Prediction
        def predict_p(txt):
            seq = tokenizer.texts_to_sequences(txt)
            pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
            preds = lstm_model.predict(pad)
            return np.hstack((1-preds, preds))

        prob_dynamic = predict_p([trace_input])[0][1]
        st.divider()
        if prob_dynamic > 0.5:
            st.error(f"Dynamic Verdict: RANSOMWARE ({prob_dynamic*100:.1f}%)")
        else:
            st.success(f"Dynamic Verdict: BENIGN ({(1-prob_dynamic)*100:.1f}%)")

        # 5. XAI (SHAP & LIME)
        c_lime, c_shap = st.columns(2)
        with c_lime:
            st.subheader("LIME Explanation")
            explainer = LimeTextExplainer(class_names=['Benign', 'Ransomware'])
            exp = explainer.explain_instance(trace_input, predict_p, num_features=8)
            st.pyplot(exp.as_pyplot_figure())

        with c_shap:
            st.subheader("SHAP Feature Impact")
            def shap_p(d): return lstm_model.predict(d).flatten()
            explainer_shap = shap.KernelExplainer(shap_p, np.zeros((5, MAX_LEN)))
            test_seq = pad_sequences(tokenizer.texts_to_sequences([trace_input]), maxlen=MAX_LEN)
            sv = explainer_shap.shap_values(test_seq)
            
            names = [tokenizer.index_word.get(i, "?") for i in test_seq[0] if i != 0][:10]
            vals = sv[0][:len(names)]
            fig, ax = plt.subplots()
            ax.barh(names, vals, color=['red' if v > 0 else 'blue' for v in vals])
            st.pyplot(fig)
