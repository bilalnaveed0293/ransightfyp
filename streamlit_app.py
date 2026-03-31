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
with tab2:
    st.header("Dynamic Analysis: Triage Sandbox & LIME")
    st.markdown("Standardized **60-second execution**. Extracting 500 APIs, 10 DLLs, and 10 Mutexes.")

    uploaded_dynamic = st.file_uploader("Upload .exe for Sandbox Analysis", type=["exe"], key="dyn_up")

    if uploaded_dynamic and st.button("Start Full Analysis"):
        import streamlit.components.v1 as components  # ← was missing from tab2 scope

        API_KEY = st.secrets["TRIAGE_API_KEY"]
        HEADERS = {"Authorization": f"Bearer {API_KEY}"}
        BASE_URL = "https://api.tria.ge/v0"

       # ── STEP 1: SUBMIT ──────────────────────────────────────────────────────
with st.spinner("Uploading to Triage sandbox..."):
    files = {"file": (uploaded_dynamic.name, uploaded_dynamic.getvalue())}
    
    # FIX: Don't specify profiles — only company accounts can do that.
    # Let Triage auto-select the best environment (will still pick Windows for .exe)
    data = {"_json": '{"kind":"file","interactive":false}'}
    
    res = requests.post(f"{BASE_URL}/samples", headers=HEADERS, files=files, data=data)

    if res.status_code not in (200, 201):
        st.error(f"Submission failed ({res.status_code}): {res.text}")
        st.stop()

    sample_id = res.json().get("id")
    if not sample_id:
        st.error(f"No sample ID returned. Response: {res.json()}")
        st.stop()

    st.info(f"✅ Submitted. Sample ID: `{sample_id}` — waiting for execution + reporting...")

        # ── STEP 2: POLL SAMPLE STATUS (up to 5 minutes) ────────────────────────
        bar = st.progress(0)
        status_text = st.empty()
        MAX_WAIT = 60   # iterations × 5s = 300s
        curr_status = "pending"

        for i in range(MAX_WAIT):
            time.sleep(5)
            check = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS)
            if check.status_code != 200:
                continue
            curr_status = check.json().get("status", "unknown")
            bar.progress(int(min((i + 1) / MAX_WAIT * 100, 100)))
            status_text.text(f"Status: {curr_status}  ({(i+1)*5}s / {MAX_WAIT*5}s)")
            if curr_status in ("reported", "failed"):
                break

        if curr_status == "failed":
            st.error("Triage analysis failed. The sample may have crashed or been rejected.")
            st.stop()

        if curr_status != "reported":
            st.warning("Triage did not finish in time. Try fetching the report manually.")
            st.stop()

        # ── STEP 3: GET TASK ID ──────────────────────────────────────────────────
        # FIX: tasks is a DICT keyed by task_id, not a list
        sample_info = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS).json()
        tasks_dict = sample_info.get("tasks", {})

        if not tasks_dict:
            st.error("No behavioral tasks found in the report.")
            st.stop()

        # Pick the first behavioral task
        task_id = None
        for tid, tinfo in tasks_dict.items():
            if tinfo.get("kind") == "behavioral":
                task_id = tid
                break

        if not task_id:
            task_id = list(tasks_dict.keys())[0]  # fallback: first task

        st.info(f"Task ID: `{task_id}`")

        # ── STEP 4: WAIT FOR TASK-LEVEL REPORT ──────────────────────────────────
        with st.spinner("Finalizing behavioral logs..."):
            for _ in range(20):
                task_check = requests.get(
                    f"{BASE_URL}/samples/{sample_id}/{task_id}", headers=HEADERS
                )
                if task_check.status_code == 200:
                    if task_check.json().get("status") == "reported":
                        break
                time.sleep(5)

        # ── STEP 5: FETCH BEHAVIORAL JSON REPORT ────────────────────────────────
        report_res = requests.get(
            f"{BASE_URL}/samples/{sample_id}/{task_id}/report_triage.json",
            headers=HEADERS,
        )

        if report_res.status_code != 200:
            st.error(f"Failed to retrieve report (HTTP {report_res.status_code}): {report_res.text}")
            st.stop()

        report_data = report_res.json()

        # ── STEP 6: FEATURE EXTRACTION (matching Xran dataset format) ───────────
        raw_apis    = []
        raw_dlls    = []
        raw_mutexes = []

        for proc in report_data.get("processes", []):

            # 500 API calls — field: calls[].api
            for call in proc.get("calls", []):
                api_name = call.get("api")
                if api_name:
                    raw_apis.append(api_name)

            # 10 DLLs — field: modules[].basename  (NOT loaded_modules)
            for mod in proc.get("modules", []):
                if isinstance(mod, str):
                    raw_dlls.append(mod)
                elif isinstance(mod, dict):
                    name = mod.get("basename") or mod.get("path", "")
                    if name:
                        raw_dlls.append(name.split("\\")[-1])  # keep filename only

            # 10 Mutexes — field: mutants[].name  (NOT mutexes, NOT inside calls)
            for mutant in proc.get("mutants", []):
                if isinstance(mutant, str):
                    raw_mutexes.append(mutant)
                elif isinstance(mutant, dict):
                    name = mutant.get("name", "")
                    if name:
                        raw_mutexes.append(name)

        # Debug expander — helps verify extraction is working
        with st.expander("🔍 Raw Extraction Debug"):
            st.write(f"APIs extracted: {len(raw_apis)} (using first 500)")
            st.write(f"DLLs extracted: {len(raw_dlls)} (using first 10)")
            st.write(f"Mutexes extracted: {len(raw_mutexes)} (using first 10)")
            st.write("Sample APIs:", raw_apis[:10])
            st.write("Sample DLLs:", raw_dlls[:10])
            st.write("Sample Mutexes:", raw_mutexes[:10])

        # Build sequence exactly matching Xran training format
        final_sequence = " ".join(raw_apis[:500] + raw_dlls[:10] + raw_mutexes[:10])

        if not final_sequence.strip():
            st.error("Extracted sequence is empty — the malware may not have executed in the sandbox.")
            st.stop()

        # ── STEP 7: LSTM PREDICTION ──────────────────────────────────────────────
        def predict_proba(texts):
            seqs   = tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(seqs, maxlen=520, padding="post", truncating="post")
            preds  = lstm_model.predict(padded)
            # Handle both binary (shape [N,1]) and softmax (shape [N,2]) outputs
            if preds.shape[1] == 1:
                return np.hstack([1 - preds, preds])
            return preds

        probs = predict_proba([final_sequence])[0]

        st.divider()
        if probs[1] > 0.5:
            st.error(f"🔥 VERDICT: RANSOMWARE — {probs[1]*100:.2f}% confidence")
        else:
            st.success(f"🛡️ VERDICT: BENIGN — {probs[0]*100:.2f}% confidence")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ransomware Probability", f"{probs[1]*100:.2f}%")
        with col2:
            st.metric("Benign Probability", f"{probs[0]*100:.2f}%")

        # ── STEP 8: LIME EXPLANATION ─────────────────────────────────────────────
        with st.spinner("Generating LIME explanation..."):
            explainer = LimeTextExplainer(class_names=["Benign", "Ransomware"])
            exp = explainer.explain_instance(
                final_sequence, predict_proba, num_features=10
            )
            st.write("### 🧠 Top Features (API calls / DLLs / Mutexes)")
            components.html(exp.as_html(), height=600, scrolling=True)
