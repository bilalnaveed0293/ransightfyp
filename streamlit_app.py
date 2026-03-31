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
import streamlit.components.v1 as components

# --- 1. Page Config & Styling ---
st.set_page_config(page_title="RanSight AI", page_icon="🛡️", layout="wide")
st.title("🛡️ RanSight: Hybrid Ransomware Detection & XAI")
st.markdown("---")

# --- 2. Initialize Clients & Models ---
@st.cache_resource
def load_resources():
    cnn = tf.keras.models.load_model("cnn1.keras")
    lstm = tf.keras.models.load_model("ransomware_lstm_dynamic.keras")
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return cnn, lstm, tok, client

try:
    cnn_model, lstm_model, tokenizer, groq_client = load_resources()
    st.sidebar.success("✅ System Ready: Models & Tokenizer Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Initialization Error: {e}")
    st.stop()

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
        import streamlit.components.v1 as components
        import json

        API_KEY = st.secrets["TRIAGE_API_KEY"]
        HEADERS = {"Authorization": f"Bearer {API_KEY}"}
        BASE_URL = "https://api.tria.ge/v0"

        # ── STEP 1: SUBMIT ──────────────────────────────────────────────────────
        with st.spinner("Uploading to Triage sandbox..."):
            files = {"file": (uploaded_dynamic.name, uploaded_dynamic.getvalue())}
            data  = {"_json": '{"kind":"file","interactive":false}'}

            res = requests.post(f"{BASE_URL}/samples", headers=HEADERS, files=files, data=data)
            if res.status_code not in (200, 201):
                st.error(f"Submission failed ({res.status_code}): {res.text}")
                st.stop()

            sample_id = res.json().get("id")
            if not sample_id:
                st.error(f"No sample ID in response: {res.json()}")
                st.stop()

            st.info(f"✅ Submitted — Sample ID: `{sample_id}`")

        # ── STEP 2: POLL UNTIL REPORTED (up to 5 min) ──────────────────────────
        bar         = st.progress(0)
        status_text = st.empty()
        MAX_ITER    = 60   # 60 × 5s = 300s
        curr_status = "pending"

        for i in range(MAX_ITER):
            time.sleep(5)
            chk = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS)
            if chk.status_code != 200:
                continue
            curr_status = chk.json().get("status", "unknown")
            bar.progress(int(min((i + 1) / MAX_ITER * 100, 100)))
            status_text.text(f"Status: {curr_status}  ({(i+1)*5}s / {MAX_ITER*5}s)")
            if curr_status in ("reported", "failed"):
                break

        if curr_status == "failed":
            st.error("Triage analysis failed — sample may have crashed or been rejected.")
            st.stop()
        if curr_status != "reported":
            st.warning("Triage timed out. Check the sample manually on tria.ge.")
            st.stop()

        # ── STEP 3: RESOLVE TASK ID ─────────────────────────────────────────────
        sample_info = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS).json()
        tasks_raw   = sample_info.get("tasks", {})

        # tasks is a DICT keyed by full task id e.g. "20240101-abc123-behavioral1"
        task_id = None
        if isinstance(tasks_raw, dict):
            for tid, tinfo in tasks_raw.items():
                if tinfo.get("kind") == "behavioral":
                    task_id = tid
                    break
        elif isinstance(tasks_raw, list):
            for t in tasks_raw:
                if t.get("kind") == "behavioral":
                    task_id = t.get("id")
                    break

        if not task_id:
            st.error("No behavioral task found in report.")
            st.stop()

        st.info(f"Task ID: `{task_id}`")

        # ── STEP 4: FETCH onemon.json (raw kernel event log) ────────────────────
        # This is where ALL api calls, dlls, mutexes actually live in Triage
        with st.spinner("Downloading behavioral event log (onemon.json)..."):
            onemon_res = requests.get(
                f"{BASE_URL}/samples/{sample_id}/{task_id}/logs/onemon.json",
                headers=HEADERS,
                stream=True
            )

        if onemon_res.status_code != 200:
            st.error(f"Failed to fetch onemon.json (HTTP {onemon_res.status_code}): {onemon_res.text}")
            st.stop()

        # onemon.json is NDJSON — one JSON object per line
        raw_apis    = []
        raw_dlls    = []
        raw_mutexes = []

        for line in onemon_res.iter_lines():
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = event.get("kind", "")

            # ── API CALLS ──────────────────────────────────────────────────────
            # kind == "call" → event has "call" field with the API name
            if kind == "call":
                api_name = event.get("call")
                if api_name:
                    raw_apis.append(api_name)

            # ── LOADED DLLs ────────────────────────────────────────────────────
            # kind == "load_image" → event has "image" field with the DLL path
            elif kind == "load_image":
                image_path = event.get("image", "")
                if image_path:
                    # Keep only the filename e.g. "kernel32.dll"
                    dll_name = image_path.split("\\")[-1].split("/")[-1]
                    if dll_name.lower().endswith(".dll"):
                        raw_dlls.append(dll_name)

            # ── MUTEXES ────────────────────────────────────────────────────────
            # kind == "CreateMutant" OR kind == "mutex" depending on onemon version
            elif kind in ("mutex", "CreateMutant") or (
                kind == "call" and event.get("call", "").lower() in ("createmutexw", "createmutexexw", "ntcreatemutant")
            ):
                # Try direct name field first
                mutex_name = event.get("name") or event.get("mutex")
                if not mutex_name:
                    # Fall back to args list
                    args = event.get("args", [])
                    if args:
                        mutex_name = args[0] if isinstance(args[0], str) else None
                if mutex_name:
                    raw_mutexes.append(mutex_name)

        # ── STEP 5: DEBUG ───────────────────────────────────────────────────────
        with st.expander("🔍 Extraction Debug"):
            st.write(f"APIs extracted: **{len(raw_apis)}** (using first 500)")
            st.write(f"DLLs extracted: **{len(raw_dlls)}** (using first 10)")
            st.write(f"Mutexes extracted: **{len(raw_mutexes)}** (using first 10)")
            st.write("Sample APIs:",    raw_apis[:15])
            st.write("Sample DLLs:",    raw_dlls[:10])
            st.write("Sample Mutexes:", raw_mutexes[:10])

        # Build sequence matching Xran training format exactly
        final_sequence = " ".join(raw_apis[:500] + raw_dlls[:10] + raw_mutexes[:10])

        if not final_sequence.strip():
            st.error("Sequence is empty — the sample may not have executed or onemon had no events.")
            st.stop()

        # ── STEP 6: LSTM PREDICTION ─────────────────────────────────────────────
        def predict_proba(texts):
            seqs   = tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(seqs, maxlen=520, padding="post", truncating="post")
            preds  = lstm_model.predict(padded)
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

        # ── STEP 7: LIME EXPLANATION ────────────────────────────────────────────
        with st.spinner("Generating LIME explanation..."):
            explainer = LimeTextExplainer(class_names=["Benign", "Ransomware"])
            exp = explainer.explain_instance(
                final_sequence, predict_proba, num_features=10
            )
            st.write("### 🧠 Top Features (API calls / DLLs / Mutexes)")
            components.html(exp.as_html(), height=600, scrolling=True)
