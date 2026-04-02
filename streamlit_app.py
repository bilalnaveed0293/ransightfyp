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
from collections import Counter

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
    
    # --- 🛠️ INTERNAL FORENSIC ENGINE ---
    def calculate_local_entropy(data, offset, window_size=256):
        start = max(0, offset - (window_size // 2))
        end = min(len(data), offset + (window_size // 2))
        chunk = data[start:end]
        if not chunk: return 0
        counts = Counter(chunk)
        probs = [c / len(chunk) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

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
        for layer in keras_model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x

        grad_model = tf.keras.models.Model(inputs=grad_model_input, outputs=[conv_output, x])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            class_channel = preds[:, 0]

        grads = tape.gradient(class_channel, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def create_overlay(pil_img, heatmap):
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((128, 128), resample=Image.BICUBIC)
        jet = cm.get_cmap("jet")
        jet_heatmap = (jet(np.array(heatmap_img))[:, :, :3] * 255).astype(np.uint8)
        img_rgb = np.stack((np.array(pil_img),)*3, axis=-1)
        overlay = (jet_heatmap * 0.4 + img_rgb * 0.6).astype(np.uint8)
        return Image.fromarray(overlay)

    # --- 📥 FILE PROCESSING ---
    uploaded_static = st.file_uploader("Upload .exe for Static Visual Analysis", type=["exe"], key="u1")

    if uploaded_static:
        file_bytes = uploaded_static.read()
        file_size = len(file_bytes)
        width = get_standard_width(file_size)
        height = int(np.ceil(file_size / width))
        
        # Convert to Image for CNN
        img_raw = np.frombuffer(file_bytes, dtype=np.uint8)
        img_raw = np.pad(img_raw, (0, (width * height) - file_size))
        pil_img = Image.fromarray(img_raw.reshape((height, width)), 'L').resize((128, 128))
        
        input_arr = np.array(pil_img).astype('float32') / 255.0
        input_arr = np.expand_dims(input_arr, axis=(0, -1))
        
        # Run Prediction
        prediction = cnn_model.predict(input_arr)
        prob = float(prediction[0][0])
        verdict = "RANSOMWARE" if prob > 0.5 else "BENIGN"
        conf = prob if prob > 0.5 else (1 - prob)

        # Generate Heatmap and find Focal Point
        heatmap = make_gradcam_heatmap(input_arr, cnn_model)
        max_idx = np.argmax(heatmap)
        y_128, x_128 = divmod(max_idx, 128)
        
        # Map back to raw file offset
        y_orig = int((y_128 / 128.0) * height)
        x_orig = int((x_128 / 128.0) * width)
        center_offset = min((y_orig * width) + x_orig, file_size - 1)

        # --- 📊 UI LAYOUT ---
        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric("Static Verdict", verdict, f"{conf*100:.1f}% Confidence")
            overlay = create_overlay(pil_img, heatmap)
            st.image(overlay, caption="Grad-CAM: CNN Focus Areas (Red = Highest Activation)", use_container_width=True)

        with col2:
            st.subheader("📝 Forensic Explainability Report")
            
            # Calculate Evidence
            entropy = calculate_local_entropy(file_bytes, center_offset)
            
            # Determine Section/Feature based on Offset
            if center_offset < 0x200:
                feature_type = "PE Header / DOS Stub"
                insight = "Anomaly detected in the file's entry structure. Suggests header manipulation or a custom loader."
            elif entropy > 7.0:
                feature_type = "Encrypted Payload Section"
                insight = "High-entropy patterns detected. This visual 'texture' is a mathematical hallmark of ransomware encryption."
            else:
                feature_type = "Standard Code/Data Block"
                insight = "Pattern matches typical non-malicious execution blocks or resource storage."

            # Render the Report
            st.markdown(f"""
            **Analysis Target:** Offset `0x{center_offset:X}`  
            **Identified Feature:** `{feature_type}`  
            **Local Shannon Entropy:** `{entropy:.2f}`
            
            ---
            **Technical Basis for Verdict:** {insight}
            
            **Forensic Summary:** The CNN model's decision was heavily weighted by a specific spatial pattern at this offset. 
            {"The high entropy value confirms that this region contains packed or encrypted data, supporting a Ransomware classification." if verdict == "RANSOMWARE" else "The structural patterns and entropy levels are consistent with benign software standards."}
            """)

            # Hex Dump for visual proof
            st.text("Raw Bytes at Hotspot:")
            start_b = max(0, center_offset - 64)
            end_b = min(file_size, center_offset + 64)
            hex_view = ' '.join([f'{b:02X}' for b in file_bytes[start_b:end_b]])
            st.caption(f"Hex: {hex_view[:100]}...")
with tab2:
            st.header("Dynamic Analysis: Triage Sandbox & LIME")
            st.markdown("Standardized **60-second execution**. Aggressively extracting APIs, DLLs, and Mutexes.")

            uploaded_dynamic = st.file_uploader("Upload .exe for Sandbox Analysis", type=["exe"], key="dyn_up")

            if uploaded_dynamic and st.button("Start Full Analysis"):
                import streamlit.components.v1 as components
                import json
                import time
                import requests
                import numpy as np
                from lime.lime_text import LimeTextExplainer

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

                # ── STEP 2: TRIGGER AUTO PROFILE ────────────────────────────────────────
                with st.spinner("Selecting analysis profile..."):
                    profile_res = requests.post(
                        f"{BASE_URL}/samples/{sample_id}/profile",
                        headers={**HEADERS, "Content-Type": "application/json"},
                        data=json.dumps({"auto": True})
                    )
                    if profile_res.status_code not in (200, 201, 409):
                        st.warning(f"Profile returned {profile_res.status_code}: {profile_res.text}")
                    else:
                        st.info("✅ Profile selected — behavioral analysis starting...")

                # ── STEP 3: POLL UNTIL REPORTED (up to 5 min) ──────────────────────────
                bar         = st.progress(0)
                status_text = st.empty()
                MAX_ITER    = 60
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
                    st.error("Triage analysis failed. The sample may be corrupted or unsupported.")
                    st.stop()
                if curr_status != "reported":
                    st.warning("Triage timed out.")
                    st.stop()

                # ── STEP 4: GET FULL TASK IDs FROM SAMPLE RESPONSE ──────────────────────
                sample_info = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS).json()
                tasks_raw   = sample_info.get("tasks", {})

                behavioral_tasks = []
                if isinstance(tasks_raw, dict):
                    for full_tid, tinfo in tasks_raw.items():
                        if tinfo.get("kind") == "behavioral":
                            behavioral_tasks.append(full_tid)
                elif isinstance(tasks_raw, list):
                    for t in tasks_raw:
                        tid = t.get("id", "")
                        if tid.startswith("behavioral"):
                            full_tid = f"{sample_id}-{tid}" if "-" not in tid else tid
                            behavioral_tasks.append(full_tid)

                if not behavioral_tasks:
                    st.error(f"No behavioral tasks found. Raw tasks: {tasks_raw}")
                    st.stop()

                st.info(f"Behavioral tasks found: {behavioral_tasks}")

                # ── STEP 5: FETCH onemon.json USING FULL TASK ID ────────────────────────
                raw_apis    = []
                raw_dlls    = []
                raw_mutexes = []
                successful_task = None

                for full_task_id in behavioral_tasks:
                    st.info(f"Fetching onemon.json for task `{full_task_id}`...")

                    for _ in range(20):
                        task_chk = requests.get(
                            f"{BASE_URL}/samples/{sample_id}/{full_task_id}",
                            headers=HEADERS
                        )
                        if task_chk.status_code == 200:
                            if task_chk.json().get("status") == "reported":
                                break
                        time.sleep(5)

                    onemon_res = requests.get(
                        f"{BASE_URL}/samples/{sample_id}/{full_task_id}/logs/onemon.json",
                        headers=HEADERS,
                        stream=True
                    )

                    if onemon_res.status_code != 200:
                        short_id = full_task_id.split("-")[-1]
                        onemon_res = requests.get(
                            f"{BASE_URL}/samples/{sample_id}/{short_id}/logs/onemon.json",
                            headers=HEADERS,
                            stream=True
                        )
                        if onemon_res.status_code != 200:
                            st.warning(f"onemon.json returned {onemon_res.status_code} for `{full_task_id}` — trying next...")
                            continue

                    # Parse NDJSON
                    task_apis    = []
                    task_dlls    = []
                    task_mutexes = []

                    for line in onemon_res.iter_lines():
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        kind = event.get("kind", "")
                        evt  = event.get("event", {})

                        # 1. API Calls & Kernel Syscalls
                        if kind == "onemon.Call" or kind.startswith("onemon.Syscall"):
                            api_name = evt.get("api") or evt.get("symbol") or evt.get("name") or evt.get("sys_name")
                            
                            if not api_name and "kind" in evt:
                                api_name = f"Syscall_{evt['kind']}"
                            elif not api_name and "sys" in evt:
                                api_name = f"Syscall_{evt['sys']}"

                            if api_name:
                                task_apis.append(str(api_name))

                        # 1.5 Bonus APIs (High-Level Actions)
                        action_name = evt.get("action")
                        if action_name:
                            task_apis.append(str(action_name))

                        # 2. DLL loads (Aggressive string searching across all events)
                        for key in ["path", "filepath", "image", "arg0", "arg1", "name"]:
                            val = evt.get(key, "")
                            if isinstance(val, str) and ".dll" in val.lower():
                                dll_name = val.replace("\\", "/").split("/")[-1]
                                if dll_name.lower().endswith(".dll") and dll_name not in task_dlls:
                                    task_dlls.append(dll_name)

                        # 3. Mutexes (Mutants)
                        if kind == "onemon.Mutant" or (kind == "onemon.Handle" and evt.get("type") == "Mutant"):
                            name = evt.get("name") or evt.get("path") or evt.get("mutant") or evt.get("obj")
                            if name and name not in task_mutexes:
                                task_mutexes.append(str(name))

                    if task_apis or task_dlls or task_mutexes:
                        raw_apis    = task_apis
                        raw_dlls    = task_dlls
                        raw_mutexes = task_mutexes
                        successful_task = full_task_id
                        st.success(f"✅ Got behavioral data from task: `{full_task_id}`")
                        break
                    else:
                        st.warning(f"No API/DLL/Mutex data extracted in `{full_task_id}` — trying next task...")

                # ── STEP 6: SEQUENCE PREPARATION ────────────────────────────────────────
                with st.expander("🔍 Extraction Summary"):
                    st.write(f"Successful task: `{successful_task}`")
                    st.write(f"APIs: **{len(raw_apis)}** | DLLs: **{len(raw_dlls)}** | Mutexes: **{len(raw_mutexes)}**")
                    st.write("Sample APIs:",    raw_apis[:15])
                    st.write("Sample DLLs:",    raw_dlls[:10])
                    st.write("Sample Mutexes:", raw_mutexes[:10])

                # Combine everything into a single space-separated string sequence
                final_sequence = " ".join(raw_apis[:500] + raw_dlls[:10] + raw_mutexes[:10])

                if not final_sequence.strip():
                    st.error("No behavioral data found to feed the model.")
                    st.stop()

                # ── STEP 7: LSTM PREDICTION ─────────────────────────────────────────────
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

                # ── STEP 8: LIME EXPLANATION ────────────────────────────────────────────
                with st.spinner("Generating LIME explanation..."):
                    explainer = LimeTextExplainer(class_names=["Benign", "Ransomware"])
                    exp = explainer.explain_instance(
                        final_sequence, predict_proba, num_features=10
                    )
                    st.write("### 🧠 Top Features (APIs / DLLs / Mutexes)")
                    components.html(exp.as_html(), height=600, scrolling=True)
