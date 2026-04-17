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
import pefile
from io import BytesIO
from groq import Groq
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from collections import Counter
import math
import joblib
import pandas as pd
import json
from typing import Dict, List, Optional

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
    
    # Load Stage 3 Memory Models
    rf_mem = joblib.load("malmem_rf_memory_model.joblib")
    try:
        mem_scaler = joblib.load("malmem_memory_scaler.joblib")
    except Exception:
        mem_scaler = None
        
    return cnn, lstm, tok, client, rf_mem, mem_scaler

try:
    cnn_model, lstm_model, tokenizer, groq_client, rf_mem, mem_scaler = load_resources()
    st.sidebar.success("✅ System Ready: All Models & Tokenizer Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Initialization Error: {e}")
    st.stop()

# --- 🛠️ GLOBAL FORENSIC HELPERS ---
def calculate_local_entropy(data, offset, window_size=256):
    start = max(0, offset - (window_size // 2))
    end = min(len(data), offset + (window_size // 2))
    chunk = data[start:end]
    if not chunk: return 0
    counts = Counter(chunk)
    probs = [c / len(chunk) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def get_exact_section(file_bytes, offset):
    try:
        pe = pefile.PE(data=file_bytes)
        for section in pe.sections:
            start = section.PointerToRawData
            end = start + section.SizeOfRawData
            if start <= offset < end:
                return section.Name.decode('utf-8', errors='ignore').strip('\x00')
        if pe.sections and offset < pe.sections[0].PointerToRawData:
            return "PE Headers"
        return "File Overlay / Appended Data"
    except Exception:
        return "Headers" if offset < 0x200 else "Unknown Section"

def hex_dump(data_bytes, start_offset):
    lines = []
    for i in range(0, len(data_bytes), 16):
        chunk = data_bytes[i:i+16]
        hex_str = ' '.join([f'{b:02X}' for b in chunk])
        ascii_str = ''.join([chr(b) if 32 <= b <= 126 else '.' for b in chunk])
        lines.append(f"0x{start_offset + i:06X}:  {hex_str:<48}  |{ascii_str}|")
    return '\n'.join(lines)

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

def predict_proba_lstm(texts):
    seqs   = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=520, padding="post", truncating="post")
    preds  = lstm_model.predict(padded)
    if preds.shape[1] == 1:
        return np.hstack([1 - preds, preds])
    return preds


# --- 3. TABS SETUP ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Static Analysis (CNN)", 
    "⚙️ Dynamic Analysis (LSTM + HA)", 
    "🚀 Full Gated Pipeline",
    "🧪 Memory Feature Extraction"
])

# ==================================================================
# TAB 1: STATIC ANALYSIS (CNN + Grad-CAM)
# ==================================================================
with tab1:
    st.header("Static Analysis: Binary Image Visualization")

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

        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric("Static Verdict", verdict, f"{conf*100:.1f}% Confidence")
            overlay = create_overlay(pil_img, heatmap)
            st.image(overlay, caption="Grad-CAM: CNN Focus Areas", use_container_width=True)

        with col2:
            st.subheader("📝 Forensic Explainability Report")
            entropy = calculate_local_entropy(file_bytes, center_offset)
            section_name = get_exact_section(file_bytes, center_offset)

            if "Header" in section_name:
                feature_type = f"Structural ({section_name})"
                insight = "The model is focusing on the file's metadata and entry headers. This suggests it detected anomalous structural transitions or header tampering."
            elif entropy > 7.2:
                feature_type = f"Cryptographic / Packed ({section_name})"
                insight = f"The model targeted the '{section_name}' section. Because the local entropy is a very high {entropy:.2f}, this is mathematical proof that this section contains encrypted shellcode or a compressed ransomware payload."
            else:
                feature_type = f"Standard Data ({section_name})"
                insight = f"The model focused on standard data patterns within the '{section_name}' section. With an entropy of {entropy:.2f}, this area does not display the chaotic randomness typically associated with active ransomware."

            st.markdown(f"""
            **Analysis Target:** Offset `0x{center_offset:X}`  
            **Identified Feature:** `{feature_type}`  
            **Local Shannon Entropy:** `{entropy:.2f}`
            
            ---
            **Technical Basis for Verdict:** {insight}
            """)

            st.write("**Raw Bytes at Hotspot Area:**")
            start_b = max(0, center_offset - 64)
            end_b = min(file_size, center_offset + 64)
            dump_output = hex_dump(file_bytes[start_b:end_b], start_b)
            st.text_area(f"Byte Offsets (0x{start_b:X} to 0x{end_b:X})", dump_output, height=250)


# ==================================================================
# TAB 2: DYNAMIC ANALYSIS (LSTM)
# ==================================================================
with tab2:
    st.header("Dynamic Analysis: Triage Sandbox & LIME")
    st.markdown("Standardized **60-second execution**. Aggressively extracting APIs, DLLs, and Mutexes.")

    uploaded_dynamic = st.file_uploader("Upload .exe for Sandbox Analysis", type=["exe"], key="dyn_up")

    if uploaded_dynamic and st.button("Start Full Analysis"):
        API_KEY = st.secrets["TRIAGE_API_KEY"]
        HEADERS = {"Authorization": f"Bearer {API_KEY}"}
        BASE_URL = "https://api.tria.ge/v0"

        with st.spinner("Uploading to Triage sandbox..."):
            files = {"file": (uploaded_dynamic.name, uploaded_dynamic.getvalue())}
            data  = {"_json": '{"kind":"file","interactive":false}'}
            res = requests.post(f"{BASE_URL}/samples", headers=HEADERS, files=files, data=data)
            
            if res.status_code not in (200, 201):
                st.error(f"Submission failed ({res.status_code}): {res.text}")
                st.stop()
            sample_id = res.json().get("id")
            st.info(f"✅ Submitted — Sample ID: `{sample_id}`")

        with st.spinner("Selecting analysis profile..."):
            requests.post(
                f"{BASE_URL}/samples/{sample_id}/profile",
                headers={**HEADERS, "Content-Type": "application/json"},
                data=json.dumps({"auto": True})
            )

        bar         = st.progress(0)
        status_text = st.empty()
        MAX_ITER    = 60
        curr_status = "pending"

        for i in range(MAX_ITER):
            time.sleep(5)
            chk = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS)
            if chk.status_code != 200: continue
            curr_status = chk.json().get("status", "unknown")
            bar.progress(int(min((i + 1) / MAX_ITER * 100, 100)))
            status_text.text(f"Status: {curr_status}  ({(i+1)*5}s / {MAX_ITER*5}s)")
            if curr_status in ("reported", "failed"): break

        if curr_status != "reported":
            st.error("Triage analysis failed or timed out.")
            st.stop()

        sample_info = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS).json()
        tasks_raw   = sample_info.get("tasks", {})
        behavioral_tasks = []
        
        if isinstance(tasks_raw, dict):
            for full_tid, tinfo in tasks_raw.items():
                if tinfo.get("kind") == "behavioral": behavioral_tasks.append(full_tid)
        elif isinstance(tasks_raw, list):
            for t in tasks_raw:
                tid = t.get("id", "")
                if tid.startswith("behavioral"):
                    behavioral_tasks.append(f"{sample_id}-{tid}" if "-" not in tid else tid)

        raw_apis, raw_dlls, raw_mutexes = [], [], []
        successful_task = None

        for full_task_id in behavioral_tasks:
            st.info(f"Fetching onemon.json for task `{full_task_id}`...")
            for _ in range(20):
                task_chk = requests.get(f"{BASE_URL}/samples/{sample_id}/{full_task_id}", headers=HEADERS)
                if task_chk.status_code == 200 and task_chk.json().get("status") == "reported": break
                time.sleep(5)

            onemon_res = requests.get(f"{BASE_URL}/samples/{sample_id}/{full_task_id}/logs/onemon.json", headers=HEADERS, stream=True)
            if onemon_res.status_code != 200:
                short_id = full_task_id.split("-")[-1]
                onemon_res = requests.get(f"{BASE_URL}/samples/{sample_id}/{short_id}/logs/onemon.json", headers=HEADERS, stream=True)
                if onemon_res.status_code != 200: continue

            task_apis    = []
            task_dlls    = []
            task_mutexes = []
            
            for line in onemon_res.iter_lines():
                if not line: continue
                try: event = json.loads(line)
                except: continue

                kind, evt = event.get("kind", ""), event.get("event", {})

                if kind == "onemon.Call" or kind.startswith("onemon.Syscall"):
                    api_name = evt.get("api") or evt.get("symbol") or evt.get("name") or evt.get("sys_name")
                    if not api_name and "kind" in evt: api_name = f"Syscall_{evt['kind']}"
                    elif not api_name and "sys" in evt: api_name = f"Syscall_{evt['sys']}"
                    if api_name: task_apis.append(str(api_name))

                action_name = evt.get("action")
                if action_name: task_apis.append(str(action_name))

                for key in ["path", "filepath", "image", "arg0", "arg1", "name"]:
                    val = evt.get(key, "")
                    if isinstance(val, str) and ".dll" in val.lower():
                        dll_name = val.replace("\\", "/").split("/")[-1]
                        if dll_name.lower().endswith(".dll") and dll_name not in task_dlls:
                            task_dlls.append(dll_name)

                if kind == "onemon.Mutant" or (kind == "onemon.Handle" and evt.get("type") == "Mutant"):
                    name = evt.get("name") or evt.get("path") or evt.get("mutant") or evt.get("obj")
                    if name and name not in task_mutexes: task_mutexes.append(str(name))

            if task_apis or task_dlls or task_mutexes:
                raw_apis = task_apis
                raw_dlls = task_dlls
                raw_mutexes = task_mutexes
                successful_task = full_task_id
                st.success(f"✅ Got behavioral data from task: `{full_task_id}`")
                break

        final_sequence = " ".join(raw_apis[:500] + raw_dlls[:10] + raw_mutexes[:10])
        if not final_sequence.strip():
            st.error("No behavioral data found.")
            st.stop()

        with st.expander("🔍 Extraction Summary"):
            st.write(f"Successful task: `{successful_task}`")
            st.write(f"APIs: **{len(raw_apis)}** | DLLs: **{len(raw_dlls)}** | Mutexes: **{len(raw_mutexes)}**")

        probs = predict_proba_lstm([final_sequence])[0]
        st.divider()
        if probs[1] > 0.5: st.error(f"🔥 VERDICT: RANSOMWARE — {probs[1]*100:.2f}% confidence")
        else: st.success(f"🛡️ VERDICT: BENIGN — {probs[0]*100:.2f}% confidence")

        col1, col2 = st.columns(2)
        col1.metric("Ransomware Probability", f"{probs[1]*100:.2f}%")
        col2.metric("Benign Probability", f"{probs[0]*100:.2f}%")

        with st.spinner("Generating LIME explanation..."):
            explainer = LimeTextExplainer(class_names=["Benign", "Ransomware"])
            exp = explainer.explain_instance(final_sequence, predict_proba_lstm, num_features=10)
            st.write("### 🧠 Top Features (APIs / DLLs / Mutexes)")
            components.html(exp.as_html(), height=600, scrolling=True)

# ==================================================================
# TAB 3: FULL GATED PIPELINE
# ==================================================================
with tab3:
    st.header("🚀 Confidence-Gated Multi-Modal Pipeline")
    st.markdown("Executes stages sequentially. Reduces analysis depth on unambiguous samples and escalates difficult samples to Triage API and Memory Analysis.")
    
    col1, col2 = st.columns(2)
    theta_1 = col1.slider("Gate 1 (Static) Threshold (θ1)", 0.50, 0.99, 0.85, 0.01)
    theta_2 = col2.slider("Gate 2 (Dynamic) Threshold (θ2)", 0.50, 0.99, 0.85, 0.01)
    
    st.markdown("---")
    gated_exe = st.file_uploader("1️⃣ Upload .exe File (Required for Stages 1 & 2)", type=["exe"], key="gate_exe")
    gated_csv = st.file_uploader("2️⃣ Upload Memory Features CSV (Required if escalated to Stage 3)", type=["csv"], key="gate_csv")

    if st.button("▶ Run Gated Pipeline") and gated_exe:
        # We read the file once here for Stage 1 (this moves the internal pointer to the end)
        file_bytes = gated_exe.read()
        
        # ---------------------------------------------------------
        # STAGE 1: STATIC ANALYSIS
        # ---------------------------------------------------------
        st.subheader("🟢 STAGE 1: Static Analysis (Custom CNN V2)")
        file_size = len(file_bytes)
        width = get_standard_width(file_size)
        height = int(np.ceil(file_size / width))

        img_raw = np.frombuffer(file_bytes, dtype=np.uint8)
        img_raw = np.pad(img_raw, (0, (width * height) - file_size))
        pil_img = Image.fromarray(img_raw.reshape((height, width)), 'L').resize((128, 128))

        input_arr = np.array(pil_img).astype('float32') / 255.0
        input_arr = np.expand_dims(input_arr, axis=(0, -1))

        with st.spinner("Running Static Analysis..."):
            prediction = cnn_model.predict(input_arr)
            prob_s1 = float(prediction[0][0])
            verdict_s1 = "RANSOMWARE" if prob_s1 > 0.5 else "BENIGN"
            conf_s1 = prob_s1 if prob_s1 > 0.5 else (1 - prob_s1)
            
        st.metric("Stage 1 Verdict", verdict_s1, f"{conf_s1*100:.2f}% Confidence")

        if conf_s1 >= theta_1:
            st.success(f"✅ Stage 1 Confidence ({conf_s1:.2f}) meets θ1 threshold ({theta_1:.2f}). Exiting pipeline early!")
            st.stop()
        else:
            st.warning(f"⚠️ Stage 1 Confidence ({conf_s1:.2f}) is below θ1 threshold ({theta_1:.2f}). Escalating to Stage 2...")

        # ---------------------------------------------------------
        # STAGE 2: DYNAMIC ANALYSIS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("🔵 STAGE 2: Dynamic Analysis (Triage API + LSTM)")
        
        API_KEY = st.secrets["TRIAGE_API_KEY"]
        HEADERS = {"Authorization": f"Bearer {API_KEY}"}
        BASE_URL = "https://api.tria.ge/v0"

        with st.spinner("Uploading to Triage sandbox..."):
            # CRITICAL FIX: Use .getvalue() directly! This prevents uploading an empty 0-byte file
            # after .read() was consumed by Stage 1. This guarantees Tab 2 parity.
            files = {"file": (gated_exe.name, gated_exe.getvalue())}
            data  = {"_json": '{"kind":"file","interactive":false}'}

            res = requests.post(f"{BASE_URL}/samples", headers=HEADERS, files=files, data=data)
            if res.status_code not in (200, 201):
                st.error(f"Submission failed ({res.status_code}): {res.text}")
                st.stop()

            sample_id = res.json().get("id")
            st.info(f"✅ Submitted — Sample ID: `{sample_id}`")

        with st.spinner("Selecting analysis profile..."):
            requests.post(
                f"{BASE_URL}/samples/{sample_id}/profile",
                headers={**HEADERS, "Content-Type": "application/json"},
                data=json.dumps({"auto": True})
            )

        bar         = st.progress(0)
        status_text = st.empty()
        MAX_ITER    = 60
        curr_status = "pending"

        for i in range(MAX_ITER):
            time.sleep(5)
            chk = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS)
            if chk.status_code != 200: continue
            curr_status = chk.json().get("status", "unknown")
            bar.progress(int(min((i + 1) / MAX_ITER * 100, 100)))
            status_text.text(f"Status: {curr_status}  ({(i+1)*5}s / {MAX_ITER*5}s)")
            if curr_status in ("reported", "failed"): break

        if curr_status == "failed" or curr_status != "reported":
            st.error("Triage analysis failed or timed out. Forcing escalation to Stage 3.")
            conf_s2 = 0.0
        else:
            sample_info = requests.get(f"{BASE_URL}/samples/{sample_id}", headers=HEADERS).json()
            tasks_raw   = sample_info.get("tasks", {})
            behavioral_tasks = []
            
            if isinstance(tasks_raw, dict):
                for full_tid, tinfo in tasks_raw.items():
                    if tinfo.get("kind") == "behavioral": behavioral_tasks.append(full_tid)
            elif isinstance(tasks_raw, list):
                for t in tasks_raw:
                    tid = t.get("id", "")
                    if tid.startswith("behavioral"):
                        behavioral_tasks.append(f"{sample_id}-{tid}" if "-" not in tid else tid)

            if not behavioral_tasks:
                st.error(f"No behavioral tasks found. Forcing escalation.")
                conf_s2 = 0.0
            else:
                raw_apis    = []
                raw_dlls    = []
                raw_mutexes = []
                successful_task = None

                for full_task_id in behavioral_tasks:
                    st.info(f"Fetching onemon.json for task `{full_task_id}`...")
                    for _ in range(20):
                        task_chk = requests.get(f"{BASE_URL}/samples/{sample_id}/{full_task_id}", headers=HEADERS)
                        if task_chk.status_code == 200 and task_chk.json().get("status") == "reported": break
                        time.sleep(5)

                    onemon_res = requests.get(f"{BASE_URL}/samples/{sample_id}/{full_task_id}/logs/onemon.json", headers=HEADERS, stream=True)
                    
                    if onemon_res.status_code != 200:
                        short_id = full_task_id.split("-")[-1]
                        onemon_res = requests.get(f"{BASE_URL}/samples/{sample_id}/{short_id}/logs/onemon.json", headers=HEADERS, stream=True)
                        if onemon_res.status_code != 200: 
                            continue

                    task_apis    = []
                    task_dlls    = []
                    task_mutexes = []

                    for line in onemon_res.iter_lines():
                        if not line: continue
                        try: event = json.loads(line)
                        except: continue

                        kind = event.get("kind", "")
                        evt  = event.get("event", {})

                        if kind == "onemon.Call" or kind.startswith("onemon.Syscall"):
                            api_name = evt.get("api") or evt.get("symbol") or evt.get("name") or evt.get("sys_name")
                            if not api_name and "kind" in evt: api_name = f"Syscall_{evt['kind']}"
                            elif not api_name and "sys" in evt: api_name = f"Syscall_{evt['sys']}"
                            if api_name: task_apis.append(str(api_name))

                        action_name = evt.get("action")
                        if action_name: task_apis.append(str(action_name))

                        for key in ["path", "filepath", "image", "arg0", "arg1", "name"]:
                            val = evt.get(key, "")
                            if isinstance(val, str) and ".dll" in val.lower():
                                dll_name = val.replace("\\", "/").split("/")[-1]
                                if dll_name.lower().endswith(".dll") and dll_name not in task_dlls:
                                    task_dlls.append(dll_name)

                        if kind == "onemon.Mutant" or (kind == "onemon.Handle" and evt.get("type") == "Mutant"):
                            name = evt.get("name") or evt.get("path") or evt.get("mutant") or evt.get("obj")
                            if name and name not in task_mutexes: task_mutexes.append(str(name))

                    if task_apis or task_dlls or task_mutexes:
                        raw_apis    = task_apis
                        raw_dlls    = task_dlls
                        raw_mutexes = task_mutexes
                        successful_task = full_task_id
                        st.success(f"✅ Got behavioral data from task: `{full_task_id}`")
                        break
                    else:
                        st.warning(f"No API/DLL/Mutex data extracted in `{full_task_id}` — trying next task...")

                final_seq = " ".join(raw_apis[:500] + raw_dlls[:10] + raw_mutexes[:10])
                
                if not final_seq.strip():
                    st.warning("No dynamic sequence captured. Forcing escalation.")
                    conf_s2 = 0.0
                else:
                    with st.expander("🔍 Extraction Summary"):
                        st.write(f"Successful task: `{successful_task}`")
                        st.write(f"APIs: **{len(raw_apis)}** | DLLs: **{len(raw_dlls)}** | Mutexes: **{len(raw_mutexes)}**")
                    
                    probs = predict_proba_lstm([final_seq])[0]
                    prob_ransom_s2 = probs[1]
                    verdict_s2 = "RANSOMWARE" if prob_ransom_s2 > 0.5 else "BENIGN"
                    conf_s2 = prob_ransom_s2 if prob_ransom_s2 > 0.5 else probs[0]
                    
                    st.metric("Stage 2 Verdict", verdict_s2, f"{conf_s2*100:.2f}% Confidence")
                    
                    with st.spinner("Generating LIME explanation for Stage 2..."):
                        explainer = LimeTextExplainer(class_names=["Benign", "Ransomware"])
                        exp = explainer.explain_instance(final_seq, predict_proba_lstm, num_features=10)
                        st.write("### 🧠 Top Features (APIs / DLLs / Mutexes)")
                        components.html(exp.as_html(), height=400, scrolling=True)

        if conf_s2 >= theta_2:
            st.success(f"✅ Stage 2 Confidence ({conf_s2:.2f}) meets θ2 threshold ({theta_2:.2f}). Exiting pipeline!")
            st.stop()
        else:
            st.warning(f"⚠️ Stage 2 Confidence ({conf_s2:.2f}) is below θ2 threshold ({theta_2:.2f}). Escalating to Stage 3...")

        # ---------------------------------------------------------
        # STAGE 3: MEMORY ANALYSIS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("🟣 STAGE 3: Memory Analysis (CIC-MalMem-2022)")
        
        if gated_csv is None:
            st.error("❌ Escalation reached Stage 3, but no Memory Features CSV was provided. Please upload the memory dump extract to complete the pipeline.")
            st.stop()
            
        with st.spinner("Analyzing Memory Forensics..."):
            mem_df = pd.read_csv(gated_csv)
            
            # Clean dataframe for prediction (Drop labels if present)
            if 'Class' in mem_df.columns:
                mem_df = mem_df.drop('Class', axis=1)
                
            # Apply Scaler if loaded
            if mem_scaler:
                mem_features = mem_scaler.transform(mem_df)
            else:
                mem_features = mem_df
                
            # Random Forest Prediction
            mem_pred = rf_mem.predict(mem_features)[0]
            mem_prob = rf_mem.predict_proba(mem_features)[0]
            mem_conf = max(mem_prob)
            
            final_verdict = "RANSOMWARE" if mem_pred == 1 else "BENIGN"
            
            st.metric("Final Stage 3 Verdict", final_verdict, f"{mem_conf*100:.2f}% Confidence")
            st.info("🎯 Pipeline Exhausted: The Memory Classifier's verdict is the final decision.")
