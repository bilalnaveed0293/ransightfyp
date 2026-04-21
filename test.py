import time
import numpy as np
import tensorflow as tf
import joblib
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── LOAD MODELS ─────────────────────────────────────────────────────────
cnn    = tf.keras.models.load_model("cnn1.keras")
lstm   = tf.keras.models.load_model("ransomware_lstm_dynamic.keras")
rf     = joblib.load("malmem_rf_memory_model.joblib")
scaler = joblib.load("malmem_memory_scaler.joblib")
with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

# ── DUMMY INPUTS (matching exact shapes your models expect) ─────────────
dummy_image  = np.random.rand(1, 128, 128, 1).astype("float32")

dummy_text   = "NtCreateFile NtWriteFile NtReadFile CreateMutexW " * 130  # ~520 tokens
dummy_seq    = pad_sequences(tok.texts_to_sequences([dummy_text]), maxlen=520, padding="post")

dummy_memory = np.random.rand(1, rf.n_features_in_)
dummy_memory_scaled = scaler.transform(dummy_memory)

N = 100  # number of runs to average over

# ── WARM UP (first prediction is always slow due to TF graph build) ─────
cnn.predict(dummy_image,  verbose=0)
lstm.predict(dummy_seq,   verbose=0)
rf.predict(dummy_memory_scaled)

# ── CNN INFERENCE TIME ───────────────────────────────────────────────────
cnn_times = []
for _ in range(N):
    s = time.perf_counter()
    cnn.predict(dummy_image, verbose=0)
    cnn_times.append((time.perf_counter() - s) * 1000)

print(f"CNN  — avg: {np.mean(cnn_times):.2f} ms | "
      f"min: {np.min(cnn_times):.2f} ms | "
      f"max: {np.max(cnn_times):.2f} ms")

# ── LSTM INFERENCE TIME ──────────────────────────────────────────────────
lstm_times = []
for _ in range(N):
    s = time.perf_counter()
    lstm.predict(dummy_seq, verbose=0)
    lstm_times.append((time.perf_counter() - s) * 1000)

print(f"LSTM — avg: {np.mean(lstm_times):.2f} ms | "
      f"min: {np.min(lstm_times):.2f} ms | "
      f"max: {np.max(lstm_times):.2f} ms")

# ── RF INFERENCE TIME ────────────────────────────────────────────────────
rf_times = []
for _ in range(N):
    s = time.perf_counter()
    rf.predict(dummy_memory_scaled)
    rf_times.append((time.perf_counter() - s) * 1000)

print(f"RF   — avg: {np.mean(rf_times):.2f} ms | "
      f"min: {np.min(rf_times):.2f} ms | "
      f"max: {np.max(rf_times):.2f} ms")

# ── PIPELINE TOTAL (if all 3 run on same sample) ─────────────────────────
pipeline_avg = np.mean(cnn_times) + np.mean(lstm_times) + np.mean(rf_times)
print(f"\nFull pipeline (worst case, all 3 models): {pipeline_avg:.2f} ms")
print(f"Best case (CNN clears sample, stops here): {np.mean(cnn_times):.2f} ms")

# ── MODEL SIZES ──────────────────────────────────────────────────────────
import os
print(f"\nCNN  size: {os.path.getsize('cnn1.keras') / (1024*1024):.2f} MB")
print(f"LSTM size: {os.path.getsize('ransomware_lstm_dynamic.keras') / (1024*1024):.2f} MB")
print(f"RF   size: {os.path.getsize('malmem_rf_memory_model.joblib') / (1024*1024):.2f} MB")

# ── PARAMETER COUNTS ─────────────────────────────────────────────────────
print(f"\nCNN  params: {cnn.count_params():,}")
print(f"LSTM params: {lstm.count_params():,}")
print(f"RF   trees:  {rf.n_estimators} | features: {rf.n_features_in_}")
