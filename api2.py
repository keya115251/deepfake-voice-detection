import torch
import os
import librosa
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template
import soundfile as sf

from utils.feature_extractor import DualFeatureExtractor
from models.classifier import DeepfakeDetector

# -----------------------
# INIT
# -----------------------
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

extractor = DualFeatureExtractor(device=device)

model = DeepfakeDetector().to(device)
MODEL_PATH = "models/best_model_multilingual.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("Multilingual model loaded successfully.")

# -----------------------
# CONFIG
# -----------------------
MAX_AUDIO_LEN   = 5 * 16000
THRESHOLD       = 0.7735
SILENCE_DB      = 30
MIN_SPEECH_SECS = 1.0


# -----------------------
# SMART AUDIO EXTRACTION
# -----------------------
def extract_best_segment(audio, sr=16000):
    trimmed, _ = librosa.effects.trim(audio, top_db=SILENCE_DB)

    if len(trimmed) < MIN_SPEECH_SECS * sr:
        return audio[:MAX_AUDIO_LEN]

    if len(trimmed) <= MAX_AUDIO_LEN:
        return trimmed

    window     = MAX_AUDIO_LEN
    step       = sr // 2
    best_start = 0
    best_rms   = -1

    for start in range(0, len(trimmed) - window, step):
        segment = trimmed[start:start + window]
        rms     = np.sqrt(np.mean(segment ** 2))
        if rms > best_rms:
            best_rms   = rms
            best_start = start

    return trimmed[best_start:best_start + window]


# -----------------------
# SHARED PREDICT LOGIC
# -----------------------
def run_prediction(audio, sr=16000):
    segment  = extract_best_segment(audio, sr)
    temp_wav = f"temp_{uuid.uuid4().hex}.wav"
    sf.write(temp_wav, segment, sr)

    try:
        mfcc, wav2vec = extractor.extract(temp_wav)
        mfcc    = mfcc.unsqueeze(0).to(device)
        wav2vec = wav2vec.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(mfcc, wav2vec)
            prob   = torch.sigmoid(output).item()

        return prob
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


# -----------------------
# HOME
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------
# PREDICT — FILE UPLOAD
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file         = request.files["file"]
    original_ext = os.path.splitext(file.filename)[-1].lower() or ".wav"
    temp_input   = f"temp_input_{uuid.uuid4().hex}{original_ext}"

    file.save(temp_input)

    try:
        audio, sr = librosa.load(temp_input, sr=16000, mono=True)

        trimmed, _ = librosa.effects.trim(audio, top_db=SILENCE_DB)
        if len(trimmed) < MIN_SPEECH_SECS * sr:
            return jsonify({"error": "Audio contains too little speech. Please upload a clearer recording."}), 400

        prob       = run_prediction(audio, sr)
        result     = "FAKE" if prob > THRESHOLD else "REAL"
        fake_score = f"{prob * 100:.1f}%"
        real_score = f"{(1 - prob) * 100:.1f}%"

        if "text/html" in request.headers.get("Accept", ""):
            return render_template(
                "result.html",
                result=result,
                fake_score=fake_score,
                real_score=real_score,
                note="Multilingual model (English / Hindi / Telugu)"
            )
        else:
            return jsonify({
                "result":     result,
                "fake_score": fake_score,
                "real_score": real_score,
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)


# -----------------------
# PREDICT — REALTIME RECORDING
# -----------------------
@app.route("/predict_realtime", methods=["POST"])
def predict_realtime():
    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400

    file       = request.files["audio"]
    filename   = file.filename or "recording.webm"
    ext        = os.path.splitext(filename)[-1].lower() or ".webm"
    temp_input = f"temp_rt_{uuid.uuid4().hex}{ext}"

    file.save(temp_input)

    try:
        audio, sr = librosa.load(temp_input, sr=16000, mono=True)

        trimmed, _ = librosa.effects.trim(audio, top_db=SILENCE_DB)
        if len(trimmed) < MIN_SPEECH_SECS * sr:
            return jsonify({"error": "Too much silence. Please speak clearly into the microphone."}), 400

        prob       = run_prediction(audio, sr)
        result     = "FAKE" if prob > THRESHOLD else "REAL"
        fake_score = f"{prob * 100:.1f}%"
        real_score = f"{(1 - prob) * 100:.1f}%"

        return jsonify({
            "result":     result,
            "fake_score": fake_score,
            "real_score": real_score,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)


# -----------------------
# SHOW RESULT (from recording redirect)
# -----------------------
@app.route("/show_result", methods=["POST"])
def show_result():
    result     = request.form.get("result")
    fake_score = request.form.get("fake_score")
    real_score = request.form.get("real_score")
    return render_template(
        "result.html",
        result=result,
        fake_score=fake_score,
        real_score=real_score,
        note="Multilingual model (English / Hindi / Telugu)"
    )


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)