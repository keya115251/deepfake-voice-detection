import torch
import os
from flask import Flask, request, jsonify, render_template

# ✅ Import your classes
from utils.feature_extractor import DualFeatureExtractor
from models.classifier import DeepfakeDetector

# -----------------------
# INIT
# -----------------------
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# -----------------------
# LOAD FEATURE EXTRACTOR
# -----------------------
extractor = DualFeatureExtractor(device=device)

# -----------------------
# LOAD MODEL
# -----------------------
model = DeepfakeDetector().to(device)

# 👉 CHANGE THIS if needed
MODEL_PATH = "models/best_model.pth"

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully.")

# -----------------------
# HOME ROUTE (UI)
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------
# PREDICT ROUTE (BROWSER + API)
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    try:
        mfcc, wav2vec = extractor.extract(filepath)

        mfcc = mfcc.unsqueeze(0)
        wav2vec = wav2vec.unsqueeze(0)

        with torch.no_grad():
            output = model(mfcc, wav2vec)
            prob = torch.sigmoid(output).item()

        result = "FAKE" if prob > 0.5 else "REAL"

        # 🔥 KEY LOGIC
        if "text/html" in request.headers.get("Accept", ""):
            return render_template(
                "result.html",
                result=result,
                confidence=f"{prob:.4f}"
            )
        else:
            return jsonify({
                "result": result,
                "confidence": prob
            })

    except Exception as e:
        return str(e), 500
# -----------------------
# RUN SERVER
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)