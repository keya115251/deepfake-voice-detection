import torch
import torch.nn.functional as F
import torchaudio
import argparse

from models.classifier import DeepfakeDetector
from utils.feature_extractor import DualFeatureExtractor


# 🔧 Load model
def load_model(model_path, device):
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    return model


# 🔧 Preprocess audio (FLAC / WAV support)
def load_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # shape: [samples]


# 🔧 Run inference
def predict(file_path, model, extractor, device):
    waveform = load_audio(file_path)

    # Extract features
    mfcc, wav2vec = extractor.extract_from_waveform(waveform)

    # Add batch dimension
    mfcc = mfcc.unsqueeze(0).to(device)
    wav2vec = wav2vec.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(mfcc, wav2vec)
        prob = torch.sigmoid(output).item()

    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, default="models/deepfake_detector.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = load_model(args.model, device)

    # Feature extractor (same as training)
    extractor = DualFeatureExtractor(device=device)

    # Predict
    prob = predict(args.audio, model, extractor, device)

    # Output
    if prob > 0.5:
        print(f"\nPrediction: FAKE")
    else:
        print(f"\nPrediction: REAL")

    print(f"Confidence: {prob:.4f}")


if __name__ == "__main__":
    main()