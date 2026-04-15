# Deepfake Voice Detection

A deep learning-based system to detect synthetic (deepfake) speech using a combination of self-supervised embeddings and acoustic features. Supports English, Hindi, and Telugu.

---

## Overview

This project detects whether an audio sample is REAL or FAKE by leveraging:

- **Wav2Vec2 embeddings** (`facebook/wav2vec2-base`) for high-level speech representations
- **MFCC** (Mel-Frequency Cepstral Coefficients) for low-level acoustic patterns
- **Dual-branch neural architecture** for feature fusion
- **Smart silence stripping** to extract the most speech-dense segment for analysis

---

## Model Architecture

The system uses a dual-input pipeline:

- **Branch 1:** Wav2Vec2 — captures semantic and contextual features
- **Branch 2:** MFCC — captures low-level acoustic cues
- **Fusion Layer:** Combines both representations
- **Classifier:** Fully connected layers for final binary prediction (REAL / FAKE)

---

## Datasets

### Training Data (Combined Multilingual Dataset — ~17.5k samples)

| Source | Language | Real | Fake |
|---|---|---|---|
| ASVspoof 2019 (LA) | English | 2,580 | 2,580 |
| AI4Bharat IndicVoices | Hindi | 2,580 | — |
| AI4Bharat IndicVoices | Telugu | 2,580 | — |
| Edge-TTS generated | Hindi | — | 2,580 |
| Edge-TTS generated | Telugu | — | 2,580 |

Real and fake samples are balanced per language to prevent the model from learning language-specific rather than spoof-specific artifacts.

---

## Performance

### ASVspoof 2019 Only (English baseline)

| Metric | Score |
|---|---|
| Accuracy | 98.9% |
| Precision | 99.8% |
| Recall | 98.8% |
| F1 Score | 99.3% |
| AUC | 0.9994 |
| EER | 0.91% |

### Multilingual Model (English + Hindi + Telugu)

| Metric | Score |
|---|---|
| Accuracy | 99.45% |
| Precision | 99.49% |
| Recall | 99.42% |
| F1 Score | 99.46% |
| AUC | 0.9997 |
| EER | 0.52% |

> Note: Indic fake samples are Edge-TTS generated. Generalization to other TTS systems is a known limitation and area of future work.

---

## Features

- Deepfake voice detection (REAL vs FAKE)
- Multilingual support: English, Hindi, Telugu
- Web-based interface (Flask) with file upload and live microphone recording
- Smart audio segmentation — automatically selects the most speech-dense 5-second window
- Supports multiple audio formats: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`
- REST API for integration
- Feature caching for faster retraining

---

## Demo (Local)

### Prerequisites

- Python 3.9+
- ffmpeg installed and added to PATH

### Installation

```bash
git clone https://github.com/keya115251/deepfake-voice-detection
cd deepfake-voice-detection
pip install -r requirements.txt
```

### Run

```bash
python api2.py
```

Open in browser:

## License

This project is for academic and research purposes.
