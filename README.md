# Deepfake Voice Detection

A deep learning-based system to detect synthetic (deepfake) speech using a combination of self-supervised embeddings and acoustic features.

---

## Overview

This project detects whether an audio sample is REAL or FAKE by leveraging:

* Wav2Vec2 embeddings for high-level speech representations
* MFCC (Mel-Frequency Cepstral Coefficients) for acoustic patterns
* Dual-branch neural architecture for feature fusion

---

## Model Architecture

The system uses a dual-input pipeline:

* Branch 1: Wav2Vec2 – captures semantic and contextual features
* Branch 2: MFCC – captures low-level acoustic cues
* Fusion Layer: Combines both representations
* Classifier: Fully connected layers for final prediction

---

## Dataset

### Primary Dataset

* ASVspoof 2019 (Logical Access)

### Planned Evaluation

* IndicVoices (AI4Bharat) for multilingual testing (in progress)

---

## Performance

Example metrics from training on ASVspoof 2019:

* Accuracy: 98.9%
* Precision: 99.8%
* Recall: 98.8%
* F1 Score: 99.3%
* AUC: 0.9994
* EER: 0.0091

---

## Features

* Deepfake voice detection (REAL vs FAKE)
* Web-based interface (Flask)
* REST API for integration
* Supports `.wav` audio input
* High accuracy on benchmark dataset
* Designed for future multilingual extension

---

## Demo (Local)

Run the Flask app:

```bash
python api.py
```

Open in browser:

```
http://127.0.0.1:5000
```

Upload a `.wav` file and get prediction instantly.

---

## API Usage

### Endpoint:

```
POST /predict
```

### Example (Python):

```python
import requests

url = "http://127.0.0.1:5000/predict"
files = {"file": open("test.wav", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

---

## Project Structure

```
deepfake-voice-detection/
│
├── api.py
├── classifier.py
├── feature_extractor.py
├── models/
├── templates/
├── requirements.txt
```

---

## Installation

```bash
git clone <repo-url>
cd deepfake-voice-detection

pip install -r requirements.txt
```

---

## Future Work

* Multilingual evaluation using IndicVoices dataset
* Mobile app integration (Android / Flutter)
* Real-time microphone input
* Improved generalization across unseen datasets
* Cloud deployment

---

## Status

* Model trained and validated on ASVspoof 2019
* Web application implemented
* Multilingual testing in progress
* Deployment and mobile integration planned

---

## Contributions

Contributions, issues, and suggestions are welcome.

---

## License

This project is for academic and research purposes.
