import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class DualFeatureExtractor:
    def __init__(self, device=None):
        print("Loading wav2vec2 model...")

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.")

    def extract_wav2vec(self, waveform):
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.squeeze(0).cpu()  # (768,)

    def extract_mfcc(self, waveform, sample_rate=16000):
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=40
        )

        # normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        # pad/trim → match CNN input
        max_len = 100
        if mfcc.shape[1] < max_len:
            pad = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
        else:
            mfcc = mfcc[:, :max_len]

        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # (1, 40, 100)

    def extract(self, file_path):
        waveform, sample_rate = librosa.load(file_path, sr=16000)

        mfcc = self.extract_mfcc(waveform, sample_rate)
        wav2vec = self.extract_wav2vec(waveform)

        return mfcc, wav2vec