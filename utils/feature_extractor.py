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

        # ✅ NORMALIZATION (VERY IMPORTANT)
        embedding = (embedding - embedding.mean()) / (embedding.std() + 1e-6)

        return embedding.squeeze(0)  # (768,)

    def extract_mfcc(self, waveform, sample_rate=16000):
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_mean = np.mean(mfcc, axis=1)

        mfcc_tensor = torch.tensor(mfcc_mean, dtype=torch.float32)

        # ✅ Normalize MFCC too
        mfcc_tensor = (mfcc_tensor - mfcc_tensor.mean()) / (mfcc_tensor.std() + 1e-6)

        return mfcc_tensor.to(self.device)

    def extract(self, file_path):
        waveform, sample_rate = librosa.load(file_path, sr=16000)

        wav2vec_embedding = self.extract_wav2vec(waveform)
        mfcc_embedding = self.extract_mfcc(waveform, sample_rate)

        return mfcc_embedding, wav2vec_embedding
    def extract_from_waveform(self, waveform, sample_rate=16000):
    # Ensure waveform is numpy (librosa expects it)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        wav2vec_embedding = self.extract_wav2vec(waveform)
        mfcc_embedding = self.extract_mfcc(waveform, sample_rate)

        return mfcc_embedding, wav2vec_embedding