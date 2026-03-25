import os
import torch
from torch.utils.data import Dataset

from utils.dataset_loader import get_file_paths
from utils.feature_extractor import DualFeatureExtractor


class DeepfakeDataset(Dataset):
    def __init__(self, audio_dir, protocol_path, max_samples=None, device=None):
        self.file_paths, self.labels = get_file_paths(audio_dir, protocol_path)

        if max_samples:
            self.file_paths = self.file_paths[:max_samples]
            self.labels = self.labels[:max_samples]

        self.extractor = DualFeatureExtractor(device=device)

        # ✅ Create cache directories
        self.mfcc_cache_dir = "data/features/mfcc"
        self.wav2vec_cache_dir = "data/features/wav2vec"

        os.makedirs(self.mfcc_cache_dir, exist_ok=True)
        os.makedirs(self.wav2vec_cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        file_name = os.path.basename(file_path).replace(".wav", ".pt")

        mfcc_path = os.path.join(self.mfcc_cache_dir, file_name)
        wav2vec_path = os.path.join(self.wav2vec_cache_dir, file_name)

        # ✅ Load from cache if available
        if os.path.exists(mfcc_path) and os.path.exists(wav2vec_path):
            mfcc = torch.load(mfcc_path)
            wav2vec = torch.load(wav2vec_path)

        else:
            mfcc, wav2vec = self.extractor.extract(file_path)

            # ⚠️ Ensure tensors are on CPU before saving
            mfcc = mfcc.cpu()
            wav2vec = wav2vec.cpu()

            torch.save(mfcc, mfcc_path)
            torch.save(wav2vec, wav2vec_path)

        return mfcc, wav2vec, label