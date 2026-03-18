import torch
from torch.utils.data import Dataset

from utils.dataset_loader import get_file_paths
from utils.feature_extractor import DualFeatureExtractor


class DeepfakeDataset(Dataset):
    def __init__(self, audio_dir, protocol_path, max_samples=None, device=None):
        # 🔥 FIX: pass BOTH paths
        self.file_paths, self.labels = get_file_paths(audio_dir, protocol_path)

        if max_samples:
            self.file_paths = self.file_paths[:max_samples]
            self.labels = self.labels[:max_samples]

        self.extractor = DualFeatureExtractor(device=device)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        mfcc, wav2vec = self.extractor.extract(file_path)

        return mfcc, wav2vec, torch.tensor(label, dtype=torch.float32)