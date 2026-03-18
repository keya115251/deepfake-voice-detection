import os
from utils.dataset_loader import get_file_paths
from utils.feature_extractor import DualFeatureExtractor




audio_dir = os.path.join(
    "data",
    "processed",
    "ASVspoof2019",
    "LA",
    "train"
)

protocol_path = os.path.join(
    "data",
    "raw",
    "ASVspoof2019",
    "LA",
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.train.trn.txt"
)

files, labels = get_file_paths(audio_dir, protocol_path)

# Take only 5 files
files = files[:5]

extractor = DualFeatureExtractor()

for i, file in enumerate(files):
    print(f"\nProcessing file {i+1}")
    embedding = extractor.extract(file)
    print("Embedding shape:", embedding.shape)