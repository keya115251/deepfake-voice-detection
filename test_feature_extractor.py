import os
from utils.feature_extractor import DualFeatureExtractor

# Path to your dataset folder
DATA_PATH = "data/processed/ASVspoof2019/LA/train"

# Initialize extractor
extractor = DualFeatureExtractor()

# Get first 5 .wav files
file_list = [
    os.path.join(DATA_PATH, f)
    for f in os.listdir(DATA_PATH)
    if f.endswith(".wav")
][:5]


# Test extraction
for i, file in enumerate(file_list):
    print(f"\nProcessing file {i+1}")
    print("File:", file)

    try:
        mfcc, wav2vec = extractor.extract(file)

        print("MFCC shape   :", mfcc.shape)      # Expected: (40,)
        print("Wav2Vec shape:", wav2vec.shape)   # Expected: (768,)

    except Exception as e:
        print("Error processing file:", e)