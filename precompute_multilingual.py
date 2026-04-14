import os
import torch
from tqdm import tqdm
from utils.feature_extractor import DualFeatureExtractor

# -----------------------
# CONFIGURATION
# -----------------------
dataset_dir    = "combined_dataset"
mfcc_cache     = "data/features/mfcc"
wav2vec_cache  = "data/features/wav2vec"

os.makedirs(mfcc_cache, exist_ok=True)
os.makedirs(wav2vec_cache, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# COLLECT ALL FILES
# -----------------------
all_files = []
for label in ["real", "fake"]:
    folder = os.path.join(dataset_dir, label)
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            all_files.append(os.path.join(folder, f))

print(f"Total files to process: {len(all_files)}")

# -----------------------
# LOAD EXTRACTOR
# -----------------------
extractor = DualFeatureExtractor(device=device)

# -----------------------
# PRECOMPUTE
# -----------------------
skipped = 0
errors  = 0

for file_path in tqdm(all_files, desc="Extracting features"):
    file_name    = os.path.basename(file_path).replace(".wav", ".pt")
    mfcc_path    = os.path.join(mfcc_cache, file_name)
    wav2vec_path = os.path.join(wav2vec_cache, file_name)

    # skip if already cached
    if os.path.exists(mfcc_path) and os.path.exists(wav2vec_path):
        skipped += 1
        continue

    try:
        mfcc, wav2vec = extractor.extract(file_path)
        torch.save(mfcc.cpu(),    mfcc_path)
        torch.save(wav2vec.cpu(), wav2vec_path)
    except Exception as e:
        errors += 1
        print(f"\nError on {file_path}: {e}")

print(f"\n✅ Done!")
print(f"   Processed : {len(all_files) - skipped - errors}")
print(f"   Skipped   : {skipped} (already cached)")
print(f"   Errors    : {errors}")