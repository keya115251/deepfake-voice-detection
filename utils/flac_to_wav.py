import os
import soundfile as sf
from tqdm import tqdm

RAW_BASE = "data/raw/ASVspoof2019/LA"
OUT_BASE = "data/processed/ASVspoof2019/LA"

SPLITS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval"
}

def convert_split(split):
    in_dir = os.path.join(RAW_BASE, SPLITS[split], "flac")
    out_dir = os.path.join(OUT_BASE, split)
    os.makedirs(out_dir, exist_ok=True)

    flac_files = [f for f in os.listdir(in_dir) if f.endswith(".flac")]

    for file in tqdm(flac_files, desc=f"Converting {split}"):
        in_path = os.path.join(in_dir, file)
        out_path = os.path.join(out_dir, file.replace(".flac", ".wav"))

        audio, sr = sf.read(in_path)
        sf.write(out_path, audio, sr)

if __name__ == "__main__":
    for split in ["train", "dev", "eval"]:
        convert_split(split)
