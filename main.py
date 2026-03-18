from utils.dataset_loader import load_protocol, get_audio_path
import os
import librosa

# Base paths
BASE_PATH = "data/processed/ASVspoof2019"
PROTOCOL_PATH = (
    "data/raw/ASVspoof2019/LA/"
    "ASVspoof2019_LA_cm_protocols/"
    "ASVspoof2019.LA.cm.train.trn.txt"
)

# Load protocol
df = load_protocol(PROTOCOL_PATH)
print("Protocol loaded:")
print(df.head(), "\n")

# Pick one sample
sample = df.iloc[0]
audio_id = sample["audio_id"]
label = sample["label"]

# Get WAV path
audio_path = get_audio_path(BASE_PATH, "train", audio_id)

print("Audio ID:", audio_id)
print("Label:", label)
print("Audio path:", audio_path)
print("File exists:", os.path.exists(audio_path))

# Load audio
audio, sr = librosa.load(audio_path, sr=None)

print("\nAudio loaded successfully!")
print("Duration (seconds):", round(len(audio) / sr, 2))
print("Sample rate:", sr)
