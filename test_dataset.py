from utils.dataset import DeepfakeDataset

audio_dir = "data/processed/ASVspoof2019/LA/train"

protocol_path = "data/raw/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

print("Creating dataset...")
dataset = DeepfakeDataset(audio_dir, protocol_path, max_samples=3)

print("Dataset length:", len(dataset))

for i in range(len(dataset)):
    print(f"\nProcessing sample {i+1}...")

    mfcc, wav2vec, label = dataset[i]

    print("MFCC shape:", mfcc.shape)
    print("Wav2Vec shape:", wav2vec.shape)
    print("Label:", label)