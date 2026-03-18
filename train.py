import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from utils.dataset import DeepfakeDataset
from models.classifier import DeepfakeDetector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


audio_dir = "data/processed/ASVspoof2019/LA/train"
protocol_path = "data/raw/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"


dataset = DeepfakeDataset(audio_dir, protocol_path, max_samples=2000)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)


model = DeepfakeDetector().to(device)


# ✅ FIXED (reduced weight)
pos_weight = torch.tensor([3.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ✅ Slightly lower LR
optimizer = optim.Adam(model.parameters(), lr=0.00003)


EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    epoch_start = time.time()

    for i, (mfcc, wav2vec, labels) in enumerate(train_loader):
        batch_start = time.time()

        mfcc = mfcc.to(device)
        wav2vec = wav2vec.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(mfcc, wav2vec)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_time = time.time() - batch_start

        # ✅ Debug outputs
        if i % 20 == 0:
            avg_loss = total_loss / (i + 1)
            print(f"Epoch {epoch+1} | Batch {i} | Avg Loss: {avg_loss:.4f} | Time: {batch_time:.2f}s")

    epoch_time = time.time() - epoch_start
    epoch_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} completed")
    print(f"Avg Loss: {epoch_loss:.4f}")
    print(f"Epoch Time: {epoch_time:.2f} seconds\n")


torch.save(model.state_dict(), "models/deepfake_detector.pth")
print("Model saved successfully!")