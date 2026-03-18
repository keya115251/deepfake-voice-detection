import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import DeepfakeDataset
from models.classifier import DeepfakeDetector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
audio_dir = "data/processed/ASVspoof2019/LA/train"
protocol_path = "data/raw/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

dataset = DeepfakeDataset(audio_dir, protocol_path, max_samples=2000)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = DeepfakeDetector().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, (mfcc, wav2vec, labels) in enumerate(train_loader):
        mfcc = mfcc.to(device)
        wav2vec = wav2vec.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(mfcc, wav2vec)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

    print(f"\nEpoch {epoch+1} completed | Total Loss: {total_loss:.4f}\n")