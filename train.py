import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.dataset import DeepfakeDataset
from models.classifier import DeepfakeDetector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


audio_dir = "data/processed/ASVspoof2019/LA/train"
protocol_path = "data/raw/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"


dataset = DeepfakeDataset(audio_dir, protocol_path, max_samples=5000)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


model = DeepfakeDetector().to(device)

pos_weight = torch.tensor([1.5]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)


EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    epoch_start = time.time()

    all_preds = []
    all_labels = []

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

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.7).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        batch_time = time.time() - batch_start

        if i % 20 == 0:
            avg_loss = total_loss / (i + 1)
            print(f"Epoch {epoch+1} | Batch {i} | Avg Loss: {avg_loss:.4f} | Time: {batch_time:.2f}s")

    epoch_time = time.time() - epoch_start
    epoch_loss = total_loss / len(train_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nEpoch {epoch+1} completed")
    print(f"Avg Loss : {epoch_loss:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Epoch Time: {epoch_time:.2f} seconds\n")

    # 🔵 VALIDATION
    model.eval()

    val_preds = []
    val_labels = []

    with torch.no_grad():
        for mfcc, wav2vec, labels in val_loader:
            mfcc = mfcc.to(device)
            wav2vec = wav2vec.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(mfcc, wav2vec)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds)
    val_rec = recall_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)

    print("---- Validation ----")
    print(f"Val Accuracy : {val_acc:.4f}")
    print(f"Val Precision: {val_prec:.4f}")
    print(f"Val Recall   : {val_rec:.4f}")
    print(f"Val F1 Score : {val_f1:.4f}\n")


torch.save(model.state_dict(), "models/deepfake_detector.pth")
print("Model saved successfully!")