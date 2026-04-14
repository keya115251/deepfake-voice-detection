import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score
)
from utils.feature_extractor import DualFeatureExtractor
from models.classifier import DeepfakeDetector

# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------
# DATASET
# -----------------------
class FolderDataset(Dataset):
    def __init__(self, dataset_dir, max_samples=None, device=None):
        real_dir = os.path.join(dataset_dir, "real")
        fake_dir = os.path.join(dataset_dir, "fake")

        real_files = [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith(".wav")]
        fake_files = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith(".wav")]

        self.samples = real_files + fake_files

        if max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, min(max_samples, len(self.samples)))

        print(f"Dataset: {len(real_files)} real, {len(fake_files)} fake, {len(self.samples)} total")

        self.extractor = DualFeatureExtractor(device=device)

        self.mfcc_cache_dir    = "data/features/mfcc"
        self.wav2vec_cache_dir = "data/features/wav2vec"
        os.makedirs(self.mfcc_cache_dir, exist_ok=True)
        os.makedirs(self.wav2vec_cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        file_name    = os.path.basename(file_path).replace(".wav", ".pt")
        mfcc_path    = os.path.join(self.mfcc_cache_dir, file_name)
        wav2vec_path = os.path.join(self.wav2vec_cache_dir, file_name)

        if os.path.exists(mfcc_path) and os.path.exists(wav2vec_path):
            mfcc    = torch.load(mfcc_path, weights_only=True)
            wav2vec = torch.load(wav2vec_path, weights_only=True)
        else:
            mfcc, wav2vec = self.extractor.extract(file_path)
            mfcc    = mfcc.cpu()
            wav2vec = wav2vec.cpu()
            torch.save(mfcc, mfcc_path)
            torch.save(wav2vec, wav2vec_path)

        return mfcc, wav2vec, torch.tensor(label, dtype=torch.float32)


# -----------------------
# DATA
# -----------------------
dataset = FolderDataset("combined_dataset")

train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                           generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)


# -----------------------
# MODEL
# -----------------------
model     = DeepfakeDetector().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -----------------------
# EER
# -----------------------
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx], thresholds[idx], fpr, tpr


# -----------------------
# TRAINING
# -----------------------
EPOCHS       = 20
patience     = 5
counter      = 0
best_val_eer = float("inf")
history      = {"train_loss": [], "val_eer": [], "val_auc": []}

os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_scores, all_labels = [], []
    start_time = time.time()

    for mfcc, wav2vec, labels in train_loader:
        mfcc    = mfcc.to(device)
        wav2vec = wav2vec.to(device)
        labels  = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(mfcc, wav2vec)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_scores.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    all_scores  = np.array(all_scores)
    all_labels  = np.array(all_labels)
    train_preds = (all_scores > 0.5).astype(int)
    train_acc   = accuracy_score(all_labels, train_preds)
    train_f1    = f1_score(all_labels, train_preds)
    epoch_loss  = total_loss / len(train_loader)

    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()
    val_scores, val_labels = [], []

    with torch.no_grad():
        for mfcc, wav2vec, labels in val_loader:
            mfcc    = mfcc.to(device)
            wav2vec = wav2vec.to(device)
            labels  = labels.to(device).unsqueeze(1)
            outputs = model(mfcc, wav2vec)
            val_scores.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())

    val_scores  = np.array(val_scores)
    val_labels  = np.array(val_labels)
    val_eer, val_thresh, fpr, tpr = compute_eer(val_labels, val_scores)
    val_auc     = roc_auc_score(val_labels, val_scores)
    val_preds   = (val_scores > val_thresh).astype(int)
    val_acc     = accuracy_score(val_labels, val_preds)
    val_prec    = precision_score(val_labels, val_preds)
    val_rec     = recall_score(val_labels, val_preds)
    val_f1      = f1_score(val_labels, val_preds)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train — Loss: {epoch_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   — Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
    print(f"        EER: {val_eer:.4f} (thr={val_thresh:.4f}) | AUC: {val_auc:.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")

    history["train_loss"].append(epoch_loss)
    history["val_eer"].append(val_eer)
    history["val_auc"].append(val_auc)

    if val_eer < best_val_eer:
        best_val_eer = val_eer
        counter = 0
        torch.save(model.state_dict(), "models/best_model_multilingual.pth")
        print("✅ Best model saved")
    else:
        counter += 1
        if counter >= patience:
            print("⛔ Early stopping triggered")
            break

torch.save(model.state_dict(), "models/final_model_multilingual.pth")
print("Final model saved!")

# -----------------------
# PLOTS
# -----------------------
plt.figure()
plt.plot(history["train_loss"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("models/train_loss_multilingual.png")
plt.show()

plt.figure()
plt.plot(history["val_eer"])
plt.title("Validation EER")
plt.xlabel("Epoch")
plt.ylabel("EER")
plt.savefig("models/val_eer_multilingual.png")
plt.show()

plt.figure()
plt.plot(history["val_auc"])
plt.title("Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.savefig("models/val_auc_multilingual.png")
plt.show()