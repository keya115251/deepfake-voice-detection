import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score
)

from utils.dataset import DeepfakeDataset
from models.classifier import DeepfakeDetector


# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------
# DATA
# -----------------------
audio_dir = "data/processed/ASVspoof2019/LA/train"
protocol_path = "data/raw/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

dataset = DeepfakeDataset(audio_dir, protocol_path, max_samples=5000)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# -----------------------
# MODEL
# -----------------------
model = DeepfakeDetector().to(device)

pos_weight = torch.tensor([1.5]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -----------------------
# EER FUNCTION
# -----------------------
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    threshold = thresholds[idx]

    return eer, threshold, fpr, tpr


# -----------------------
# TRAINING SETTINGS
# -----------------------
EPOCHS = 20
best_val_eer = float("inf")
patience = 5
counter = 0

history = {
    "train_loss": [],
    "val_eer": [],
    "val_auc": []
}


# -----------------------
# TRAINING LOOP
# -----------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()

    all_scores = []
    all_labels = []

    for mfcc, wav2vec, labels in train_loader:
        mfcc = mfcc.to(device)
        wav2vec = wav2vec.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(mfcc, wav2vec)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        labs = labels.cpu().numpy().flatten()

        all_scores.extend(probs)
        all_labels.extend(labs)

    # -----------------------
    # TRAIN METRICS
    # -----------------------
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    train_preds = (all_scores > 0.5).astype(int)

    train_acc = accuracy_score(all_labels, train_preds)
    train_f1 = f1_score(all_labels, train_preds)

    epoch_loss = total_loss / len(train_loader)

    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()
    val_scores = []
    val_labels = []

    with torch.no_grad():
        for mfcc, wav2vec, labels in val_loader:
            mfcc = mfcc.to(device)
            wav2vec = wav2vec.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(mfcc, wav2vec)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_scores.extend(probs)
            val_labels.extend(labels.cpu().numpy().flatten())

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)

    # EER + ROC
    val_eer, val_thresh, fpr, tpr = compute_eer(val_labels, val_scores)
    val_auc = roc_auc_score(val_labels, val_scores)

    # Use EER threshold (IMPORTANT FIX)
    val_preds = (val_scores > val_thresh).astype(int)

    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds)
    val_rec = recall_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)

    # -----------------------
    # LOGGING
    # -----------------------
    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {epoch_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f} | F1: {train_f1:.4f}")

    print("---- Validation ----")
    print(f"Acc : {val_acc:.4f}")
    print(f"Prec: {val_prec:.4f}")
    print(f"Rec : {val_rec:.4f}")
    print(f"F1  : {val_f1:.4f}")
    print(f"EER : {val_eer:.4f} (thr={val_thresh:.4f})")
    print(f"AUC : {val_auc:.4f}")

    print(f"Time: {time.time() - start_time:.2f}s")

    # Save history
    history["train_loss"].append(epoch_loss)
    history["val_eer"].append(val_eer)
    history["val_auc"].append(val_auc)

    # -----------------------
    # EARLY STOPPING
    # -----------------------
    if val_eer < best_val_eer:
        best_val_eer = val_eer
        counter = 0
        torch.save(model.state_dict(), "models/best_model.pth")
        print("✅ Best model saved")
    else:
        counter += 1
        if counter >= patience:
            print("⛔ Early stopping triggered")
            break

    # -----------------------
    # ROC CURVE
    # -----------------------
    plt.figure()
    plt.plot(fpr, tpr, label=f"EER={val_eer:.4f}, AUC={val_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve - Epoch {epoch+1}")
    plt.legend()
    plt.show()


# -----------------------
# FINAL SAVE
# -----------------------
torch.save(model.state_dict(), "models/final_model.pth")
print("Final model saved!")


# -----------------------
# PLOT TRAINING CURVES
# -----------------------
plt.figure()
plt.plot(history["train_loss"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(history["val_eer"])
plt.title("Validation EER")
plt.xlabel("Epoch")
plt.ylabel("EER")
plt.show()

plt.figure()
plt.plot(history["val_auc"])
plt.title("Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.show()