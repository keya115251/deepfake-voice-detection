import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# MFCC Branch (CNN)
# =========================
class MFCC_CNN(nn.Module):
    def __init__(self):
        super(MFCC_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d((2, 1))  # ✅ only reduce height

        self.flatten = nn.Flatten()

        # Input: [batch, 1, 40, 1]
        # After conv + pool → [batch, 16, 20, 1]
        self.fc = nn.Linear(16 * 20 * 1, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x


# =========================
# Wav2Vec Branch (MLP)
# =========================
class Wav2VecBranch(nn.Module):
    def __init__(self):
        super(Wav2VecBranch, self).__init__()

        self.fc1 = nn.Linear(768, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


# =========================
# Final Classifier
# =========================
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()

        self.mfcc_branch = MFCC_CNN()
        self.wav2vec_branch = Wav2VecBranch()

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),   # ✅ helps stabilize training
            nn.Linear(64, 1)
        )

    def forward(self, mfcc, wav2vec):
        # ✅ reshape MFCC for CNN
        mfcc = mfcc.unsqueeze(1).unsqueeze(-1)  # [B,1,40,1]

        mfcc_feat = self.mfcc_branch(mfcc)
        wav2vec_feat = self.wav2vec_branch(wav2vec)

        # ✅ fuse features
        combined = torch.cat((mfcc_feat, wav2vec_feat), dim=1)

        output = self.classifier(combined)

        return output