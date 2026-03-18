import torch
import torch.nn as nn
import torch.nn.functional as F

class MFCC_CNN(nn.Module):
    def __init__(self):
        super(MFCC_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 🔥 Adaptive pooling fixes everything
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        self.fc = nn.Linear(32 * 5 * 5, 128)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = self.adaptive_pool(x)  # 🔥 force fixed size

        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc(x))

        return x


class Wav2VecBranch(nn.Module):
    def __init__(self, input_dim=768):
        super(Wav2VecBranch, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()

        self.mfcc_branch = MFCC_CNN()
        self.wav2vec_branch = Wav2VecBranch()

        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, mfcc, wav2vec):
        mfcc_feat = self.mfcc_branch(mfcc)
        wav2vec_feat = self.wav2vec_branch(wav2vec)

        combined = torch.cat((mfcc_feat, wav2vec_feat), dim=1)

        output = self.classifier(combined)

        return torch.sigmoid(output)