import torch
from models.classifier import DeepfakeDetector

model = DeepfakeDetector()

mfcc = torch.randn(4, 1, 40, 100)   # batch=4
wav2vec = torch.randn(4, 768)

output = model(mfcc, wav2vec)

print("Output shape:", output.shape)