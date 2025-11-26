import torch
from torch import nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)       
        )

    def forward(self, x):
        return self.model(x)
