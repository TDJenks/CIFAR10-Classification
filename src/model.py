import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv1: input = 3x32x32 -> output = 32x16x16 (after pooling)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)    
        self.bn1 = nn.BatchNorm2d(32)

        # Conv2: input = 32x16x16 -> output = 64x8x8
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv3: input = 64x8x8 -> output = 128x8x8 (no pool here)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling: 32x32 -> 16x16 etc.
        self.pool = nn.MaxPool2d(2, 2)

        # Flatten output: 128 × 8 × 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  
        self.fc2 = nn.Linear(256, 10)  # output layer -> 10 classes

        # Dropout
        self.dropout = nn.Dropout(p=0.5)  # 50% chance to zero out neurons

    def forward(self, x):
        # Conv -> Normalization -> ReLU -> Pool(except last)  x3
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # FC flattening
        x = x.view(x.size(0), -1)

        # Dropout to help learning and prevent overfitting :p
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # TODO Softmax 
       
        return x