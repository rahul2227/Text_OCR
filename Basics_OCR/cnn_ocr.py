import torch.nn as nn
import torch.nn.functional as F
import torch


class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x28x28
        self.pool = nn.MaxPool2d(2, 2)  # Output: 32x14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x14x14
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 47)  # 47 classes in EMNIST balanced split

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))  # FC1 + ReLU
        x = self.fc2(x)  # FC2
        return x


# Instantiate the model_LSTM
model = OCRModel().to(torch.device("mps"))
