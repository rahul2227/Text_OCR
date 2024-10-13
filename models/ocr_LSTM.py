import os

import torch
import torch.nn as nn
import torch.optim as optim
import json

from utils.utils import get_project_root


# TODO: Check from where the optimizer is being imported


# -------------------------------
# Check for GPU availability and set device to mpu if available
# -------------------------------

def set_torch_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


device = set_torch_device()
print(f'Using device: {device}')


# -------------------------------
# CNN Feature Extractor
# -------------------------------

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Input Channels=1 for grayscale
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by a factor of 2

            # Conv Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


# -------------------------------
# Reshaping CNN Output for RNN Layers
# -------------------------------

class CNNToRNN(nn.Module):
    def __init__(self, cnn_output_channels, fixed_height, num_pooling_layers=4):
        super(CNNToRNN, self).__init__()
        self.fixed_height = fixed_height
        self.cnn_output_channels = cnn_output_channels
        self.num_pooling_layers = num_pooling_layers

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        expected_height = self.fixed_height // (2 ** self.num_pooling_layers)
        assert height == expected_height, f"Height mismatch: Expected {expected_height}, got {height}"

        # Permute to (width, batch_size, channels, height)
        x = x.permute(3, 0, 1, 2)  # (width, batch_size, channels, height)
        x = x.contiguous().view(width, batch_size, channels * height)  # (width, batch_size, channels * height)
        return x


# -------------------------------
# Bidirectional LSTM Layers
# -------------------------------

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.1):
        super(BidirectionalLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # x shape: (seq_length, batch_size, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out: (seq_length, batch_size, hidden_size * 2)
        logits = self.fc(lstm_out)  # logits: (seq_length, batch_size, num_classes)
        return logits


# -------------------------------
# Bidirectional LSTM Layers
# -------------------------------

class OCRModel(nn.Module):
    def __init__(self, fixed_height, fixed_width, num_classes, hidden_size=256, num_lstm_layers=2, dropout=0.1):
        super(OCRModel, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.cnn_to_rnn = CNNToRNN(cnn_output_channels=512, fixed_height=fixed_height)
        self.rnn = BidirectionalLSTM(input_size=512 * (fixed_height // (2 ** 4)),  # Adjust based on CNN layers
                                     hidden_size=hidden_size,
                                     num_layers=num_lstm_layers,
                                     num_classes=num_classes,
                                     dropout=dropout)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        # x shape: (batch_size, 1, fixed_height, fixed_width)
        x = self.cnn(x)  # (batch_size, 512, H', W')
        x = self.cnn_to_rnn(x)  # (W', batch_size, 512 * H')
        x = self.rnn(x)  # ('W', batch_size, num_classes)
        x = self.log_softmax(x)  # ('W', batch_size, num_classes)
        return x


# -------------------------------
# Compiling model with CTC loss
# -------------------------------

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

# -------------------------------
# Defining the model
# -------------------------------

# TODO: get essential mappings like char_to_idx from json mappings
root_dir = get_project_root()
mappings_path = os.path.join(root_dir, 'data_preprocessing/mappings.json')
# Load mappings
print("Loading mappings...")
with open(mappings_path, 'r', encoding='utf-8') as f:
    mappings = json.load(f)

char_to_idx = mappings['char_to_idx']
idx_to_char = {int(k): v for k, v in mappings['idx_to_char'].items()}  # Ensure keys are integers
max_width = mappings['max_width']
fixed_height = mappings['fixed_height']

# Define essential parameters
# fixed_height = 128
fixed_width = max_width # 1024  # This should match the 'max_width' from preprocessing
num_classes = len(char_to_idx)  # Number of unique characters including <PAD> and <UNK>
hidden_size = 256
num_lstm_layers = 2
dropout = 0.1

# Instantiate the model
model = OCRModel(fixed_height=fixed_height,
                 fixed_width=fixed_width,
                 num_classes=num_classes,
                 hidden_size=hidden_size,
                 num_lstm_layers=num_lstm_layers,
                 dropout=dropout).to(device)

print(model)

# Optimizer
learning_rate = 1e-3

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
