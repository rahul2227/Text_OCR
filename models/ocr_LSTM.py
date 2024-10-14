
import torch.nn as nn

from models.feature_extractor import CNNFeatureExtractor
from utils.utils import set_torch_device

device = set_torch_device()
print(f'Using device: {device}')


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


