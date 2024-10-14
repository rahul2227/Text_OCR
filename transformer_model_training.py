# transformer_model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import json
import os
import math

from data_loaders.data_loader import OCRDataset, collate_fn
from models.ocr_transformer import TransformerOCR
from utils.utils import get_project_root  # Assuming you have this utility

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# root directory
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
fixed_width = max_width  # From preprocessing
num_classes = len(char_to_idx)  # Number of unique characters including <PAD> and <UNK>
hidden_size = 512
num_transformer_layers = 6
nhead = 8
dim_feedforward = 2048
dropout = 0.1
max_seq_length = 100  # Adjust based on maximum transcription length

# Instantiate the model
model = TransformerOCR(
    img_feature_dim=512,              # From CNNFeatureExtractor
    img_feature_seq_len=fixed_width // (2 ** 4),  # Assuming 4 max-pooling layers
    d_model=hidden_size,
    nhead=nhead,
    num_encoder_layers=num_transformer_layers,
    num_decoder_layers=num_transformer_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    num_classes=num_classes,
    max_seq_length=max_seq_length
).to(device)

print(model)

# Define the loss function (CTC Loss is not suitable for Transformer)
# Instead, use CrossEntropyLoss with appropriate padding and masking
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>']).to(device)

# Define the optimizer
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Instantiate the Dataset and DataLoader
root_dir = get_project_root()
dataset = OCRDataset(
    images_path=os.path.join(root_dir, 'data_preprocessing/preprocessed_images.npy'),
    encoded_transcriptions_path=os.path.join(root_dir, 'data_preprocessing/encoded_transcriptions.json'),
    transform=None  # No additional transforms needed as preprocessing is done
)

batch_size = 16  # Adjust based on memory constraints
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

# -------------------------------
# 3. Define the Training Loop
# -------------------------------

def train_transformer(model, dataloader, criterion, optimizer, idx_to_char, num_epochs=10, clip=1.0):
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (images, transcriptions, input_lengths, target_lengths) in enumerate(dataloader):
            images = images.to(device)  # (batch_size, 1, H, W)
            transcriptions = transcriptions.to(device)  # (sum(target_lengths))
            input_lengths = input_lengths.to(device)    # (batch_size)
            target_lengths = target_lengths.to(device)  # (batch_size)

            batch_size_actual = images.size(0)
            max_target_length = target_lengths.max().item()

            # Prepare target sequences for Transformer:
            # Shift targets to the right and add a <PAD> token at the beginning
            # Here, for simplicity, we can prepend a <PAD> token as the first input to the decoder
            # and remove the last token from the target
            tgt_input = torch.full((batch_size_actual, 1), char_to_idx['<PAD>'], dtype=torch.long).to(device)
            tgt_input = torch.cat([tgt_input, transcriptions.view(batch_size_actual, -1)[:, :-1]], dim=1)  # (batch_size, tgt_seq_len)

            # Ensure target sequences are padded to max_seq_length
            tgt_input_padded = torch.zeros(batch_size_actual, max_seq_length, dtype=torch.long).fill_(char_to_idx['<PAD>']).to(device)
            tgt_input_padded[:, :tgt_input.size(1)] = tgt_input

            tgt_output_padded = torch.zeros(batch_size_actual, max_seq_length, dtype=torch.long).fill_(char_to_idx['<PAD>']).to(device)
            tgt_output_padded[:, :transcriptions.size(0)//batch_size_actual] = transcriptions.view(batch_size_actual, -1)[:, :max_seq_length]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, tgt_input_padded)  # (tgt_seq_len, batch_size, num_classes)
            outputs = outputs.permute(1, 0, 2)  # (batch_size, tgt_seq_len, num_classes)

            # Reshape for loss computation
            outputs = outputs.contiguous().view(-1, num_classes)  # (batch_size * tgt_seq_len, num_classes)
            tgt_output = tgt_output_padded.contiguous().view(-1)  # (batch_size * tgt_seq_len)

            # Compute loss
            loss = criterion(outputs, tgt_output)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    print("Training Completed.")