import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import matplotlib.pyplot as plt

from PIL import Image

# Import the corrected DataLoader and Model
from data_loaders.data_loader import dataloader  # Ensure this uses the corrected collate_fn
from models.ocr_LSTM import device


# -------------------------------
# Training the model_LSTM
# -------------------------------

def train_LSTM(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, transcriptions, input_lengths, target_lengths) in enumerate(dataloader):
            images = images.to(device)  # (batch_size, 1, H, W)
            transcriptions = transcriptions.to(device)  # (sum(target_lengths))
            input_lengths = input_lengths.to(device)    # (batch_size)
            target_lengths = target_lengths.to(device)  # (batch_size)

            # Debugging: Print tensor shapes for the first batch of each epoch
            if batch_idx == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}")
                print(f"Images shape: {images.shape}")  # Expected: [batch_size, 1, 128, 5125]
                print(f"Transcriptions shape: {transcriptions.shape}")  # Expected: [sum(target_lengths)]
                print(f"Input lengths shape: {input_lengths.shape}")  # Expected: [batch_size]
                print(f"Target lengths shape: {target_lengths.shape}")  # Expected: [batch_size]")

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # (T, N, C) where T = W', N = batch_size, C = num_classes

            # CTC Loss expects (T, N, C)
            loss = criterion(outputs, transcriptions, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    print("Training Completed.")