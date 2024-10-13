import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import os
import matplotlib.pyplot as plt

from PIL import Image

from utils.utils import get_project_root


# -------------------------------
# Data preparation for training
# -------------------------------
class OCRDataset(Dataset):
    def __init__(self, images_path, encoded_transcriptions_path, transform=None):
        self.images = np.load(images_path)
        with open(encoded_transcriptions_path, 'r', encoding='utf-8') as f:
            self.transcriptions = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx):
        image = self.images[idx]
        transcription = self.transcriptions[idx]

        # Inspect the shape of the image
        # print(f"Original image shape: {image.shape}")

        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)  # Now [H, W]
            # print(f"Squeezed image shape: {image.shape}")

        elif image.ndim == 4 and image.shape[-1] == 1:
            image = image.squeeze(-1).squeeze(-1)  # Now [H, W]
            # print(f"Squeezed image shape: {image.shape}")

        elif image.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Convert image to tensor and add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)  # [1, H, W]
        # print(f"Final image tensor shape: {image.shape}")

        # Ensure that image is 3D
        assert image.dim() == 3, f"Image tensor must be 3D, got {image.dim()}D"

        # Convert transcription to tensor
        transcription = torch.tensor(transcription, dtype=torch.long)

        return image, transcription

# -------------------------------
# For handling the batching appropriately for CTC loss
# -------------------------------

def collate_fn(batch):
    """
    Collate function to be used with DataLoader for CTC loss.
    """
    images, transcriptions = zip(*batch)
    images = torch.stack(images, 0)  # (batch_size, 1, H, W)

    # Calculate target lengths before concatenation
    target_lengths = torch.tensor([len(t) for t in transcriptions], dtype=torch.long)

    # Concatenate transcriptions into a single tensor
    transcriptions = torch.cat(transcriptions, 0)  # (sum(target_lengths))

    # Calculate input lengths based on CNN feature extractor's downsampling
    batch_size = images.size(0)
    cnn_output_width = images.size(3) // (2 ** 4)  # Assuming 4 max-pooling layers
    input_lengths = torch.full(size=(batch_size,), fill_value=cnn_output_width, dtype=torch.long)

    return images, transcriptions, input_lengths, target_lengths


# -------------------------------
# Creating Data Loader
# -------------------------------

# Instantiate the dataset
root_dir = get_project_root()
dataset = OCRDataset(images_path=os.path.join(root_dir, 'data_preprocessing/preprocessed_images.npy'),
                     encoded_transcriptions_path=os.path.join(root_dir, 'data_preprocessing/encoded_transcriptions.json'),
                     transform=None)

# Define batch size
batch_size = 16  # 32

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn)