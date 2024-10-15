import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import os
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
# For handling the batching appropriately
# -------------------------------

def collate_fn_lstm(batch):
    """
    Collate function to be used with DataLoader for LSTM (CTC Loss).
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


def collate_fn_transformer(batch):
    """
    Collate function to be used with DataLoader for Transformer (CrossEntropy Loss).
    Prepares target input and target output sequences.
    """
    images, transcriptions = zip(*batch)
    images = torch.stack(images, 0)  # (batch_size, 1, H, W)

    # Find the maximum target sequence length in the batch
    max_tgt_len = max([len(t) for t in transcriptions])

    # Prepare target sequences with padding
    padded_transcriptions = []
    for t in transcriptions:
        pad_length = max_tgt_len - len(t)
        padded_t = torch.cat(
            [t, torch.full((pad_length,), fill_value=0, dtype=torch.long)])  # Assuming <PAD> is index 0
        padded_transcriptions.append(padded_t)

    # Convert to tensor: (batch_size, max_tgt_len)
    padded_transcriptions = torch.stack(padded_transcriptions, 0)

    # Prepare decoder input by shifting right and adding <SOS> token
    # Assuming <SOS> is index 2 for Transformer
    sos_index = 2
    eos_index = 3
    tgt_input = torch.full((padded_transcriptions.size(0), 1), fill_value=sos_index, dtype=torch.long)
    tgt_input = torch.cat([tgt_input, padded_transcriptions[:, :-1]], dim=1)  # (batch_size, max_tgt_len)

    # Target output is the actual transcription
    tgt_output = padded_transcriptions  # (batch_size, max_tgt_len)

    return images, tgt_input, tgt_output


# -------------------------------
# Creating Data Loader
# -------------------------------

def get_dataloader(model_type='LSTM', batch_size=16):
    """
    Create DataLoader based on the model type.

    Args:
        model_type (str): 'LSTM' or 'Transformer'.
        batch_size (int): Batch size.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    # Instantiate the dataset
    root_dir = get_project_root()

    # Determine filenames based on model type
    preprocessed_images_filename = f'preprocessed_images_{model_type}.npy'
    encoded_transcriptions_filename = f'encoded_transcriptions_{model_type}.json'

    images_path = os.path.join(root_dir, 'data_preprocessing', preprocessed_images_filename)
    encoded_transcriptions_path = os.path.join(root_dir, 'data_preprocessing', encoded_transcriptions_filename)

    # Check if files exist
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Preprocessed images not found at {images_path}")
    if not os.path.exists(encoded_transcriptions_path):
        raise FileNotFoundError(f"Encoded transcriptions not found at {encoded_transcriptions_path}")

    dataset = OCRDataset(
        images_path=images_path,
        encoded_transcriptions_path=encoded_transcriptions_path,
        transform=None
    )

    # Select appropriate collate function
    if model_type == 'LSTM':
        collate_fn = collate_fn_lstm
    elif model_type == 'Transformer':
        collate_fn = collate_fn_transformer
    else:
        raise ValueError("model_type must be either 'LSTM' or 'Transformer'.")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return dataloader