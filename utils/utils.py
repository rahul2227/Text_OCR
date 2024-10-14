import json
import os
from pathlib import Path

import torch

IMAGES_PATH = os.path.join(Path(__file__).parent.parent, 'data_preprocessing/preprocessed_images.npy')
ENCODED_TRANSCRIPTION_PATH = os.path.join(Path(__file__).parent.parent, 'data_preprocessing/encoded_transcriptions.json')
LSTM_MODEL_SAVE_PATH = os.path.join(Path(__file__).parent.parent, 'models/LSTM')
TRANSFORMER_MODEL_SAVE_PATH = os.path.join(Path(__file__).parent.parent, 'models/TRANSFORMER')


# -------------------------------
# Get project root
# -------------------------------
def get_project_root() -> Path:
    return Path(__file__).parent.parent


# ------------------------------------
# Get mapping from preprocessed data
# ------------------------------------

def get_mappings():
    print('loading mappings')
    root_dir = get_project_root()
    mappings_path = os.path.join(root_dir, 'data_preprocessing/mappings.json')
    # Load mappings
    print("Loading mappings...")
    with open(mappings_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    return mappings


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