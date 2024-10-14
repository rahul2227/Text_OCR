import json
import os
from pathlib import Path

import torch
from matplotlib.widgets import EllipseSelector


# -------------------------------
# Get project root
# -------------------------------
def get_project_root() -> Path:
    return Path(__file__).parent.parent

root_dir = get_project_root()
IMAGES_PATH_TRANSFORMER = os.path.join(root_dir, 'data_preprocessing/preprocessed_images_Transformer.npy')
ENCODED_TRANSCRIPTION_PATH_TRANSFORMER = os.path.join(root_dir, 'data_preprocessing/encoded_transcriptions_Transformer.json')
LSTM_MODEL_SAVE_PATH = os.path.join(root_dir, 'models/LSTM')
TRANSFORMER_MODEL_SAVE_PATH = os.path.join(root_dir, 'models/TRANSFORMER')
MAPPINGS_PATH_TRANSFORMER = os.path.join(root_dir, 'data_preprocessing/mappings_Transformer.json')
MAPPINGS_PATH_LSTM = os.path.join(root_dir, 'data_preprocessing/mappings_LSTM.json')

# ------------------------------------
# Get mapping from preprocessed data
# ------------------------------------

def get_mappings(model_type='LSTM'):
    print('loading mappings')

    if model_type == 'LSTM':
        mappings_path = MAPPINGS_PATH_LSTM
    else:
        mappings_path = MAPPINGS_PATH_TRANSFORMER

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