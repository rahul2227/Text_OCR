# TextOCR: Optical Character Recognition with LSTM and Transformer Models

Welcome to TextOCR, a comprehensive Optical Character Recognition (OCR) project that leverages both LSTM and Transformer architectures to transcribe handwritten text from images. This project is designed to handle the complexities of handwriting recognition, providing robust models for various OCR tasks.

Table of Contents

	1.	Project Overview
	2.	Features
	3.	Prerequisites
	4.	Installation
	5.	Dataset Preparation
	6.	Data Preprocessing
	7.	Training Models
	•	Training the LSTM Model
	•	Training the Transformer Model
	8.	Usage
	•	Training
	•	Loading and Evaluating Models
	9.	Troubleshooting
	10.	Acknowledgements

# Project Overview

TextOCR is an end-to-end OCR system that processes handwritten text images and transcribes them into machine-readable text. The project encompasses data preprocessing, model training, and evaluation phases, supporting both LSTM and Transformer-based architectures to cater to different OCR requirements.

# Features
- Dual Architecture Support: Train and deploy both LSTM and Transformer-based OCR models.
- Data Preprocessing Pipeline: Comprehensive scripts to preprocess and prepare data for training.
- Flexible Training Scripts: Command-line interface to train models with customizable parameters.
- MPS Fallback Support: Enable training on Apple’s Metal Performance Shaders (MPS) with fallback to CPU for unsupported operations.
- Modular Codebase: Organized code structure for easy maintenance and scalability.

# Prerequisites

Before you begin, ensure you have met the following requirements:

- Operating System: macOS (for MPS support), Linux, or Windows.
- Python Version: 3.8 or higher.
- Hardware: Apple Silicon (e.g., M1, M2) for MPS support (optional).
- CUDA-Compatible GPU: Optional, for faster training on non-Apple devices.

# Installation

1. Clone the Repository

```bash
git clone https://github.com/rahul2227/Text_OCR.git
cd TextOCR
```

2. Create a Virtual Environment

It’s recommended to use conda or venv to manage dependencies.

Using Conda:

```bash
conda create -n textocr python=3.12
conda activate textocr
```

Using venv:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: Ensure that you have the latest version of PyTorch installed, especially for MPS support.

#### For macOS with MPS support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

```
#### For CUDA-enabled GPUs
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

Adjust the PyTorch installation command based on your hardware and CUDA version.

# Dataset Preparation

1. Obtain the Dataset

For this project, we’ll assume you’re using the IAM Handwriting Database. Ensure you have the necessary permissions to use the dataset.


2. Directory Structure

Organize your dataset in the following structure:

```mermaid
TextOCR/
├── data/
│   ├── ascii/
│   │   └── lines.txt
│   └── lines/
│       ├── line01.png
│       ├── line02.png
│       └── ...
├── models/
├── data_loaders/
├── utils/
├── model_training.py
├── main.py
└── README.md

```

- data/ascii/lines.txt: Contains transcriptions and metadata.
- data/lines/: Contains line image files (e.g., .png).

Note: Ensure that the lines.txt file correctly maps each image to its transcription.

# Data Preprocessing

Before training the models, preprocess the data to ensure it’s in the correct format.

1. Preprocessing Script

The preprocessing involves:

- Parsing Transcriptions: Extracting text data from lines.txt.
- Mapping Images to Transcriptions: Ensuring each image has a corresponding transcription.
- Image Preprocessing: Converting images to grayscale, resizing, normalizing, and padding.
- Label Encoding: Creating mappings between characters and indices.

2. Running the Preprocessing

Use the provided data_preprocessing.py script to preprocess the data for both LSTM and Transformer models.

```bash
python data_preprocessing.py --data_dir ../data --LSTM --resize_image_smaller --smaller_max_width 2048

python data_preprocessing.py --data_dir ../data --transformer --resize_image_smaller --smaller_max_width 2048

```

Output Files:

- For LSTM:
  - preprocessed_images_LSTM.npy
  - encoded_transcriptions_LSTM.json
  - mappings_LSTM.json
- For Transformer:
  - preprocessed_images_Transformer.npy
  - encoded_transcriptions_Transformer.json
  - mappings_Transformer.json

# Training Models

Training the LSTM Model

The LSTM-based OCR model utilizes CTC (Connectionist Temporal Classification) Loss, suitable for sequence prediction tasks like OCR.

1. Training Command

```bash
python main.py --train_LSTM --num_epochs 10 --batch_size 16
```

2. Training Script Overview

main.py handles the training process based on command-line arguments. When --train_LSTM is specified, it:

- Loads the preprocessed data.
- Initializes the LSTM model.
- Defines the loss function and optimizer.
- Trains the model for the specified number of epochs.
- Saves the trained model.


Training the Transformer Model

The Transformer-based OCR model utilizes CrossEntropy Loss, suitable for sequence-to-sequence tasks.

1. Training Command

```bash
python main.py --train_transformer --num_epochs 10 --batch_size 16

```

2. Training Script Overview

When --train_transformer is specified, main.py:

- Loads the preprocessed data.
- Initializes the Transformer model.
- Defines the loss function and optimizer.
- Trains the model for the specified number of epochs.
- Saves the trained model.




## Usage

### Training

Use the main.py script to train either the LSTM or Transformer model.

Training the LSTM Model

```bash
python main.py --train_LSTM --num_epochs 10 --batch_size 16

```
Training the Transformer Model
```bash
python main.py --train_transformer --num_epochs 10 --batch_size 16

```

Optional Arguments:

- --save_model: Save the trained model (default: True).
- --load_model: Load an existing model for evaluation or further training (default: False).

Example with Optional Arguments:

```bash
python main.py --train_LSTM --num_epochs 10 --batch_size 16 --save_model True
```

Loading and Evaluating Models

After training, models are saved in the specified directories:

- LSTM Model: data_preprocessing/models/LSTM/ocr_lstm_model.pth
- Transformer Model: data_preprocessing/models/Transformer/transformer_ocr_model.pth


## Common Issues

1. CTC Loss Not Implemented on MPS

Error Message:

NotImplementedError: The operator 'aten::_ctc_loss' is not currently implemented for the MPS device.

Solution:

	•	Enable MPS fallback by setting the environment variable PYTORCH_ENABLE_MPS_FALLBACK=1 at the very beginning of your main.py.

Implementation:

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


2. Warnings About enable_nested_tensor

Warning Message:

UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)

Solution:

	•	Set batch_first=True in both TransformerEncoderLayer and TransformerDecoderLayer.


## Acknowledgements

- PyTorch: An open-source machine learning library.
- IAM Handwriting Database: For providing a comprehensive dataset for handwriting recognition.
