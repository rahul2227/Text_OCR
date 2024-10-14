import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable MPS fallback to CPU
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import argparse
import json


import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_loaders.data_loader import get_dataloader # Ensure this uses the corrected collate_fn
from model_training import train_LSTM
from models.ocr_LSTM import OCRModel
from models.ocr_transformer import TransformerOCR
from transformer_model_training import train_transformer
from utils.utils import get_mappings, set_torch_device, \
    TRANSFORMER_MODEL_SAVE_PATH, LSTM_MODEL_SAVE_PATH

# -------------------------------
# Training the model_LSTM
# -------------------------------

device = set_torch_device()

# -------------------------------
# 4. Start Training
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch OCR Model')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train_LSTM', type=bool, default=False, help='train LSTM')
    group.add_argument('--train_transformer', type=bool, default=False, help='train transformer')


    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs', required=False)
    parser.add_argument('--batch_size', type=int, default=16, help='number of epochs', required=False)
    # arguments for model save and load
    parser.add_argument('--save_model', type=bool, help='save the current model', required=False, default=True)
    parser.add_argument('--load_model', type=bool, help='load the current model', required=False, default=False)

    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Validate arguments
    if args.train_transformer and args.train_LSTM:
        raise ValueError("Cannot train both Transformer and LSTM simultaneously. Choose one.")
    if not args.train_transformer and not args.train_LSTM:
        raise ValueError("Specify at least one model to train: --train_transformer or --train_LSTM")


    # getting the mappings of the preprocessed data
    mappings = get_mappings()

    char_to_idx = mappings['char_to_idx']
    idx_to_char = {int(k): v for k, v in mappings['idx_to_char'].items()}  # Ensure keys are integers
    max_width = mappings['max_width']
    fixed_height = mappings['fixed_height']

    # defining the loss with ctc loss
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    if args.train_LSTM:
        # Define essential parameters
        # fixed_height = 128
        fixed_width = max_width  # 1024  # This should match the 'max_width' from preprocessing
        num_classes = len(char_to_idx)  # Number of unique characters including <PAD> and <UNK>
        hidden_size = 256
        num_lstm_layers = 2
        dropout = 0.1

        # Defining the LSTM model
        # Instantiate the model_LSTM
        model_LSTM = OCRModel(fixed_height=fixed_height,
                              fixed_width=fixed_width,
                              num_classes=num_classes,
                              hidden_size=hidden_size,
                              num_lstm_layers=num_lstm_layers,
                              dropout=dropout).to(device)

        print(model_LSTM)

        # Optimizer
        learning_rate = 1e-3

        optimizer = optim.Adam(model_LSTM.parameters(), lr=learning_rate)

        data_loader = get_dataloader(model_type='LSTM',batch_size = batch_size)

        # Start training
        train_LSTM(model_LSTM, data_loader, ctc_loss, optimizer, num_epochs=num_epochs)

        if args.save_model:
            # Save the trained model_LSTM
            torch.save(model_LSTM.state_dict(), os.path.join(LSTM_MODEL_SAVE_PATH, 'ocr_lstm_model.pth'))
            print("Model saved successfully.")

        if args.load_model:
            # To load the model_LSTM later
            model_LSTM = OCRModel(
                fixed_height=fixed_height,
                fixed_width=fixed_width,
                num_classes=num_classes,
                hidden_size=hidden_size,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout
            ).to(device)

            model_LSTM.load_state_dict(torch.load(os.path.join(LSTM_MODEL_SAVE_PATH, 'ocr_lstm_model.pth')))
            model_LSTM.eval()
            print("OCR LSTM Model loaded and set to evaluation mode.")

    if args.train_transformer:

        # Define essential parameters
        fixed_width = max_width  # From preprocessing
        num_classes = len(char_to_idx)  # Number of unique characters including <PAD> and <UNK>
        hidden_size = 512
        num_transformer_layers = 6
        nhead = 8
        dim_feedforward = 2048
        dropout = 0.1
        max_seq_length = 100  # Adjust based on maximum transcription length

        # Instantiate the model_transformer
        model_transformer = TransformerOCR(input_dim=512, output_dim=num_classes, d_model=hidden_size, nhead=nhead,
                                           num_encoder_layers=num_transformer_layers,
                                           num_decoder_layers=num_transformer_layers, dim_feedforward=dim_feedforward,
                                           dropout=dropout).to(device)

        print(model_transformer)

        # Define the loss function (CTC Loss is not suitable for Transformer)
        # Instead, use CrossEntropyLoss with appropriate padding and masking
        criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>']).to(device)

        # Define the optimizer
        learning_rate = 1e-4
        optimizer = optim.Adam(model_transformer.parameters(), lr=learning_rate)

        # Instantiate the Dataset and DataLoader
        # dataset = OCRDataset(
        #     images_path= IMAGES_PATH,
        #     encoded_transcriptions_path=ENCODED_TRANSCRIPTION_PATH,
        #     transform=None  # No additional transforms needed as preprocessing is done
        # )
        #
        # # batch_size = 16  # Adjust based on memory constraints
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     collate_fn=collate_fn
        # )

        dataloader = get_dataloader(model_type='Transformer', batch_size=batch_size)

        train_transformer(model_transformer, dataloader, criterion, optimizer, num_epochs=num_epochs)

        if args.save_model:
            # Save the trained model
            torch.save(model_transformer.state_dict(), os.path.join(TRANSFORMER_MODEL_SAVE_PATH, 'transformer_ocr_model.pth'))
            print("Model saved successfully.")

        if args.load_model:
            model_transformer = TransformerOCR(512, d_model=hidden_size, nhead=nhead,
                                               num_encoder_layers=num_transformer_layers,
                                               num_decoder_layers=num_transformer_layers,
                                               dim_feedforward=dim_feedforward, dropout=dropout).to(device)

            model_transformer.load_state_dict(torch.load(os.path.join(TRANSFORMER_MODEL_SAVE_PATH, 'transformer_ocr_model.pth')))
            model_transformer.eval()
            print("OCR Transformer Model loaded and set to evaluation mode.")

