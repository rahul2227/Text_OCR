import argparse
import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_loaders.data_loader import dataloader, OCRDataset, collate_fn  # Ensure this uses the corrected collate_fn
from model_training import train_LSTM
from models.ocr_LSTM import OCRModel
from models.ocr_transformer import TransformerOCR
from transformer_model_training import train_transformer
from utils.utils import get_mappings, IMAGES_PATH, ENCODED_TRANSCRIPTION_PATH, set_torch_device, \
    TRANSFORMER_MODEL_SAVE_PATH, LSTM_MODEL_SAVE_PATH

# -------------------------------
# Training the model_LSTM
# -------------------------------

device = set_torch_device()

# def train_LSTM(model, dataloader, criterion, optimizer, num_epochs=10):
#     model.train()
#
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for batch_idx, (images, transcriptions, input_lengths, target_lengths) in enumerate(dataloader):
#             images = images.to(device)  # (batch_size, 1, H, W)
#             transcriptions = transcriptions.to(device)  # (sum(target_lengths))
#             input_lengths = input_lengths.to(device)    # (batch_size)
#             target_lengths = target_lengths.to(device)  # (batch_size)
#
#             # Debugging: Print tensor shapes for the first batch of each epoch
#             if batch_idx == 0:
#                 print(f"Epoch {epoch+1}, Batch {batch_idx+1}")
#                 print(f"Images shape: {images.shape}")  # Expected: [batch_size, 1, 128, 5125]
#                 print(f"Transcriptions shape: {transcriptions.shape}")  # Expected: [sum(target_lengths)]
#                 print(f"Input lengths shape: {input_lengths.shape}")  # Expected: [batch_size]
#                 print(f"Target lengths shape: {target_lengths.shape}")  # Expected: [batch_size]")
#
#             optimizer.zero_grad()
#
#             # Forward pass
#             outputs = model(images)  # (T, N, C) where T = W', N = batch_size, C = num_classes
#
#             # CTC Loss expects (T, N, C)
#             loss = criterion(outputs, transcriptions, input_lengths, target_lengths)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#             if (batch_idx + 1) % 100 == 0:
#                 print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
#
#         avg_loss = epoch_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
#
#     print("Training Completed.")


# -------------------------------
# 4. Start Training
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch OCR Model')
    parser.add_argument('--train_LSTM', type=bool, default=False, help='train LSTM', required=False)
    parser.add_argument('--train_transformer', type=bool, default=False, help='train transformer', required=False)
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs', required=False)
    parser.add_argument('--batch_size', type=int, default=16, help='number of epochs', required=False)
    # arguments for model save and load
    parser.add_argument('--save_model', type=bool, help='save the current model', required=False, default=True)
    parser.add_argument('--load_model', type=bool, help='load the current model', required=False, default=False)

    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size

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

        # Start training
        train_LSTM(model_LSTM, dataloader, ctc_loss, optimizer, num_epochs=num_epochs)

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

        # Instantiate the model_LSTM
        model_transformer = TransformerOCR(
            img_feature_dim=512,  # From CNNFeatureExtractor
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

        print(model_transformer)

        # Define the loss function (CTC Loss is not suitable for Transformer)
        # Instead, use CrossEntropyLoss with appropriate padding and masking
        criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>']).to(device)

        # Define the optimizer
        learning_rate = 1e-4
        optimizer = optim.Adam(model_transformer.parameters(), lr=learning_rate)

        # Instantiate the Dataset and DataLoader
        dataset = OCRDataset(
            images_path= IMAGES_PATH,
            encoded_transcriptions_path=ENCODED_TRANSCRIPTION_PATH,
            transform=None  # No additional transforms needed as preprocessing is done
        )

        # batch_size = 16  # Adjust based on memory constraints
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        train_transformer(model_transformer, dataloader, criterion, optimizer, num_epochs=num_epochs, char_to_idx=char_to_idx, max_seq_length=max_seq_length, num_classes=num_classes)

        if args.save_model:
            # Save the trained model
            torch.save(model_transformer.state_dict(), os.path.join(TRANSFORMER_MODEL_SAVE_PATH, 'transformer_ocr_model.pth'))
            print("Model saved successfully.")

        if args.load_model:
            model_transformer = TransformerOCR(
                img_feature_dim=512,  # From CNNFeatureExtractor
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

            model_transformer.load_state_dict(torch.load(os.path.join(TRANSFORMER_MODEL_SAVE_PATH, 'transformer_ocr_model.pth')))
            model_transformer.eval()
            print("OCR Transformer Model loaded and set to evaluation mode.")

