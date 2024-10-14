import os

import torch

from data_loaders.data_loader import dataloader
from models.ocr_LSTM import device, OCRModel
from utils.utils import get_mappings, LSTM_MODEL_SAVE_PATH


# Function to decode encoded transcription
def decode_transcription(encoded_seq, idx_to_char_mapping):
    """
    Decode a list of integer indices back to a string transcription.

    Args:
        encoded_seq (List[int]): Encoded transcription.
        idx_to_char_mapping (Dict[int, str]): Mapping from index to character.

    Returns:
        str: Decoded transcription.
    """
    return ''.join([idx_to_char_mapping.get(idx, '<UNK>') for idx in encoded_seq])

def evaluate(model, dataloader, idx_to_char):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, transcriptions, input_lengths, target_lengths in dataloader:
            images = images.to(device)
            transcriptions = transcriptions.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)  # (batch_size, 'W', num_classes)
            outputs = outputs.permute(1, 0, 2)  # ('W', batch_size, num_classes)
            outputs = torch.log_softmax(outputs, dim=2)

            # Decode predictions
            _, preds = torch.max(outputs, 2)  # (W', batch_size)

            preds = preds.transpose(0, 1).cpu().numpy()
            transcriptions = transcriptions.cpu().numpy()

            for pred, target in zip(preds, transcriptions):
                pred_text = decode_transcription(pred, idx_to_char).replace('<PAD>', '').replace('<UNK>', '')
                target_text = decode_transcription(target, idx_to_char).replace('<PAD>', '').replace('<UNK>', '')

                if pred_text == target_text:
                    correct += 1
                total += 1

    accuracy = (correct / total) * 100
    print(f"Evaluation Accuracy: {accuracy:.2f}%")


# Running the evaluations
# Evaluate the model_LSTM
mappings = get_mappings()

fixed_height = mappings['fixed_height']
fixed_width = mappings['max_width']  # 1024  # This should match the 'max_width' from preprocessing
num_classes = len(mappings['char_to_idx'])  # Number of unique characters including <PAD> and <UNK>
hidden_size = 256
num_lstm_layers = 2
dropout = 0.1

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
evaluate(model_LSTM, dataloader, mappings['idx_to_char'])