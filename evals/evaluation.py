import torch

from data_loaders.data_loader import dataloader
from models.ocr_LSTM import device, model


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
# Evaluate the model
evaluate(model, dataloader, idx_to_char)