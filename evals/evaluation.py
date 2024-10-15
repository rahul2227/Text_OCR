import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data_loaders.data_loader import get_dataloader
from models.ocr_LSTM import device, OCRModel
from utils.utils import get_mappings, LSTM_MODEL_SAVE_PATH, get_project_root, set_torch_device


# Function to decode encoded transcription
def decode_transcription(encoded_seq, idx_to_char_mapping):
    """
    Decode a list of integer indices back to a string transcription.

    Args:
        encoded_seq (List[int] or int): Encoded transcription sequence or a single index.
        idx_to_char_mapping (Dict[int, str]): Mapping from index to character.

    Returns:
        str: Decoded transcription.
    """
    if isinstance(encoded_seq, int):
        return idx_to_char_mapping.get(encoded_seq, '<UNK>')
    return ''.join([idx_to_char_mapping.get(idx, '<UNK>') for idx in encoded_seq])


def evaluate(model, dataloader, idx_to_char, device, num_samples=5):
    """
    Evaluates the model's accuracy on the provided dataloader and collects sample predictions.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        idx_to_char (Dict[int, str]): Mapping from index to character.
        device (torch.device): Device to perform computation on.
        num_samples (int, optional): Number of sample predictions to collect. Defaults to 5.

    Returns:
        float: Evaluation accuracy in percentage.
        List[Tuple[torch.Tensor, str, str]]: List of tuples containing image tensor, predicted text, and target text.
    """
    model.eval()
    total = 0
    correct = 0
    samples_collected = 0
    sample_predictions = []

    with torch.no_grad():
        # Wrap the dataloader with tqdm for a progress bar
        for images, transcriptions, input_lengths, target_lengths in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            transcriptions = transcriptions.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)  # (batch_size, 'W', num_classes)
            outputs = outputs.permute(1, 0, 2).contiguous()  # ('W', batch_size, num_classes)
            outputs = torch.log_softmax(outputs, dim=2)

            # Decode predictions
            _, preds = torch.max(outputs, 2)  # (W', batch_size)

            preds = preds.transpose(0, 1).cpu().numpy()  # (batch_size, W')
            transcriptions = transcriptions.cpu().numpy()  # (sum(target_lengths))

            # Reconstruct target sequences based on target_lengths
            target_lengths = target_lengths.cpu().numpy()
            start_idx = 0
            for i, (pred, length) in enumerate(zip(preds, target_lengths)):
                end_idx = start_idx + length
                target_seq = transcriptions[start_idx:end_idx]
                start_idx = end_idx

                pred_text = decode_transcription(pred, idx_to_char).replace('<PAD>', '').replace('<UNK>', '')
                target_text = decode_transcription(target_seq, idx_to_char).replace('<PAD>', '').replace('<UNK>', '')

                if pred_text == target_text:
                    correct += 1
                total += 1

                # Collect sample predictions
                if samples_collected < num_samples:
                    image = images[i].cpu()
                    sample_predictions.append((image, pred_text, target_text))
                    samples_collected += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy, sample_predictions


def plot_sample_predictions(sample_predictions):
    """
    Plots sample predictions alongside their target transcriptions.

    Args:
        sample_predictions (List[Tuple[torch.Tensor, str, str]]): List of tuples containing image tensor, predicted text, and target text.
    """
    num_samples = len(sample_predictions)
    plt.figure(figsize=(15, 3 * num_samples))
    for idx, (image, pred, target) in enumerate(sample_predictions):
        image_np = image.numpy()
        # If the image has a single channel, squeeze it
        if image_np.shape[0] == 1:
            image_np = np.squeeze(image_np, axis=0)
            plt.imshow(image_np, cmap='gray')
        else:
            # If the image has multiple channels, transpose it for plotting
            image_np = np.transpose(image_np, (1, 2, 0))
            plt.imshow(image_np)

        plt.title(f"Sample {idx + 1}\nPredicted: {pred} | Target: {target}")
        plt.axis('off')
        plt.show()


def plot_accuracy(accuracy):
    """
    Plots the evaluation accuracy.

    Args:
        accuracy (float): The accuracy percentage to plot.
    """
    plt.figure(figsize=(6, 6))
    plt.bar(['Accuracy'], [accuracy], color='skyblue')
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Evaluation Accuracy')
    for index, value in enumerate([accuracy]):
        plt.text(index, value + 1, f"{value:.2f}%", ha='center')
    plt.show()


def load_model(model_class, fixed_height, fixed_width, num_classes, hidden_size, num_lstm_layers, dropout,
              model_save_path, device, preferred_model='ocr_lstm_model.pth', fallback_model='best_model_LSTM.pth'):
    """
    Loads the model weights from the preferred path if exists, else from the fallback path.

    Args:
        model_class (torch.nn.Module): The model class to instantiate.
        fixed_height (int): Fixed height for input images.
        fixed_width (int): Fixed width for input images.
        num_classes (int): Number of output classes.
        hidden_size (int): Hidden size for LSTM layers.
        num_lstm_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        model_save_path (str): Directory path where models are saved.
        device (torch.device): Device to load the model on.
        preferred_model (str, optional): Preferred model filename. Defaults to 'ocr_lstm_model.pth'.
        fallback_model (str, optional): Fallback model filename if preferred model doesn't exist. Defaults to 'best_model_LSTM.pth'.

    Returns:
        torch.nn.Module: Loaded model.
    """
    preferred_path = os.path.join(model_save_path, preferred_model)
    fallback_path = os.path.join(model_save_path, fallback_model)

    if os.path.exists(preferred_path):
        print(f"Loading model from {preferred_path}")
        model = model_class(
            fixed_height=fixed_height,
            fixed_width=fixed_width,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout
        ).to(device)
        # Handle FutureWarning by setting weights_only=True if supported
        try:
            model.load_state_dict(torch.load(preferred_path, map_location=device, weights_only=True))
        except TypeError:
            # If weights_only is not supported, load normally
            model.load_state_dict(torch.load(preferred_path, map_location=device))
        print(f"Successfully loaded {preferred_model}")
    elif os.path.exists(fallback_path):
        print(f"Preferred model {preferred_model} not found. Loading fallback model from {fallback_path}")
        model = model_class(
            fixed_height=fixed_height,
            fixed_width=fixed_width,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout
        ).to(device)
        # Handle FutureWarning by setting weights_only=True if supported
        try:
            model.load_state_dict(torch.load(fallback_path, map_location=device, weights_only=True))
        except TypeError:
            # If weights_only is not supported, load normally
            model.load_state_dict(torch.load(fallback_path, map_location=device))
        print(f"Successfully loaded {fallback_model}")
    else:
        raise FileNotFoundError(f"Neither '{preferred_model}' nor '{fallback_model}' found in '{model_save_path}'.")

    return model


def main_evaluation():
    # Load mappings
    print("Loading mappings...")
    mappings = get_mappings()

    fixed_height = mappings['fixed_height']
    fixed_width = mappings['max_width']  # e.g., 1024
    num_classes = len(mappings['char_to_idx'])  # Number of unique characters including <PAD> and <UNK>
    hidden_size = 256
    num_lstm_layers = 2
    dropout = 0.1

    # Determine the device (CPU, GPU, or MPS)
    device = set_torch_device()
    print(f"Using device: {device}")

    # Load the model with checkpoint checking
    try:
        model_LSTM = load_model(
            model_class=OCRModel,
            fixed_height=fixed_height,
            fixed_width=fixed_width,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            model_save_path=LSTM_MODEL_SAVE_PATH,
            device=device,
            preferred_model='ocr_lstm_model.pth',
            fallback_model='best_model_LSTM.pth'
        )
    except FileNotFoundError as e:
        print(e)
        return

    model_LSTM.eval()  # Set model to evaluation mode
    data_loader = get_dataloader(model_type='LSTM', batch_size=16)
    print("OCR LSTM Model loaded and set to evaluation mode.")

    # Evaluate the model and get sample predictions
    accuracy, sample_predictions = evaluate(model_LSTM, data_loader, mappings['idx_to_char'], device)

    # Plot the evaluation accuracy
    plot_accuracy(accuracy)

    # Plot sample predictions
    if sample_predictions:
        plot_sample_predictions(sample_predictions)
    else:
        print("No sample predictions to display.")


if __name__ == "__main__":
    main_evaluation()