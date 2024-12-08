import os
import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from jiwer import wer, cer

from data_loaders.data_loader import get_dataloader
from models.ocr_LSTM import device as device_lstm, OCRModel
from models.ocr_transformer import TransformerOCR  # Ensure this is correctly implemented
from utils.utils import (
    get_mappings,
    LSTM_MODEL_SAVE_PATH,
    TRANSFORMER_MODEL_SAVE_PATH,
    get_project_root,
    set_torch_device
)

# -------------------------------
# 1. Utility Functions
# -------------------------------

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

def evaluate(model, dataloader, idx_to_char, device, num_samples=5, model_type='LSTM'):
    """
    Evaluates the model's CER and WER on the provided dataloader and collects sample predictions.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        idx_to_char (Dict[int, str]): Mapping from index to character.
        device (torch.device): Device to perform computation on.
        num_samples (int, optional): Number of sample predictions to collect. Defaults to 5.
        model_type (str, optional): Type of the model ('LSTM' or 'Transformer'). Defaults to 'LSTM'.

    Returns:
        float: Evaluation CER in percentage.
        float: Evaluation WER in percentage.
        List[Tuple[torch.Tensor, str, str]]: List of tuples containing image tensor, predicted text, and target text.
    """
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    total = 0
    samples_collected = 0
    sample_predictions = []
    skipped_samples = 0  # Counter for skipped samples

    # Wrap the dataloader with tqdm for a progress bar
    with torch.no_grad():
        for batch_idx, (images, transcriptions, input_lengths, target_lengths) in enumerate(tqdm(
            dataloader, desc=f"Evaluating {model_type} Model", unit="batch"
        )):
            images = images.to(device)
            transcriptions = transcriptions.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)  # Output shape depends on the model
            if model_type == 'LSTM':
                outputs = outputs.permute(1, 0, 2).contiguous()  # (batch_size, W', num_classes)
                outputs = torch.log_softmax(outputs, dim=2)
                _, preds = torch.max(outputs, 2)  # (batch_size, W')
            elif model_type == 'Transformer':
                # Assuming TransformerOCR outputs logits of shape (batch_size, tgt_seq_len, num_classes)
                outputs = outputs.contiguous()
                outputs = torch.log_softmax(outputs, dim=2)
                _, preds = torch.max(outputs, 2)  # (batch_size, tgt_seq_len)
            else:
                raise ValueError("model_type must be either 'LSTM' or 'Transformer'.")

            preds = preds.cpu().numpy()  # (batch_size, seq_len)
            transcriptions = transcriptions.cpu().numpy()  # (sum(target_lengths))

            # Reconstruct target sequences based on target_lengths
            target_lengths = target_lengths.cpu().numpy()
            start_idx = 0
            for i, (pred, length) in enumerate(zip(preds, target_lengths)):
                end_idx = start_idx + length
                target_seq = transcriptions[start_idx:end_idx]
                start_idx = end_idx

                pred_text = decode_transcription(pred, idx_to_char).replace('<PAD>', '').replace('<UNK>', '').strip()
                target_text = decode_transcription(target_seq, idx_to_char).replace('<PAD>', '').replace('<UNK>', '').strip()

                # Debugging: Print some samples
                if batch_idx == 0 and i < 5:
                    print(f"\nSample {i + 1} in first batch:")
                    print(f"Predicted Text: '{pred_text}'")
                    print(f"Target Text:    '{target_text}'")
                    print(f"Target Length: {length}")
                    print("-" * 50)

                # Check if target_text is empty or contains only whitespace
                if not target_text:
                    skipped_samples += 1
                    continue  # Skip this sample

                # Compute CER and WER for the current sample
                sample_cer = cer(target_text, pred_text)
                sample_wer = wer(target_text, pred_text)

                total_cer += sample_cer
                total_wer += sample_wer
                total += 1

                # Collect sample predictions
                if samples_collected < num_samples:
                    image = images[i].cpu()
                    sample_predictions.append((image, pred_text, target_text))
                    samples_collected += 1

    # Compute average CER and WER
    average_cer = (total_cer / total) * 100 if total > 0 else 0
    average_wer = (total_wer / total) * 100 if total > 0 else 0
    print(f"\nEvaluation CER for {model_type} Model: {average_cer:.2f}%")
    print(f"Evaluation WER for {model_type} Model: {average_wer:.2f}%")
    print(f"Total Samples Evaluated: {total}")
    print(f"Total Samples Skipped (Empty References): {skipped_samples}")

    return average_cer, average_wer, sample_predictions

def plot_sample_predictions(sample_predictions, model_type='LSTM'):
    """
    Plots sample predictions alongside their target transcriptions.

    Args:
        sample_predictions (List[Tuple[torch.Tensor, str, str]]): List of tuples containing image tensor, predicted text, and target text.
        model_type (str, optional): Type of the model ('LSTM' or 'Transformer'). Defaults to 'LSTM'.
    """
    num_samples = len(sample_predictions)
    if num_samples == 0:
        print("No sample predictions to display.")
        return

    cols = 3  # Number of columns in the grid
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for idx, (image, pred, target) in enumerate(sample_predictions):
        plt.subplot(rows, cols, idx + 1)
        image_np = image.numpy()
        # If the image has a single channel, squeeze it
        if image_np.shape[0] == 1:
            image_np = np.squeeze(image_np, axis=0)
            plt.imshow(image_np, cmap='gray')
        else:
            # If the image has multiple channels, transpose it for plotting
            image_np = np.transpose(image_np, (1, 2, 0))
            plt.imshow(image_np)

        plt.title(f"{model_type} Sample {idx + 1}\nPredicted: {pred} | Target: {target}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_metrics(cer, wer, model_type='LSTM'):
    """
    Plots the evaluation metrics (CER and WER) for the model.

    Args:
        cer (float): Character Error Rate in percentage.
        wer (float): Word Error Rate in percentage.
        model_type (str, optional): Type of the model ('LSTM' or 'Transformer'). Defaults to 'LSTM'.
    """
    metrics = {'CER': cer, 'WER': wer}
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ['salmon', 'skyblue']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, max(values) * 1.2 if values else 100)
    plt.ylabel('Error Rate (%)')
    plt.title(f'{model_type} Model Evaluation Metrics')

    # Add metric labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom')

    plt.show()

# -------------------------------
# 2. Model Loading Functions
# -------------------------------

def load_lstm_model(model_class, fixed_height, fixed_width, num_classes, hidden_size, num_lstm_layers, dropout,
                   model_save_path, device, preferred_model='ocr_lstm_model.pth', fallback_model='best_model_LSTM.pth'):
    """
    Loads the LSTM model weights from the preferred path if exists, else from the fallback path.

    Args:
        model_class (torch.nn.Module): The LSTM model class to instantiate.
        fixed_height (int): Fixed height for input images.
        fixed_width (int): Fixed width for input images.
        num_classes (int): Number of output classes.
        hidden_size (int): Hidden size for LSTM layers.
        num_lstm_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        model_save_path (str): Directory path where LSTM models are saved.
        device (torch.device): Device to load the model on.
        preferred_model (str, optional): Preferred LSTM model filename. Defaults to 'ocr_lstm_model.pth'.
        fallback_model (str, optional): Fallback LSTM model filename if preferred model doesn't exist. Defaults to 'best_model_LSTM.pth'.

    Returns:
        torch.nn.Module: Loaded LSTM model.
    """
    preferred_path = os.path.join(model_save_path, preferred_model)
    fallback_path = os.path.join(model_save_path, fallback_model)

    if os.path.exists(preferred_path):
        print(f"Loading LSTM model from {preferred_path}")
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
            state_dict = torch.load(preferred_path, map_location=device, weights_only=True)
        except TypeError:
            # If weights_only is not supported, load normally
            state_dict = torch.load(preferred_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded {preferred_model}")
    elif os.path.exists(fallback_path):
        print(f"Preferred LSTM model '{preferred_model}' not found. Loading fallback model from '{fallback_path}'")
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
            state_dict = torch.load(fallback_path, map_location=device, weights_only=True)
        except TypeError:
            # If weights_only is not supported, load normally
            state_dict = torch.load(fallback_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded {fallback_model}")
    else:
        raise FileNotFoundError(f"Neither '{preferred_model}' nor '{fallback_model}' found in '{model_save_path}'.")

    return model

def load_transformer_model(model_class, fixed_height, fixed_width, num_classes, hidden_size, num_transformer_layers, dropout,
                          model_save_path, device, preferred_model='ocr_transformer_model.pth', fallback_model='best_model_transformer.pth'):
    """
    Loads the Transformer model weights from the preferred path if exists, else from the fallback path.

    Args:
        model_class (torch.nn.Module): The Transformer model class to instantiate.
        fixed_height (int): Fixed height for input images.
        fixed_width (int): Fixed width for input images.
        num_classes (int): Number of output classes.
        hidden_size (int): Hidden size for Transformer layers.
        num_transformer_layers (int): Number of Transformer layers.
        dropout (float): Dropout rate.
        model_save_path (str): Directory path where Transformer models are saved.
        device (torch.device): Device to load the model on.
        preferred_model (str, optional): Preferred Transformer model filename. Defaults to 'ocr_transformer_model.pth'.
        fallback_model (str, optional): Fallback Transformer model filename if preferred model doesn't exist. Defaults to 'best_model_transformer.pth'.

    Returns:
        torch.nn.Module: Loaded Transformer model.
    """
    preferred_path = os.path.join(model_save_path, preferred_model)
    fallback_path = os.path.join(model_save_path, fallback_model)

    if os.path.exists(preferred_path):
        print(f"Loading Transformer model from {preferred_path}")
        model = model_class(
            input_dim=fixed_width,  # Adjust based on Transformer model's requirements
            output_dim=num_classes,
            d_model=hidden_size,
            nhead=8,  # Example value; adjust based on your architecture
            num_encoder_layers=num_transformer_layers,
            num_decoder_layers=num_transformer_layers,
            dim_feedforward=2048,  # Example value; adjust based on your architecture
            dropout=dropout
        ).to(device)
        # Handle FutureWarning by setting weights_only=True if supported
        try:
            state_dict = torch.load(preferred_path, map_location=device, weights_only=True)
        except TypeError:
            # If weights_only is not supported, load normally
            state_dict = torch.load(preferred_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded {preferred_model}")
    elif os.path.exists(fallback_path):
        print(f"Preferred Transformer model '{preferred_model}' not found. Loading fallback model from '{fallback_path}'")
        model = model_class(
            input_dim=fixed_width,
            output_dim=num_classes,
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers=num_transformer_layers,
            num_decoder_layers=num_transformer_layers,
            dim_feedforward=2048,
            dropout=dropout
        ).to(device)
        # Handle FutureWarning by setting weights_only=True if supported
        try:
            state_dict = torch.load(fallback_path, map_location=device, weights_only=True)
        except TypeError:
            # If weights_only is not supported, load normally
            state_dict = torch.load(fallback_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded {fallback_model}")
    else:
        raise FileNotFoundError(f"Neither '{preferred_model}' nor '{fallback_model}' found in '{model_save_path}'.")

    return model

# -------------------------------
# 3. Command-Line Argument Parsing
# -------------------------------

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate OCR Models (LSTM or Transformer)')

    parser.add_argument('--model_type', type=str, choices=['LSTM', 'Transformer'], required=True,
                        help='Type of model to evaluate: LSTM or Transformer')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of sample predictions to visualize (default: 5)')

    return parser.parse_args()

# -------------------------------
# 4. Plotting Functions
# -------------------------------

def plot_sample_predictions(sample_predictions, model_type='LSTM'):
    """
    Plots sample predictions alongside their target transcriptions.

    Args:
        sample_predictions (List[Tuple[torch.Tensor, str, str]]): List of tuples containing image tensor, predicted text, and target text.
        model_type (str, optional): Type of the model ('LSTM' or 'Transformer'). Defaults to 'LSTM'.
    """
    num_samples = len(sample_predictions)
    if num_samples == 0:
        print("No sample predictions to display.")
        return

    cols = 3  # Number of columns in the grid
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for idx, (image, pred, target) in enumerate(sample_predictions):
        plt.subplot(rows, cols, idx + 1)
        image_np = image.numpy()
        # If the image has a single channel, squeeze it
        if image_np.shape[0] == 1:
            image_np = np.squeeze(image_np, axis=0)
            plt.imshow(image_np, cmap='gray')
        else:
            # If the image has multiple channels, transpose it for plotting
            image_np = np.transpose(image_np, (1, 2, 0))
            plt.imshow(image_np)

        plt.title(f"{model_type} Sample {idx + 1}\nPredicted: {pred} | Target: {target}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_metrics(cer, wer, model_type='LSTM'):
    """
    Plots the evaluation metrics (CER and WER) for the model.

    Args:
        cer (float): Character Error Rate in percentage.
        wer (float): Word Error Rate in percentage.
        model_type (str, optional): Type of the model ('LSTM' or 'Transformer'). Defaults to 'LSTM'.
    """
    metrics = {'CER': cer, 'WER': wer}
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ['salmon', 'skyblue']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, max(values) * 1.2 if values else 100)
    plt.ylabel('Error Rate (%)')
    plt.title(f'{model_type} Model Evaluation Metrics')

    # Add metric labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom')

    plt.show()

# -------------------------------
# 5. Main Evaluation Function
# -------------------------------

def main_evaluation():
    # Parse command-line arguments
    args = parse_arguments()
    model_type = args.model_type
    num_samples = args.num_samples

    # Load mappings
    print("Loading mappings...")
    if model_type == 'LSTM':
        mappings_LSTM = get_mappings(model_type)
    else:
        mappings_Transformer = get_mappings(model_type)

    # fixed_height = mappings['fixed_height']
    # fixed_width = mappings['max_width']  # e.g., 1024
    # num_classes = len(mappings['char_to_idx'])  # Number of unique characters including <PAD> and <UNK>
    hidden_size = 256
    num_lstm_layers = 2
    num_transformer_layers = 4  # Adjust based on your Transformer architecture
    dropout = 0.1

    # Determine the device (CPU, GPU, or MPS)
    device = set_torch_device()
    print(f"Using device: {device}")

    # Initialize variables
    model = None
    model_display_name = ''

    # Load the specified model
    if model_type == 'LSTM':
        try:
            model = load_lstm_model(
                model_class=OCRModel,
                fixed_height=mappings_LSTM['fixed_height'],
                fixed_width=mappings_LSTM['max_width'],
                num_classes=len(mappings_LSTM['char_to_idx']),
                hidden_size=hidden_size,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout,
                model_save_path=LSTM_MODEL_SAVE_PATH,
                device=device,
                preferred_model='ocr_lstm_model.pth',
                fallback_model='best_model_LSTM.pth'
            )
            model_display_name = 'LSTM'
        except FileNotFoundError as e:
            print(e)
            return
    elif model_type == 'Transformer':
        try:
            model = load_transformer_model(
                model_class=TransformerOCR,
                fixed_height=mappings_Transformer['fixed_height'],
                fixed_width=mappings_Transformer['max_width'],
                num_classes=len(mappings_Transformer['char_to_idx']),
                hidden_size=hidden_size,
                num_transformer_layers=num_transformer_layers,
                dropout=dropout,
                model_save_path=TRANSFORMER_MODEL_SAVE_PATH,  # Ensure this path is defined
                device=device,
                preferred_model='ocr_transformer_model.pth',
                fallback_model='best_model_transformer.pth'
            )
            model_display_name = 'Transformer'
        except FileNotFoundError as e:
            print(e)
            return
    else:
        print("Invalid model type selected. Choose either 'LSTM' or 'Transformer'.")
        return

    model.eval()  # Set model to evaluation mode
    print(f"{model_display_name} Model loaded and set to evaluation mode.")

    # Initialize evaluation data loader
    # Assuming both models can use the same dataloader
    data_loader = get_dataloader(model_type=model_type, batch_size=16)  # Adjust batch_size if needed
    print("DataLoader initialized for evaluation.")

    # Evaluate the model and get sample predictions
    if model_type == 'LSTM':
        cer_score, wer_score, sample_predictions = evaluate(
            model=model,
            dataloader=data_loader,
            idx_to_char=mappings_LSTM['idx_to_char'],
            device=device,
            num_samples=num_samples,
            model_type=model_display_name
        )
    elif model_type == 'Transformer':
        cer_score, wer_score, sample_predictions = evaluate(
            model=model,
            dataloader=data_loader,
            idx_to_char=mappings_Transformer['idx_to_char'],
            device=device,
            num_samples=num_samples,
            model_type=model_display_name
        )
    else:
        cer_score, wer_score, sample_predictions = 0.0, 0.0, []

    # Plot the evaluation metrics
    if model_type in ['LSTM', 'Transformer']:
        plot_metrics(cer=cer_score, wer=wer_score, model_type=model_display_name)
    else:
        print("No valid model type provided for plotting.")

    # Plot sample predictions
    if sample_predictions:
        print(f"\nSample Predictions for {model_display_name} Model:")
        plot_sample_predictions(sample_predictions, model_type=model_display_name)
    else:
        print("No sample predictions to display.")

# -------------------------------
# 6. Entry Point
# -------------------------------

if __name__ == "__main__":
    main_evaluation()