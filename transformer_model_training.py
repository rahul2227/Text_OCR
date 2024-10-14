import os
import torch
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt  # Import matplotlib for plotting

from models.early_stopping import EarlyStopping  # Ensure this path is correct
from utils.utils import get_project_root, set_torch_device  # Ensure this function is correctly implemented


# -------------------------------
# 3. Define the Training Loop with Early Stopping, Nested tqdm, and Loss Curve Plotting
# -------------------------------

def train_transformer(model, dataloader, criterion, optimizer, num_epochs=10, patience=5, min_delta=0.0,
                     save_path='best_model_transformer.pth'):
    """
    Trains the Transformer-based OCR model using the provided dataloader, loss criterion, and optimizer.
    Implements early stopping based on training loss, uses nested tqdm progress bars, and plots the loss curve.

    Args:
        model (torch.nn.Module): The Transformer model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        min_delta (float, optional): Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.0.
        save_path (str, optional): Path to save the best model checkpoint. Defaults to 'best_model_transformer.pth'.
    """
    model.train()  # Set model to training mode

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=min_delta, path=save_path)

    # Initialize list to store training losses
    train_losses = []

    # Determine the device (CPU, GPU, or MPS)
    device = set_torch_device()  # Assuming this function returns the correct device
    model.to(device)

    # Initialize the outer tqdm progress bar for epochs
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        epoch_loss = 0.0

        # Initialize the inner tqdm progress bar for training batches
        batch_bar = tqdm(enumerate(dataloader, 1),  # Start enumeration at 1
                        total=len(dataloader),
                        desc=f"Epoch {epoch}/{num_epochs} - Training",
                        unit="batch",
                        leave=False)  # Set leave=False to remove the batch bar after epoch

        for batch_idx, (images, tgt_input, tgt_output) in batch_bar:
            # Move data to the specified device
            images = images.to(device)          # (batch_size, 1, H, W)
            tgt_input = tgt_input.to(device)    # (batch_size, tgt_seq_len)
            tgt_output = tgt_output.to(device)  # (batch_size, tgt_seq_len)

            # Debugging: Print tensor shapes for the first batch of each epoch
            if batch_idx == 1:
                print(f"\nEpoch {epoch}/{num_epochs} - First Batch Shapes:")
                print(f"Images shape: {images.shape}")         # Expected: [batch_size, 1, H, W]
                print(f"Tgt Input shape: {tgt_input.shape}")   # Expected: [batch_size, tgt_seq_len]
                print(f"Tgt Output shape: {tgt_output.shape}") # Expected: [batch_size, tgt_seq_len]")

            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            outputs = model(images, tgt_input)  # [tgt_seq_len, batch_size, output_dim]

            # Reshape outputs and targets for CrossEntropyLoss
            outputs = outputs.permute(1, 0, 2).contiguous()  # [batch_size, tgt_seq_len, num_classes]
            outputs = outputs.view(-1, outputs.size(-1))    # [batch_size * tgt_seq_len, num_classes]
            tgt_output = tgt_output.view(-1)               # [batch_size * tgt_seq_len]

            loss = criterion(outputs, tgt_output)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()  # Update model parameters

            epoch_loss += loss.item()  # Accumulate loss

            # Update the inner progress bar with the current loss
            batch_bar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(dataloader)  # Calculate average training loss for the epoch
        train_losses.append(avg_train_loss)            # Record the average loss

        # Update the outer progress bar with average training loss
        epoch_bar.set_postfix(train_loss=avg_train_loss)

        # Print epoch-level information
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}")

        # Call EarlyStopping with the average training loss
        early_stopping(avg_train_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    print("Training Completed.")

    # Plotting the training loss curve
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.xticks(epochs_range)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally, save the plot
    plot_save_path = os.path.join(get_project_root(), 'models', 'training_loss_curve_transformer.png')
    plt.savefig(plot_save_path)
    print(f"Training loss curve saved to {plot_save_path}")