import torch
from models.ocr_LSTM import device
from utils.utils import set_torch_device


# -------------------------------
# 3. Define the Training Loop
# -------------------------------

def train_transformer(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, tgt_input, tgt_output) in enumerate(dataloader):
            images = images.to(device)        # (batch_size, 1, H, W)
            tgt_input = tgt_input.to(device)  # (batch_size, tgt_seq_len)
            tgt_output = tgt_output.to(device)  # (batch_size, tgt_seq_len)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, tgt_input)  # [tgt_seq_len, batch_size, output_dim]

            # Reshape outputs and targets for CrossEntropyLoss
            outputs = outputs.permute(1, 0, 2).contiguous()  # [batch_size, tgt_seq_len, num_classes]
            outputs = outputs.view(-1, outputs.size(-1))   # [batch_size * tgt_seq_len, num_classes]
            tgt_output = tgt_output.view(-1)              # [batch_size * tgt_seq_len]

            loss = criterion(outputs, tgt_output)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    print("Training Completed for Transformer.")