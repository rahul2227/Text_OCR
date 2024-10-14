import torch
from models.ocr_LSTM import device

# -------------------------------
# 3. Define the Training Loop
# -------------------------------

def train_transformer(model, dataloader, criterion, optimizer, char_to_idx, num_classes, max_seq_length, num_epochs=10, clip=1.0):
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (images, transcriptions, input_lengths, target_lengths) in enumerate(dataloader):
            images = images.to(device)  # (batch_size, 1, H, W)
            transcriptions = transcriptions.to(device)  # (sum(target_lengths))
            input_lengths = input_lengths.to(device)    # (batch_size)
            target_lengths = target_lengths.to(device)  # (batch_size)

            batch_size_actual = images.size(0)
            max_target_length = target_lengths.max().item()

            # Prepare target sequences for Transformer:
            # Shift targets to the right and add a <PAD> token at the beginning
            # Here, for simplicity, we can prepend a <PAD> token as the first input to the decoder
            # and remove the last token from the target
            tgt_input = torch.full((batch_size_actual, 1), char_to_idx['<PAD>'], dtype=torch.long).to(device)
            tgt_input = torch.cat([tgt_input, transcriptions.view(batch_size_actual, -1)[:, :-1]], dim=1)  # (batch_size, tgt_seq_len)

            # Ensure target sequences are padded to max_seq_length
            tgt_input_padded = torch.zeros(batch_size_actual, max_seq_length, dtype=torch.long).fill_(char_to_idx['<PAD>']).to(device)
            tgt_input_padded[:, :tgt_input.size(1)] = tgt_input

            tgt_output_padded = torch.zeros(batch_size_actual, max_seq_length, dtype=torch.long).fill_(char_to_idx['<PAD>']).to(device)
            tgt_output_padded[:, :transcriptions.size(0)//batch_size_actual] = transcriptions.view(batch_size_actual, -1)[:, :max_seq_length]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, tgt_input_padded)  # (tgt_seq_len, batch_size, num_classes)
            outputs = outputs.permute(1, 0, 2)  # (batch_size, tgt_seq_len, num_classes)

            # Reshape for loss computation
            outputs = outputs.contiguous().view(-1, num_classes)  # (batch_size * tgt_seq_len, num_classes)
            tgt_output = tgt_output_padded.contiguous().view(-1)  # (batch_size * tgt_seq_len)

            # Compute loss
            loss = criterion(outputs, tgt_output)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    print("Training Completed.")