
import torch
import torch.nn as nn
import math

from models.feature_extractor import CNNFeatureExtractor


# -----------------------------------------------------
# Flatten and Reshape CNN output to create a Sequence
# -----------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Implements the sinusoidal positional encoding for transformers.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (sequence_length, batch_size, embedding_dim)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return x


# -----------------------------------------------------
# Transformer Encoder and Decoder layer
# -----------------------------------------------------


class TransformerOCR(nn.Module):
    def __init__(self,
                 img_feature_dim=512,    # From CNN Feature Extractor
                 img_feature_seq_len=256,  # Width after CNN pooling
                 d_model=512,            # Embedding dimension for transformer
                 nhead=8,                # Number of attention heads
                 num_encoder_layers=6,   # Number of transformer encoder layers
                 num_decoder_layers=6,   # Number of transformer decoder layers
                 dim_feedforward=2048,   # Feedforward network dimension
                 dropout=0.1,            # Dropout rate
                 num_classes=82,         # Number of output classes (characters)
                 max_seq_length=100):    # Maximum sequence length for output
        super(TransformerOCR, self).__init__()

        self.cnn = CNNFeatureExtractor()  # Reuse the CNN feature extractor

        # After CNN, feature map shape: (batch_size, 512, 8, 256)
        # We need to reshape it to (sequence_length, batch_size, embedding_dim)
        self.img_feature_dim = img_feature_dim
        self.img_feature_seq_len = img_feature_seq_len  # Typically the width after CNN pooling

        # Project CNN features to d_model
        self.fc = nn.Linear(img_feature_dim, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=self.img_feature_seq_len)

        # Transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)

        # Decoder: Assume target sequences have max length of max_seq_length
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.decoder_positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)

        # Final output layer
        self.output_layer = nn.Linear(d_model, num_classes)

        # Generate a tensor to mask future tokens in the decoder (for autoregressive generation)
        self.register_buffer("tgt_mask", self.generate_square_subsequent_mask(max_seq_length))

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # (sz, sz)

    def forward(self, src, tgt):
        """
        Args:
            src: Tensor of shape (batch_size, 1, H, W)
            tgt: Tensor of shape (batch_size, tgt_seq_len)
        Returns:
            Tensor of shape (tgt_seq_len, batch_size, num_classes)
        """
        # Feature Extraction
        x = self.cnn(src)  # (batch_size, 512, 8, 256)

        batch_size, channels, height, width = x.size()

        # Flatten and reshape
        x = x.permute(3, 0, 1, 2)  # (width, batch_size, channels, height)
        x = x.contiguous().view(width, batch_size, channels * height)  # (width, batch_size, 512 * 8 = 4096)

        # Project to d_model
        x = self.fc(x)  # (width, batch_size, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)  # (width, batch_size, d_model)

        # Encoder
        memory = self.transformer.encoder(x)  # (width, batch_size, d_model)

        # Prepare target
        tgt_seq_len = tgt.size(1)
        tgt = tgt.permute(1, 0)  # (tgt_seq_len, batch_size)
        tgt_emb = self.target_embedding(tgt)  # (tgt_seq_len, batch_size, d_model)
        tgt_emb = self.decoder_positional_encoding(tgt_emb)  # (tgt_seq_len, batch_size, d_model)

        # Decoder
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask=self.tgt_mask[:tgt_seq_len, :tgt_seq_len])  # (tgt_seq_len, batch_size, d_model)

        # Output Layer
        output = self.output_layer(output)  # (tgt_seq_len, batch_size, num_classes)

        return output