
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
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
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
        x = x + self.pe[:x.size(0)]
        return x



# -----------------------------------------------------
# Transformer Encoder
# -----------------------------------------------------

# models/transformer_ocr.py (continued)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Transformer Encoder.

        Args:
            input_dim (int): The dimension of input features.
            d_model (int): The number of expected features in the encoder/decoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        """
        Args:
            src (Tensor): Tensor of shape [seq_len, batch_size, input_dim]

        Returns:
            Tensor: Encoded features of shape [seq_len, batch_size, d_model]
        """
        src = self.input_projection(src)  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


# -----------------------------------------------------
# Transformer Decoder
# -----------------------------------------------------

# models/transformer_ocr.py (continued)

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Transformer Decoder.

        Args:
            output_dim (int): The number of output classes (e.g., number of characters).
            d_model (int): The number of expected features in the encoder/decoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_layers (int): The number of sub-decoder-layers in the decoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TransformerDecoder, self).__init__()
        self.output_projection = nn.Embedding(output_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (Tensor): Target sequence tensor of shape [tgt_seq_len, batch_size]
            memory (Tensor): Encoder output tensor of shape [src_seq_len, batch_size, d_model]
            tgt_mask (Tensor, optional): Mask for the target sequence.
            memory_mask (Tensor, optional): Mask for the memory sequence.

        Returns:
            Tensor: Output logits of shape [tgt_seq_len, batch_size, output_dim]
        """
        tgt = self.output_projection(tgt)  # [tgt_seq_len, batch_size, d_model]
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        output = self.fc_out(output)  # [tgt_seq_len, batch_size, output_dim]
        return output

# -----------------------------------------------------
# Transformer Encoder and Decoder layer
# -----------------------------------------------------


# models/transformer_ocr.py (continued)

class TransformerOCR(nn.Module):
    def __init__(self,
                 input_dim=512,
                 output_dim=82,  # Number of classes including <PAD>, <UNK>, etc.
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        """
        Complete Transformer-based OCR Model.

        Args:
            input_dim (int): Dimension of input features from CNN.
            output_dim (int): Number of output classes.
            d_model (int): Dimension of model.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of Transformer encoder layers.
            num_decoder_layers (int): Number of Transformer decoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(TransformerOCR, self).__init__()
        self.encoder = TransformerEncoder(input_dim=input_dim, d_model=d_model, nhead=nhead,
                                         num_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
                                         dropout=dropout)
        self.decoder = TransformerDecoder(output_dim=output_dim, d_model=d_model, nhead=nhead,
                                         num_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                         dropout=dropout)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of the mask (sequence length).

        Returns:
            Tensor: Mask tensor of shape [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        """
        Forward pass of the Transformer OCR model.

        Args:
            src (Tensor): Source sequence tensor of shape [src_seq_len, batch_size, input_dim]
            tgt (Tensor): Target sequence tensor of shape [tgt_seq_len, batch_size]

        Returns:
            Tensor: Output logits of shape [tgt_seq_len, batch_size, output_dim]
        """
        memory = self.encoder(src)  # [src_seq_len, batch_size, d_model]
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)  # [tgt_seq_len, tgt_seq_len]
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return output