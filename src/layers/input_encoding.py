import torch
import torch.nn as nn
import math
from torch import Tensor


class InputEncoding(nn.Module):
    """
    Because the model contains no recurrence or convolution, the input encoding
    must inject information about the relative or absolute position of the tokens in the sequence

    Followup Work:
        It would be interesting to investigate how different Encodings would perform with
        different test data. There is an external paper linked:
        See: Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning

    Formula:
        PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

    Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            embed_dim (int): Dimensionality of the token and the positional embeddings.
            max_seq_len (int): Max sequence length supported by the positional encoding.
    """
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int) -> None:
        super(InputEncoding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, embed_dim)

    def _generate_positional_encoding(self, max_seq_len: int, embed_dim: int) -> Tensor:
        """
        Generate the positional encodings for all positions and embedding dimensions.
        """
        pos = torch.arange(0, max_seq_len).unsqueeze(1)  # Shape: (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)) # Could precompute for slight speed up
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Combine token embeddings with positional encodings.
        Args:
            input_ids (Tensor): Tensor of token indices (batch_size, seq_len).
        Returns:
            Tensor: Encoded input (batch_size, seq_len, embed_dim).
        """
        token_embeds = self.token_embedding(input_ids)  # Shape: (batch_size, seq_len, embed_dim)
        position_embeds = self.positional_encoding[:, :token_embeds.size(1), :]  # Shape: (1, seq_len, embed_dim)
        return token_embeds + position_embeds