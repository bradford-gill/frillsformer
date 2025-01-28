from src.layers.mulithead_attention import MultiheadAttention
from src.layers.add_norm import AddNorm
from src.layers.feed_forward import FeedForward

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout=0.1):
        """
        Initializes the Encoder Block.
        To visualize this Module see Figure 1 in the paper (see README)


        Args:
            embed_dim (int): Embedding dimension (dim_model).
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward network dimension.
            dropout (float): Dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiheadAttention(embed_dim, num_heads, dim_model=embed_dim)
        self.add_norm_1 = AddNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.add_norm_2 = AddNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention + Add & Norm
        attn_output = self.self_attention(x, mask=mask)
        x = self.add_norm_1(x, attn_output)

        # Feedforward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.add_norm_2(x, ff_output)
        return x
    
