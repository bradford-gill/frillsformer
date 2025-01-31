import torch
import torch.nn as nn
from src.layers.add_norm import AddNorm
from src.layers.mulithead_attention import MultiheadAttention
from src.layers.feed_forward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initializes the Decoder Block.
        To visualize this Module see Figure 1 in the paper (see README)

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward network dimension.
            dropout (float): Dropout rate.
        """

        super(DecoderBlock, self).__init__()

        self.self_attention = MultiheadAttention(embed_dim, num_heads, dim_model=embed_dim)
        self.add_norm1 = AddNorm(embed_dim)

        self.cross_attention = MultiheadAttention(embed_dim, num_heads, dim_model=embed_dim)
        self.add_norm2 = AddNorm(embed_dim)

        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.add_norm3 = AddNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Decoder Block.

        Args:
            x (torch.Tensor): Input tensor (decoder input) of shape (batch_size, tgt_seq_length, embed_dim).
            encoder_output (torch.Tensor): Encoder's output of shape (batch_size, src_seq_length, embed_dim).
            tgt_mask (torch.Tensor): Mask for the target sequence (self-attention).
            memory_mask (torch.Tensor): Mask for the encoder output (cross-attention).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_seq_length, embed_dim).
        """

        # Self-attention and Add&Norm
        self_attn_output = self.self_attention(x, mask=tgt_mask)
        x = self.add_norm1(x, self_attn_output)

        # Cross-attention and Add&Norm
        cross_attn_output = self.cross_attention(x, context=encoder_output, mask=memory_mask)
        x = self.add_norm2(x, cross_attn_output)

        ff_output = self.feed_forward(x)
        x = self.add_norm3(x, ff_output)

        return x