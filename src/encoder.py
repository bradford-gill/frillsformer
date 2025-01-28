import torch
import torch.nn as nn
from src.blocks.encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initializes the Encoder.

        Args:
            num_layers (int): Number of encoder blocks.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward dimension.
            dropout (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).
            mask (torch.Tensor): Optional attention mask.

        Returns:
            torch.Tensor: Output tensor of the encoder.
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply a final layer normalization after all encoder blocks
        x = self.norm(x)
        return x