import torch
import torch.nn as nn
from src.blocks.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initializes the Decoder.

        Args:
            num_layers (int): Number of decoder blocks.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward network dimension.
            dropout (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x:torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor, memory_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input tensor (decoder input) of shape (batch_size, tgt_seq_length, embed_dim)
            encoder_output (torch.Tensor): Encoder's output of shape (batch_size, src_seq_length, embed_dim)
            tgt_mask (torch.Tensor): Mask for target sequence
            memory_mask (torch.Tensor, optional): Mask for the encoder output

        Returns:
            torch.Tensor: Output tensor of the decoder.
        """

        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        
        x = self.norm(x)
        return x

