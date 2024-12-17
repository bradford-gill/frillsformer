import torch
import torch.nn as nn

class AddNorm(nn.Module):
    """
    The AddNorm layer is a component of the Transformer used to stabilize and enhance learning.
    It comes the output of a sublayer (either Attention or FeedForward) with the original input
    then ensures that the combination is balanced.

    Simple Formula:
    Given x, the input vector & sublayer_output, the output from the current sublayer

    1. Add the input with the sublayer output
        resid = x + sublayer_output
    2. Normalize to balance the result
        y = LayerNorm(resid)
    """
    def __init__(self, embed_dim: int, eps: float = 1e-6) -> None:
        """_summary_

        Args:
            embed_dim (int): Size of the input embeddings
            eps (float, optional): Small constant to prevent div by zero. Defaults to 1e-6.
        """
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Process the input through the Add&Norm

        Args:
            x: Input tensor from the previous layer
            sublayer_output: Output tensor from the current sublayer

        Returns:
            torch.Tensor: Normalized tensor after adding x & sublayer_output
        """
        # Add residual connection and normalize
        return self.norm(x + sublayer_output)