import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    The feed forward layer is a component of the Transformer used to process each token independently
    Attention is used in the Transformer to capture relationships between Tokens, Feed Forward layer
    transformers each token's representation to make it richer and more informative _individually_

    Simple formula:
    Given x, the input vector for a token
    1. Expand it
        h = ReLU(xW1 + b1)
    2. Compress it
        y = hW2 + b2

    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1) -> None:
        """
        Initializes the feed-forward layer component

        :param d_model: Input & Output dimensionality of the model
        :param d_ffn: The hidden layer dimensionality of the FFN
        :param dropout: Dropout rate for regularization
        """
        super().__init__()

        # First fully connect layer
        self.fc1 = nn.Linear(d_model, d_ffn)

        # Non-linear activation function
        self.activation = nn.ReLU()

        # Second fully connected layer
        self.fc2 = nn.Linear(d_ffn, d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the feed-forward layer
        :param x: Input tensor of shape (batch, seq_length, d_model)
        :return: Output tensor of the same shape as the input
        """

        # Transform to higher dimension, apply ReLU, apply dropout, project back
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
