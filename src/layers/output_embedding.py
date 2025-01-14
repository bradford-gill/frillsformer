import torch
import torch.nn as nn
from torch import Tensor


class OutputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, tie_weights: nn.Embedding = None) -> None:

        """
        Initializes the OutputEmbedding layer.

        Args:
            vocab_size (int): Size of the vocabulary, see InputEncoding.
            embed_dim (int): Dimensionality of the embeddings.
            tie_weights (nn.Embedding, optional): If provided, ties the weights of the linear projection
                to the embedding weights for input tokens.
        """
        super(OutputEmbedding, self).__init__()
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
        if tie_weights is not None:
            """
            The embedding matrix for input tokens (nn.Embedding) is reused for the output linear layer.
            This significantly reduces the number of parameters in the model, especially for large vocabularies.
            
            See "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2016)
            """
            self.proj.weight = tie_weights.weight
    
    def forward(self, decoder_outputs: Tensor) -> Tensor:
        """
        Applies the linear transformation to map decoder outputs to vocabulary logits.

        Args:
            decoder_outputs (Tensor): Decoder output embeddings of shape (batch_size, seq_len, embed_dim).

        Returns:
            Tensor: Logits over the vocabulary of shape (batch_size, seq_len, vocab_size).
        """
        return self.proj(decoder_outputs)