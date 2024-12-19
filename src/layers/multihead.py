"""Multihead attention"""

import torch
import torch.nn as nn

from src.layers.scaled_dot import ScaledDotProduct

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_k: int = 64,
        dim_v: int = 64,
        height: int = 8,
    ):
        super(MultiHeadAttention, self).__init__()

        self.height = height

        self.dim_k = dim_k
        self.dim_v = dim_v

        self.attn_heads = [
            ScaledDotProduct(dim_k=dim_k, dim_v=dim_v,)
            for _ in range(height)
        ]

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):
        """Forward layer on MHA

        Parameters
        ----------
        query : _type_
            _description_
        key : _type_
            _description_
        value : _type_
            _description_
        mask : _type_, optional
            _description_, by default None
        """
        
        batch_size, seq_length, _ = x.size()
        
        
        x = torch.concat([head(query, key, value) for head in self.attn_heads])
