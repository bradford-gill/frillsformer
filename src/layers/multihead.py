"""Multihead attention"""

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_k: int, dim_v: int, height: int):
        super(MultiHeadAttention, self).__init__()
        
        self.height = height
        
        self.dim_k = dim_k
        self.dim_v = dim_v
        
        self.linear_k = nn.Linear()
        
        
        
        
    def forward(self, query, key, value, mask=None):
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
        
        
        
        
        
