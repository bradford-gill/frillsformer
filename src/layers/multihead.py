"""Multihead attention"""

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,):
        super(MultiHeadAttention, self).__init__()
        raise NotImplementedError("The __init__ method is not implemented yet.")
        
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
        raise NotImplementedError("The forward method is not implemented yet.")
