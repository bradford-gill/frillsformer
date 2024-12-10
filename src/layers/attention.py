import torch
import torch.nn as nn
import torch.nn.functional as f

class Attention(nn.Module):
    def __init__(self, dim_k: int, dim_v: int):
        super(Attention, self).__init__()
        self.dim_k = dim_k
        
        # create linear layers
        self.linear_q = nn.Linear(dim_k, dim_k)
        self.linear_k = nn.Linear(dim_k, dim_k)
        self.linear_v = nn.Linear(dim_v, dim_v)
        
        
    def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
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
        
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        
        x = query * key.T 
        x = x / torch.sqrt(torch.tensor(self.dim_k, dtype=torch.float32))
        x = x * value 

        return f.softmax(x)
        
