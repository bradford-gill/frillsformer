import torch
import torch.nn as nn

from src.layers.scaled_dot import ScaledDotProduct

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8, dim_model: int = 512):
        super().__init__()
        
        self.num_heads = num_heads
        self.dim_model = dim_model
        
        self.dim_k = dim_model // num_heads
        
        assert dim_model % num_heads == 0, "Dim K must be divisible by the number of heads"
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*dim_model)
        
        self.o_proj = nn.Linear(dim_model, dim_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """init params using xavier uniform"""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_attention: bool = False,):
        batch_size, seq_length, _ = x.size()
        
        if mask is not None:
            mask = expand_mask(mask)
            
        # Separate Q, K, V from linear output
        qkv = self.qkv_proj(x)
        
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.dim_k)
        qkv = qkv.permute(0, 2, 1, 3) # batch, head, seq len, dims
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Determine value outputs
        values, attn = ScaledDotProduct(self.dim_k)(q, k, v, mask=mask)
        
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.dim_model)
        
        o = self.o_proj(values)
        
        return (o, attn) if return_attention else o
         
    
    
def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    """support use of different mask shapes, insure that mask has four dimensions

    Parameters
    ----------
    mask : torch.Tensor
        _description_

    Returns
    -------
    torch.Tensor
        shape -> batch_size, num_heads, seq length, seq length
    """
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional last two dims seq_length by seq_length"
    
    if mask.ndim == 3:
        # expand to 1, seq len, seq len
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        # expand to 1, num_heads, seq len, seq len
        mask = mask.unsqueeze(0)
    return mask