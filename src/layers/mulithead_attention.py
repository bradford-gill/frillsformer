import torch
import torch.nn as nn

from src.layers.scaled_dot import ScaledDotProduct

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8, dim_model: int = 512):
        """
        Multihead Attention mechanism for Transformers.

        Args:
            input_dim (int): Dimension of input features.
            num_heads (int): Number of attention heads.
            dim_model (int): Dimension of the output model (must be divisible by num_heads).
        """
        
        super().__init__()
        
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_k = dim_model // num_heads
        
        assert dim_model % num_heads == 0, "Dim K must be divisible by the number of heads"
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * dim_model)
        self.o_proj = nn.Linear(dim_model, dim_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """init params using xavier uniform"""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            self.o_proj.bias.data.fill_(0)
        
    
    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None, return_attention: bool = False,):
        """
        Forward pass for Multihead Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            context (torch.Tensor): Context tensor ONLY for cross-attention. Defaults to None (self-attention).
            mask (torch.Tensor): Optional attention mask.
            return_attention (bool): Whether to return attention weights. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, dim_model).
            (Optional) torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_length, seq_length).
        """
        
        batch_size, seq_length, _ = x.size()

        if context is None:
            # In the event context is None, we have self-attention, so we can use the same input
            context = x
        
        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
           
        q = q.view(batch_size, seq_length, self.num_heads, self.dim_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, context.size(1), self.num_heads, self.dim_k).permute(0, 2, 1, 3)
        v = v.view(batch_size, context.size(1), self.num_heads, self.dim_k).permute(0, 2, 1, 3)

        
        # Determine value outputs
        values, attn = ScaledDotProduct(self.dim_k)(q, k, v, mask=mask)
        
        # [Batch, SeqLen, Head, Dims]
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.dim_model)
        
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