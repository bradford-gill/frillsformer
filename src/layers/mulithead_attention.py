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
        
        # Projections for self-attention (q, k, v all from x)
        self.qkv_proj = nn.Linear(input_dim, 3 * dim_model)
        
        # Projections for cross-attention (q from x, k/v from context)
        self.q_proj = nn.Linear(input_dim, dim_model)
        self.kv_proj = nn.Linear(input_dim, 2 * dim_model)
        
        self.o_proj = nn.Linear(dim_model, dim_model)
        self.scaled_dot_product = ScaledDotProduct(self.dim_k)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        # Initialize biases if they exist
        for module in [self.qkv_proj, self.q_proj, self.kv_proj, self.o_proj]:
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
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
            # Self-attention: q, k, v all from x
            qkv = self.qkv_proj(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
        else:
            # Cross-attention: q from x, k/v from context
            q = self.q_proj(x)
            kv = self.kv_proj(context)
            k, v = torch.chunk(kv, 2, dim=-1)
        
        # Reshape q, k, v to [batch_size, num_heads, seq_length, dim_k]
        def reshape(tensor, target_seq_length):
            return tensor.view(batch_size, target_seq_length, self.num_heads, self.dim_k).permute(0, 2, 1, 3)
        
        q = reshape(q, seq_length)
        k = reshape(k, k.size(1))  # Use k's sequence length from context
        v = reshape(v, v.size(1))  # Use v's sequence length from context
        
        # Compute scaled dot-product attention
        values, attn = self.scaled_dot_product(q, k, v, mask=mask)
        
        # Concatenate heads and project
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.dim_model)
        output = self.o_proj(values)
        
        return (output, attn) if return_attention else output


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Support use of different mask shapes, insure that mask has four dimensions
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


    while mask.ndim < 4:
        mask = mask.unsqueeze(1)
    return mask