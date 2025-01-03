from types import NoneType
import torch
import torch.nn as nn
import torch.nn.functional as f


class ScaledDotProduct(nn.Module):
    def __init__(self, dim_k: int):
        super(ScaledDotProduct, self).__init__()
        self.dim_k = dim_k

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """forward on scaled dot

        Parameters
        ----------
        query : torch.Tensor
        key : torch.Tensor
        value : torch.Tensor
        mask : torch.Tensor , optional
            by default None

        Returns
        -------
        torch.Tensor
            tensor of scaled dor product
        """

        x = torch.matmul(query, key.transpose(-2, -1))
        x = x / torch.sqrt(torch.tensor(self.dim_k, dtype=torch.float32))
        if mask is not None:
            # -9e15 to approx neg infinity -> 0s out in softmax -> (exp(-9e15) ≈ 0)
            x = x.masked_fill(mask == 0, -9e15)
        
        attn = f.softmax(x)
        x = torch.matmul(attn, value)

        return x, attn
