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

        x = query * key.T  # Jack: can you check this is sound math?
        x = x / torch.sqrt(torch.tensor(self.dim_k, dtype=torch.float32))
        if x:
            # -9e15 to approx neg infinity -> 0s out in softmax -> (exp(-9e15) ≈ 0)
            x = x.masked_fill(mask == 0, -9e15)

        x = x * value
        x = f.softmax(x)

        return x
