import torch
import einops

class Linear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        # initialize weight matrix
        weight = torch.zeros((out_features, in_features), dtype=dtype, device=device)
        std_dev = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(weight, 0, std_dev, -3 * std_dev, 3 * std_dev)

        self.weight = torch.nn.Parameter(weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # return torch.einsum("oi, ...i -> ...o", self.weight, x)
        return einops.einsum(self.weight, x, "out in, ... in -> ... out")