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
    

class Embedding(torch.nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        # initialize weight matrix
        matrix = torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(matrix, 0, 1, -3, 3)

        self.weight = torch.nn.Parameter(matrix)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        # return torch.einsum("oi, ...i -> ...o", self.weight, x)
        return self.weight[token_ids]