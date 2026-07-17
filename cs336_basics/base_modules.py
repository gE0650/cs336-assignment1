import torch
import einops
import math

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

        return self.weight[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        # initialize weight matrix
        weight = torch.ones((d_model), dtype=dtype, device=device)

        self.weight = torch.nn.Parameter(weight)
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_type = x.dtype
        x = x.to(torch.float32)
        rms = (einops.einsum(x, x, "... d_model, ... d_model -> ...") / x.shape[-1] + self.eps) ** 0.5
        inv_rms = 1 / rms
        x_scaled = einops.einsum(x, self.weight, inv_rms, "... d_model, d_model, ... -> ... d_model")
        return x_scaled.to(in_type)
    
class SwiGLU(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid1 = einops.einsum(self.w1.weight, x, "out in, ... in -> ... out")
        mid2 = mid1 * torch.sigmoid(mid1)
        mid3 = einops.einsum(self.w3.weight, x, "out in, ... in -> ... out")
        return einops.einsum(mid2 * mid3, self.w2.weight, "... in, out in -> ... out")
    

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        cos = torch.zeros((max_seq_len, d_k // 2), device=device)
        sin = torch.zeros((max_seq_len, d_k // 2), device=device)
        for i in range(max_seq_len):
            for j in range(d_k // 2):
                cos[i][j] = math.cos(i * (theta ** -((2 * j) / d_k)))
                sin[i][j] = math.sin(i * (theta ** -((2 * j) / d_k)))

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... seq_len (d_pair two) -> ... seq_len d_pair two", two=2)
        #x_odd = einops.rearrange(x[..., 0], "... d 1 -> ... (d 1)")
        #x_even = einops.rearrange(x[..., 1], "... d 1 -> ... (d 1)")
        x_odd = x[..., 0]
        x_even = x[..., 1]
        
        new_odd = x_odd * self.cos_cached[token_positions] - x_even * self.sin_cached[token_positions]
        new_even = x_odd * self.sin_cached[token_positions] + x_even * self.cos_cached[token_positions]
        
        out = torch.stack([new_odd, new_even], dim=-1)
        out = einops.rearrange(out, "... seq_len d_pair two -> ... seq_len (d_pair two)")
        return out
