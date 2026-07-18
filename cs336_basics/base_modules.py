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
        positions = torch.arange(max_seq_len, device=device)
        dims = torch.arange(d_k // 2, device=device)
        thetas = (theta ** -(2 * dims / d_k))[None, :] * positions[:, None]
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)


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

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None
):
    weights = einops.einsum(queries, keys, "... q_len d_k, ... k_len d_k" \
                            " -> ... q_len k_len") / math.sqrt(keys.shape[-1])
    if mask is not None:
        weights[..., ~mask] = -math.inf
    weights = softmax(weights, dim=-1)
    output = einops.einsum(weights, values, "... q_len k_len, ... k_len d_v" \
    " -> ... q_len d_v")

    return output

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, self.d_k * num_heads)
        self.k_proj = Linear(d_model, self.d_k * num_heads)
        self.v_proj = Linear(d_model, self.d_k * num_heads)
        self.output_proj = Linear(self.d_k * num_heads, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        seq_len = x.shape[-2]
        q_positions = torch.arange(seq_len)[:, None]
        k_positions = torch.arange(seq_len)[None, :]
        mask = k_positions <= q_positions # presume x is on CPU

        Attn_res = []
        for i in range(self.num_heads):
            q = Q[..., i * self.d_k : (i + 1) * self.d_k]
            k = K[..., i * self.d_k : (i + 1) * self.d_k]
            v = V[..., i * self.d_k : (i + 1) * self.d_k]

            Attn_res.append(scaled_dot_product_attention(q, k, v, mask))

        output = torch.concat(Attn_res, dim=-1)
        return self.output_proj(output)

class MultiheadSelfAttentionWithRoPE(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.RoPE = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

        self.q_proj = Linear(d_model, self.d_k * num_heads)
        self.k_proj = Linear(d_model, self.d_k * num_heads)
        self.v_proj = Linear(d_model, self.d_k * num_heads)
        self.output_proj = Linear(self.d_k * num_heads, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        seq_len = x.shape[-2]
        q_positions = torch.arange(seq_len)[:, None]
        k_positions = torch.arange(seq_len)[None, :]
        mask = k_positions <= q_positions # presume x is on CPU

        Attn_res = []
        for i in range(self.num_heads):
            q = Q[..., i * self.d_k : (i + 1) * self.d_k]
            k = K[..., i * self.d_k : (i + 1) * self.d_k]
            v = V[..., i * self.d_k : (i + 1) * self.d_k]
            q = self.RoPE(q, token_positions)
            k = self.RoPE(k, token_positions)

            Attn_res.append(scaled_dot_product_attention(q, k, v, mask))

        output = torch.concat(Attn_res, dim=-1)
        return self.output_proj(output)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        seq_len = in_features.shape[-2]
        token_positions = torch.arange(0, seq_len)
        mid1 = in_features + self.attn(self.ln1(in_features), token_positions)
        return mid1 + self.ffn(self.ln2(mid1))
    

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.Sequential()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta))
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embeddings(x)
        attn_res = self.layers(embedded)
        final = self.lm_head(self.ln_final(attn_res))
        return final