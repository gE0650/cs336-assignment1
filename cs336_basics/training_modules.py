import torch
# import einops
# import math

def crossEntrophy(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    max_pre = torch.max(predicted, dim=-1, keepdim=True).values
    shifted_pre = predicted - max_pre
    log_sum_exp = torch.logsumexp(shifted_pre, dim=-1)
    loss = log_sum_exp - torch.gather(shifted_pre, -1, target[..., None])
    return torch.mean(loss)