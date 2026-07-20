import torch
from typing import Optional
from collections.abc import Callable, Iterable
# import einops
import math

def crossEntrophy(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    max_pre = torch.max(predicted, dim=-1, keepdim=True).values
    shifted_pre = predicted - max_pre
    log_sum_exp = torch.logsumexp(shifted_pre, dim=-1)
    loss = log_sum_exp - torch.gather(shifted_pre, -1, target[..., None])
    return torch.mean(loss)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay = 0, betas = (0.9, 0.999), eps = 1e-8):
        if lr < 0:
            raise ValueError("lr 不对")
        defaults = {"lr": lr,
                    "beta_1": betas[0],
                    "beta_2": betas[1],
                    "eps": eps,
                    "weight_decay": weight_decay,}

        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 1)
                lr_t = lr * (1 - beta_2 ** t) ** 0.5 / (1 - beta_1 ** t)

                p.data -= lr * weight_decay * p.data

                state["m"] = beta_1 * state.get("m", torch.zeros_like(p.data)) + (1 - beta_1) * grad
                state["v"] = beta_2 * state.get("v", torch.zeros_like(p.data)) + (1 - beta_2) * grad ** 2
                
                p.data -= lr_t * state["m"] / (state["v"] ** 0.5 + eps)

                state["t"] = t + 1

        return loss
    
def cosLRS(t, a_max, a_min, T_w, T_c) -> float:
    if t < T_w:
        return a_max * t / T_w
    elif t <= T_c:
        return a_min + 0.5 * (a_max - a_min) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
    else:
        return a_min
    
def gradClip(params: Iterable[torch.nn.Parameter], M: float):
    grad_sum = 0
    for param in params:
        if param.grad is not None:
            grad_sum += torch.sum(param.grad ** 2)
    if grad_sum ** 0.5 > M:
        factor = M / (grad_sum ** 0.5 + 1e-6)
        for param in params:
            if param.grad is None:
                continue
            param.grad *= factor