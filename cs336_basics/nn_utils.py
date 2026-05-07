import torch
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Iterable

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_val = in_features.max(dim=dim, keepdim=True).values
    exp_val = torch.exp(in_features - max_val)
    return exp_val / exp_val.sum(dim=dim, keepdim=True)

def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    max_val = inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(inputs - max_val).sum(dim=-1, keepdim=True)) + max_val
    log_probs = inputs - log_sum_exp
    return -log_probs[torch.arange(inputs.shape[0], device=inputs.device), targets].mean()

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads: return
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for g in grads: g.detach().mul_(clip_coef)
