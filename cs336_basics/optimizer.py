import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad, state = p.grad.data, self.state[p]
                if len(state) == 0:
                    state['step'], state['exp_avg'], state['exp_avg_sq'] = 0, torch.zeros_like(p.data), torch.zeros_like(p.data)
                exp_avg, exp_avg_sq, beta1, beta2 = state['exp_avg'], state['exp_avg_sq'], group['betas'][0], group['betas'][1]
                state['step'] += 1
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

def get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters: return max_lr * (it / warmup_iters)
    if it >= cosine_cycle_iters: return min_lr
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)
