import torch
import torch.nn as nn
import math
import numpy as np
from torch import Tensor
from .nn_utils import run_softmax

def get_batch(dataset, batch_size, context_length, device):
    max_idx = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_idx + 1, size=batch_size)
    x = np.stack([dataset[i:i+context_length] for i in start_indices])
    y = np.stack([dataset[i+1:i+context_length+1] for i in start_indices])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(2.0 / (in_features + out_features)))
    def forward(self, x): return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=1.0)
    def forward(self, ids): return self.weight[ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x):
        v = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x.to(torch.float32) * torch.rsqrt(v + self.eps) * self.weight).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    def forward(self, x):
        g = self.w1(x)
        return self.w2((g * torch.sigmoid(g)) * self.w3(x))

def apply_rope(x, theta, token_positions):
    d_k = x.size(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=x.device).float() / d_k))
    angles = token_positions.unsqueeze(-1).float() * freqs
    cos, sin = torch.cos(angles).repeat_interleave(2, -1), torch.sin(angles).repeat_interleave(2, -1)
    rotated = torch.empty_like(x)
    rotated[..., 0::2], rotated[..., 1::2] = -x[..., 1::2], x[..., 0::2]
    return x * cos + rotated * sin

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.num_heads, self.d_k = num_heads, d_model // num_heads
        self.q_proj, self.k_proj, self.v_proj, self.output_proj = [Linear(d_model, d_model, device, dtype) for _ in range(4)]
    def forward(self, x, mask=None, theta=None, token_positions=None):
        b, s, d = x.shape
        q, k, v = [proj(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2) for proj in [self.q_proj, self.k_proj, self.v_proj]]
        if theta is not None:
            pos = token_positions if token_positions is not None else torch.arange(s, device=x.device).expand(b, s)
            q, k = apply_rope(q, theta, pos.unsqueeze(-2)), apply_rope(k, theta, pos.unsqueeze(-2))
        if mask is None: mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=x.device))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = run_softmax(scores.masked_fill(~mask, float('-inf')), dim=-1)
        return self.output_proj((attn @ v).transpose(1, 2).contiguous().view(b, s, d))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.ln1, self.ln2, self.theta = RMSNorm(d_model, device=device, dtype=dtype), RMSNorm(d_model, device=device, dtype=dtype), theta
        self.attn, self.ffn = CausalSelfAttention(d_model, num_heads, device, dtype), SwiGLU(d_model, d_ff, device, dtype)
    def forward(self, x, mask=None, token_positions=None):
        x = x + self.attn(self.ln1(x), mask=mask, theta=self.theta, token_positions=token_positions)
        return x + self.ffn(self.ln2(x))

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype) for _ in range(num_layers)])
        self.ln_final, self.lm_head = RMSNorm(d_model, device=device, dtype=dtype), Linear(d_model, vocab_size, device, dtype)
    def forward(self, in_indices):
        b, s = in_indices.shape
        pos = torch.arange(s, device=in_indices.device).expand(b, s)
        x = self.token_embeddings(in_indices)
        for layer in self.layers: x = layer(x, token_positions=pos)
        return self.lm_head(self.ln_final(x))
