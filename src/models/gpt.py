"""Track-3-inspired GPT model for optimizer studies.

The architecture follows the spirit of modded-nanogpt's optimization track:
separate q/k/v matrices, RMSNorm, RoPE, ReLU^2 MLP, and routing-friendly
parameter names. The output head is tied to the token embedding to keep memory
use down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_length: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    head_dim: int | None = None

    @property
    def block_size(self) -> int:
        return self.sequence_length

    @classmethod
    def gpt124m(cls) -> "GPTConfig":
        return cls()

    @classmethod
    def small(cls) -> "GPTConfig":
        return cls(sequence_length=256, vocab_size=50304, n_layer=4, n_head=4, n_embd=256)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.arange(x.size(1), dtype=torch.float32, device=x.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_dim = config.head_dim or config.n_embd // config.n_head
        assert config.n_head * head_dim == config.n_embd
        self.n_head = config.n_head
        self.head_dim = head_dim
        self.q = Linear(config.n_embd, config.n_embd)
        self.k = Linear(config.n_embd, config.n_embd)
        self.v = Linear(config.n_embd, config.n_embd)
        self.proj = Linear(config.n_embd, config.n_embd)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim)
        k = self.k(x).view(B, T, self.n_head, self.head_dim)
        v = self.v(x).view(B, T, self.n_head, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=self.head_dim ** -0.5,
            is_causal=True,
        ).transpose(1, 2)
        return self.proj(y.contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = Linear(config.n_embd, 4 * config.n_embd)
        self.proj = Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.fc(x).relu().square())


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if name.endswith("weight"):
                if "proj" in name and not name.startswith("embed"):
                    p.data.zero_()
                elif name.startswith("embed"):
                    # The head is tied to the embedding and fed RMS-normalized
                    # activations, so logits scale with the embedding row
                    # norms: std 1/sqrt(d) gives unit rows and an initial loss
                    # of ~ln(vocab) ~= 10.8 (std 1 gave ~760 and wasted the
                    # first ~1k steps crushing the logit scale).
                    p.data.normal_(std=self.config.n_embd ** -0.5)
                else:
                    p.data.normal_(std=math.sqrt(0.33 / p.size(-1)))
            elif name.endswith("bias"):
                p.data.zero_()
            elif name.endswith("gains"):
                p.data.fill_(1.0)
            else:
                raise ValueError(f"uninitialized parameter: {name}")

    def forward(self, idx: Tensor, targets: Tensor | None = None):
        assert idx.size(1) <= self.config.sequence_length
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = F.linear(self.norm(x), self.embed.weight).float()
        if targets is None:
            return logits[:, [-1], :], None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n
