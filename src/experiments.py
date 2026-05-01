from __future__ import annotations

import os
import urllib.request

import torch
import torch.nn.functional as F
from torch import nn
import metalcore

metalcore.enable_pytorch_overrides(activations=False, embedding_bag=False, normalization=False, softmax=False, optimizers=False, linalg=True)


class _LinRegModel(nn.Module):
    def __init__(self, m, n, device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(m, n, device=device) * 0.01)  # (m, n)

    def forward(self, X):
        return self.W @ X  # (m, N)


class LinearRegressionExperiment:
    """min_W 1/2 ||WX - Y||_F^2  with W∈R^{m×n}, X∈R^{n×N}, Y∈R^{m×N}.

    Gradient: ∇f(W) = (WX - Y) X^T
    """

    def __init__(self, device, batch_size, feature_dim=32, output_dim=16, samples=2048, **_):
        self.device = device
        self.feature_dim = feature_dim   # n
        self.output_dim = output_dim     # m
        self.samples = samples           # N
        # X: (n, N),  Y: (m, N)
        W_true = torch.randn(output_dim, feature_dim, device=device)
        self._X = torch.randn(feature_dim, samples, device=device)
        self._Y = W_true @ self._X + 0.05 * torch.randn(output_dim, samples, device=device)

    def build_model(self):
        return _LinRegModel(self.output_dim, self.feature_dim, self.device)

    def next_batch(self):
        return self._X, self._Y

    def loss(self, model, batch):
        X, Y = batch
        return 0.5 * ((model(X) - Y) ** 2).mean()


class _MFModel(nn.Module):
    def __init__(self, m, k, n):
        super().__init__()
        self.L = nn.Parameter(torch.randn(m, k) * 0.1)  # m x k
        self.R = nn.Parameter(torch.randn(k, n) * 0.1)  # k x n

    def forward(self):
        return self.L @ self.R  # m x n


class MatrixFactorizationExperiment:
    def __init__(self, device, batch_size, matrix_rows=64, matrix_cols=64, rank=8, **_):
        self.device = device
        self.matrix_rows = matrix_rows
        self.matrix_cols = matrix_cols
        self.rank = rank
        L = torch.randn(matrix_rows, rank, device=device)   # m x k
        R = torch.randn(rank, matrix_cols, device=device)   # k x n
        self._target = L @ R                                 # m x n

    def build_model(self):
        return _MFModel(self.matrix_rows, self.rank, self.matrix_cols).to(self.device)

    def next_batch(self):
        return self._target

    def loss(self, model, batch):
        return ((model() - batch) ** 2).mean()


_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
_SHAKESPEARE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")


def _load_shakespeare() -> str:
    if not os.path.exists(_SHAKESPEARE_PATH):
        os.makedirs(os.path.dirname(_SHAKESPEARE_PATH), exist_ok=True)
        print("Downloading tinyshakespeare...")
        urllib.request.urlretrieve(_SHAKESPEARE_URL, _SHAKESPEARE_PATH)
    with open(_SHAKESPEARE_PATH) as f:
        return f.read()


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class _MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[_TransformerBlock(d_model, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        return self.head(self.ln_f(self.blocks(x)))


class ShakespeareExperiment:
    def __init__(self, device, batch_size, block_size=128, d_model=128, n_heads=4, n_layers=4, **_):
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        text = _load_shakespeare()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self._train = data[:n]

    def build_model(self):
        return _MiniGPT(self.vocab_size, self.d_model, self.n_heads, self.n_layers, self.block_size).to(self.device)

    def next_batch(self):
        ix = torch.randint(len(self._train) - self.block_size, (self.batch_size,))
        x = torch.stack([self._train[i:i + self.block_size] for i in ix]).to(self.device)
        y = torch.stack([self._train[i + 1:i + self.block_size + 1] for i in ix]).to(self.device)
        return x, y

    def loss(self, model, batch):
        x, y = batch
        logits = model(x)
        return F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))


EXPERIMENTS = {
    "linear_regression": LinearRegressionExperiment,
    "matrix_factorization": MatrixFactorizationExperiment,
    "shakespeare": ShakespeareExperiment,
}
