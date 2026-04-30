from __future__ import annotations

import torch
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


EXPERIMENTS = {
    "linear_regression": LinearRegressionExperiment,
    "matrix_factorization": MatrixFactorizationExperiment,
}
