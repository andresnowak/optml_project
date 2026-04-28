from __future__ import annotations

import torch
from torch import nn


class LinearRegressionExperiment:
    def __init__(self, device, batch_size, feature_dim=32, samples=2048, **_):
        self.device = device
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.samples = samples
        self._cursor = 0
        self._inputs = torch.randn(samples, feature_dim, device=device)
        w = torch.randn(feature_dim, 1, device=device)
        self._targets = self._inputs @ w + 0.05 * torch.randn(samples, 1, device=device)

    def build_model(self):
        return nn.Linear(self.feature_dim, 1, bias=False).to(self.device)

    def next_batch(self):
        s, e = self._cursor, self._cursor + self.batch_size
        if e <= self.samples:
            x, y = self._inputs[s:e], self._targets[s:e]
        else:
            x = torch.cat([self._inputs[s:], self._inputs[:e - self.samples]])
            y = torch.cat([self._targets[s:], self._targets[:e - self.samples]])
        self._cursor = e % self.samples
        return x, y

    def loss(self, model, batch):
        x, y = batch
        return ((model(x) - y) ** 2).mean()


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
