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
    def __init__(self, size, rank):
        super().__init__()
        self.L = nn.Parameter(torch.randn(size, rank) * 0.1)
        self.R = nn.Parameter(torch.randn(size, rank) * 0.1)

    def forward(self):
        return self.L @ self.R.T


class MatrixFactorizationExperiment:
    def __init__(self, device, batch_size, matrix_size=64, rank=8, **_):
        self.device = device
        self.matrix_size = matrix_size
        self.rank = rank
        left = torch.randn(matrix_size, rank, device=device)
        right = torch.randn(matrix_size, rank, device=device)
        self._target = left @ right.T

    def build_model(self):
        return _MFModel(self.matrix_size, self.rank).to(self.device)

    def next_batch(self):
        return self._target

    def loss(self, model, batch):
        return ((model() - batch) ** 2).mean()


EXPERIMENTS = {
    "linear_regression": LinearRegressionExperiment,
    "matrix_factorization": MatrixFactorizationExperiment,
}
