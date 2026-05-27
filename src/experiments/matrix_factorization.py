from __future__ import annotations

import torch
from torch import nn

from .base import BaseExperiment


class _MFModel(nn.Module):
    def __init__(self, m: int, k: int, n: int):
        super().__init__()
        self.L = nn.Parameter(torch.randn(m, k) * 0.1)  # m x k
        self.R = nn.Parameter(torch.randn(k, n) * 0.1)  # k x n

    def forward(self) -> torch.Tensor:
        return self.L @ self.R  # m x n


class MatrixFactorizationExperiment(BaseExperiment):
    def __init__(self, device, batch_size, matrix_rows: int = 64, matrix_cols: int = 64, rank: int = 8):
        self.device = device
        self.matrix_rows = matrix_rows
        self.matrix_cols = matrix_cols
        self.rank = rank
        L = torch.randn(matrix_rows, rank, device=device)
        R = torch.randn(rank, matrix_cols, device=device)
        self._target = L @ R

    def build_model(self) -> nn.Module:
        return _MFModel(self.matrix_rows, self.rank, self.matrix_cols).to(self.device)

    def next_batch(self):
        return self._target

    def loss(self, model: nn.Module, batch) -> torch.Tensor:
        return ((model() - batch) ** 2).mean()
