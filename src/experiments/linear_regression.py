from __future__ import annotations

import torch
from torch import nn

from .base import BaseExperiment


class _LinRegModel(nn.Module):
    def __init__(self, m: int, n: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.randn(m, n, device=device) * 0.01)  # (m, n)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.W @ X  # (m, N)


class LinearRegressionExperiment(BaseExperiment):
    """min_W 1/2 ||WX - Y||_F^2  with W∈R^{m×n}, X∈R^{n×N}, Y∈R^{m×N}.

    Gradient: ∇f(W) = (WX - Y) X^T
    """

    def __init__(self, device, batch_size, feature_dim: int = 32, output_dim: int = 16, samples: int = 2048):
        self.device = device
        self.feature_dim = feature_dim   # n
        self.output_dim = output_dim     # m
        self.samples = samples           # N
        W_true = torch.randn(output_dim, feature_dim, device=device)
        self._X = torch.randn(feature_dim, samples, device=device)
        self._Y = W_true @ self._X + 0.05 * torch.randn(output_dim, samples, device=device)

    def build_model(self) -> nn.Module:
        return _LinRegModel(self.output_dim, self.feature_dim, self.device)

    def next_batch(self):
        return self._X, self._Y

    def loss(self, model: nn.Module, batch) -> torch.Tensor:
        X, Y = batch
        return 0.5 * ((model(X) - Y) ** 2).mean()


class IllConditionedLinearRegressionExperiment(LinearRegressionExperiment):
    """Linear regression with a controlled spectrum for the design matrix."""

    def __init__(
        self,
        device,
        batch_size,
        feature_dim: int = 32,
        output_dim: int = 16,
        samples: int = 2048,
        condition_number: float = 1e4,
    ):
        self.device = device
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.samples = samples
        self.condition_number = condition_number

        if condition_number < 1.0:
            raise ValueError("condition_number must be at least 1.0")

        W_true = torch.randn(output_dim, feature_dim, device=device)
        q, _ = torch.linalg.qr(torch.randn(samples, feature_dim), mode="reduced")
        q = q.to(device)
        singular_values = torch.logspace(
            0.0,
            -torch.log10(torch.tensor(float(condition_number))).item(),
            feature_dim,
        ).to(device)
        self._X = singular_values.unsqueeze(1) * q.T * (samples ** 0.5)
        self._Y = W_true @ self._X + 0.05 * torch.randn(output_dim, samples, device=device)
