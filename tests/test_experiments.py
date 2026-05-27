import pytest
import torch
from torch import nn

from src.experiments import EXPERIMENTS, BaseExperiment


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_all_experiments_inherit_base(name: str) -> None:
    assert issubclass(EXPERIMENTS[name], BaseExperiment)


def test_linear_regression_well_conditioned(cpu: torch.device) -> None:
    cls = EXPERIMENTS["linear_regression"]
    exp = cls(device=cpu, batch_size=8, feature_dim=4, output_dim=2, samples=16)
    model = exp.build_model()
    batch = exp.next_batch()
    loss = exp.loss(model, batch)
    assert isinstance(model, nn.Module)
    assert loss.dim() == 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_matrix_factorization_loss_is_scalar(cpu: torch.device) -> None:
    cls = EXPERIMENTS["matrix_factorization"]
    exp = cls(device=cpu, batch_size=1, matrix_rows=8, matrix_cols=8, rank=2)
    model = exp.build_model()
    batch = exp.next_batch()
    loss = exp.loss(model, batch)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_metric_default_is_empty(cpu: torch.device) -> None:
    cls = EXPERIMENTS["matrix_factorization"]
    exp = cls(device=cpu, batch_size=1, matrix_rows=4, matrix_cols=4, rank=2)
    model = exp.build_model()
    assert exp.metric(model) == {}
