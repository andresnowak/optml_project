from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseExperiment(ABC):
    """Contract every benchmark experiment must satisfy.

    Per step the training loop calls ``next_batch()`` then ``loss(model, batch)``.
    At logging steps it calls ``metric(model)``; experiments that override it
    are expected to compute the metric on a held-out evaluation batch they
    own — never on the training batch.
    """

    @abstractmethod
    def build_model(self) -> nn.Module: ...

    @abstractmethod
    def next_batch(self): ...

    @abstractmethod
    def loss(self, model: nn.Module, batch) -> torch.Tensor: ...

    def metric(self, model: nn.Module) -> dict[str, float]:
        return {}
