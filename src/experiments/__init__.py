# Optional metalcore import: enables a CPU fallback for `torch.linalg.svd` on
# MPS (Mac) backends. Not needed on cuda; absent in slim cluster images.
try:
    import metalcore
    metalcore.enable_pytorch_overrides(activations=False, embedding_bag=False, normalization=False, softmax=False, optimizers=False, linalg=True)
except ImportError:
    pass

from .base import BaseExperiment
from .linear_regression import LinearRegressionExperiment
from .matrix_factorization import MatrixFactorizationExperiment
from .shakespeare import ShakespeareExperiment


EXPERIMENTS: dict[str, type[BaseExperiment]] = {
    "linear_regression": LinearRegressionExperiment,
    "matrix_factorization": MatrixFactorizationExperiment,
    "shakespeare": ShakespeareExperiment,
}

__all__ = [
    "BaseExperiment",
    "EXPERIMENTS",
    "LinearRegressionExperiment",
    "MatrixFactorizationExperiment",
    "ShakespeareExperiment",
]
