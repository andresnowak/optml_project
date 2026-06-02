from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

import torch
import wandb

from src.matplotlib_backend import ScalarPlot, SVDPlot


class BaseLogger(ABC):
    @abstractmethod
    def log(self, metrics: dict, step: int) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...

    def start_run(self, name: str, config: dict | None = None, metric_prefix: str = "") -> None:
        pass


class MemoryLogger(BaseLogger):
    """In-memory sink for batch sweeps; tensor metrics (e.g. SVD logs) are dropped."""

    def __init__(self) -> None:
        self.history: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def log(self, metrics: dict, step: int) -> None:
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                continue
            self.history[name].append((step, float(value)))

    def finish(self) -> None:
        pass


class MatplotlibLogger(BaseLogger):
    def __init__(self, title: str, log_scale: bool = False, smooth: int = 1,
                 save_path: str | None = None, svd_top_k: int | None = None):
        self.title = title
        self.save_path = save_path
        self.log_scale = log_scale
        self.smooth = smooth
        self.svd_top_k = svd_top_k
        self._scalar_plots: dict[str, ScalarPlot] = {}
        self._svd_plots: dict[str, SVDPlot] = {}

    def log(self, metrics: dict, step: int) -> None:
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if name not in self._svd_plots:
                    self._svd_plots[name] = SVDPlot(name, self.svd_top_k)
                self._svd_plots[name].collect(value.cpu().numpy(), step)
            else:
                parts = name.rsplit("/", 1)
                metric, series = (parts[-1], parts[0]) if len(parts) > 1 else (name, "")
                if metric not in self._scalar_plots:
                    self._scalar_plots[metric] = ScalarPlot(metric, self.log_scale, self.smooth)
                self._scalar_plots[metric].collect(series, float(value), step)

    def finish(self) -> None:
        import matplotlib.pyplot as plt

        n_scalar = len(self._scalar_plots)
        n_svd = len(self._svd_plots)
        if n_scalar + n_svd == 0:
            return

        width_ratios = [6] * n_scalar + [7] * n_svd
        fig, axes = plt.subplots(
            1, n_scalar + n_svd,
            figsize=(sum(width_ratios), 5),
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes[0]
        ax_idx = 0

        for metric, plot in self._scalar_plots.items():
            title = self.title if n_scalar == 1 else f"{self.title} — {metric}"
            plot.render(axes[ax_idx], title)
            ax_idx += 1

        for plot in self._svd_plots.values():
            plot.render(axes[ax_idx], fig)
            ax_idx += 1

        if self.save_path:
            fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {self.save_path}")
        plt.show()


class WandbLogger(BaseLogger):
    def __init__(self, project: str, entity: str | None = None,
                 config: dict | None = None, svd_top_k: int | None = None):
        self._project = project
        self._entity = entity
        self._base_config = config or {}
        self.svd_top_k = svd_top_k
        self._run_active = False
        self._metric_prefix = ""

    def start_run(self, name: str, config: dict | None = None, metric_prefix: str = "") -> None:
        if self._run_active:
            wandb.finish()
        wandb.init(project=self._project, entity=self._entity, name=name,
                   config=config if config is not None else self._base_config)
        self._run_active = True
        self._metric_prefix = metric_prefix

    def log(self, metrics: dict, step: int) -> None:
        payload = {}
        for name, value in metrics.items():
            name = name.removeprefix(self._metric_prefix)
            if isinstance(value, torch.Tensor):
                svs = value.detach().float().cpu().numpy().reshape(-1)
                n_svs = len(svs) if self.svd_top_k is None else min(self.svd_top_k, len(svs))
                for i in range(n_svs):
                    payload[f"{name}/sigma_{i + 1}"] = float(svs[i])
                finite_svs = svs[np.isfinite(svs)]
                if finite_svs.size:
                    data_range = float(finite_svs.max() - finite_svs.min())
                    if not np.isfinite(data_range) or np.isclose(data_range, 0.0):
                        payload[f"{name}/spectrum"] = wandb.Histogram(finite_svs, num_bins=1)
                    else:
                        try:
                            payload[f"{name}/spectrum"] = wandb.Histogram(finite_svs)
                        except ValueError:
                            payload[f"{name}/spectrum"] = wandb.Histogram(finite_svs, num_bins=1)
            else:
                payload[name] = value
        wandb.log(payload, step=step)

    def finish(self) -> None:
        if self._run_active:
            wandb.finish()
            self._run_active = False


def make_logger(backend: str | None, title: str, **kwargs) -> BaseLogger | None:
    if backend == "matplotlib":
        return MatplotlibLogger(
            title=title,
            log_scale=kwargs.get("log_scale", False),
            smooth=kwargs.get("smooth", 1),
            save_path=kwargs.get("save_path"),
            svd_top_k=kwargs.get("svd_top_k"),
        )
    if backend == "wandb":
        return WandbLogger(
            project=kwargs["wandb_project"],
            entity=kwargs.get("wandb_entity"),
            config=kwargs.get("config", {}),
            svd_top_k=kwargs.get("svd_top_k"),
        )
    return None
