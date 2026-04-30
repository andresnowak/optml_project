from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


class ScalarPlot:
    """Accumulates scalar time-series for one metric and renders them as line plots."""

    def __init__(self, metric: str, log_scale: bool, smooth: int):
        self.metric = metric
        self.log_scale = log_scale
        self.smooth = smooth
        self._series: dict[str, list[tuple[int, float]]] = {}

    def collect(self, series: str, value: float, step: int) -> None:
        self._series.setdefault(series, []).append((step, value))

    def render(self, ax: plt.Axes, title: str) -> None:
        colors = plt.cm.tab10.colors
        sorted_series = sorted(self._series.items(), key=lambda kv: kv[1][-1][1])
        for i, (series, records) in enumerate(sorted_series):
            color = colors[i % len(colors)]
            steps = np.array([r[0] for r in records])
            values = np.array([r[1] for r in records])
            label = series or self.metric
            if self.smooth > 1:
                ax.plot(steps, values, color=color, alpha=0.15, linewidth=0.8)
                smoothed = np.convolve(values, np.ones(self.smooth) / self.smooth, mode="valid")
                x_sm = steps[self.smooth - 1:]
                ax.plot(x_sm, smoothed, label=label, color=color, linewidth=2)
                x_end, y_end = x_sm[-1], smoothed[-1]
            else:
                ax.plot(steps, values, label=label, color=color, linewidth=2)
                x_end, y_end = steps[-1], values[-1]
            ax.annotate(f"{y_end:.4f}", xy=(x_end, y_end), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        color=color, fontsize=8, fontweight="bold")
        if self.log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"{self.metric} (log)" if self.log_scale else self.metric)
        ax.set_title(title)
        if any(s for s, _ in sorted_series):
            ax.legend(loc="upper right", framealpha=0.9)
        style_ax(ax)


class SVDPlot:
    """Accumulates singular value time-series and renders them as line plots with a colorbar."""

    def __init__(self, name: str, svd_top_k: int | None):
        self.name = name
        self.svd_top_k = svd_top_k
        self._records: list[tuple[int, np.ndarray]] = []

    def collect(self, svs: np.ndarray, step: int) -> None:
        self._records.append((step, svs))

    def render(self, ax: plt.Axes, fig: plt.Figure) -> None:
        steps = np.array([r[0] for r in self._records])
        svs_matrix = np.stack([r[1] for r in self._records])
        n_svs = svs_matrix.shape[1] if self.svd_top_k is None else min(self.svd_top_k, svs_matrix.shape[1])
        cmap = plt.cm.viridis
        alpha = max(0.3, 1.0 - n_svs / 80)
        for j in range(n_svs):
            color = cmap(j / max(n_svs - 1, 1))
            ax.plot(steps, svs_matrix[:, j], color=color, linewidth=1.2, alpha=alpha)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_svs))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("Rank", fontsize=9)
        cb.set_ticks(np.linspace(1, n_svs, min(n_svs, 6), dtype=int))
        ax.set_xlabel("Step")
        ax.set_ylabel("Singular value")
        ax.set_title(f"Grad SVD — {self.name}")
        style_ax(ax)
