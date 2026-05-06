# Muon vs SpecMuon Assessment Guide

## Goal

This project should answer a practical question: for a given optimization problem, when is plain `muon` sufficient and when is `specmuon` worth the extra spectral adaptivity?

## Core difference

- `muon`: orthogonalizes matrix gradients and gives equalized direction updates. It is often strong when spectra are not extremely stiff.
- `specmuon`: keeps Muon geometry but adds SAV-based damping/adaptation on top singular directions. It is designed for stiff, multi-scale, or unstable spectral regimes.

## Problem taxonomy

### Prefer `muon` first (Muon-only assessment)

Use these when gradients are matrix-shaped but not dominated by severe spectral stiffness:

- Well-conditioned or moderately conditioned matrix objectives.
- Smooth training with stable loss decrease and limited oscillations.
- Cases where top singular values are not disproportionately larger than the rest.

Typical signs:

- Fast monotonic descent with `muon`.
- `specmuon` gives similar final loss but no clear stability gain.
- Gradient spectrum is not sharply concentrated on very few modes.

Recommended experiments here:

- `linear_regression`
- `matrix_factorization`
- `orthogonal_procrustes`

### Prefer `specmuon` (SpecMuon-focused assessment)

Use these when optimization is stiff, ill-conditioned, or dominated by a few high-energy singular modes:

- Ill-conditioned operators / anisotropic features.
- Sparse or masked reconstruction where dominant modes can cause overshoot.
- Multi-scale residual equations where a few modes are much harder than others.

Typical signs:

- `muon` oscillates, plateaus, or is highly sensitive to learning rate.
- `specmuon` keeps descent stable at larger or broader LR ranges.
- Top singular modes dominate gradient energy for long periods.

Recommended experiments here:

- `ill_conditioned_linear_regression`
- `matrix_completion`
- `sylvester_equation`

### Use both (`muon` + `specmuon`) and compare

If regime is unknown or mixed, run both plus baselines (`adamw`, `adam`, `sgd`) with LR sweeps and compare:

- Best final loss.
- Time/steps to a fixed target loss.
- Stability (oscillation/divergence rate).
- LR robustness (width of good-LR region).

`shakespeare` belongs here because it is a mixed-parameter deep model where matrix and non-matrix parameter groups behave differently.

## Experiment-by-experiment expectations

| experiment | primary focus | likely winner pattern |
|---|---|---|
| `linear_regression` | baseline convex matrix regression | `muon` often enough |
| `ill_conditioned_linear_regression` | anisotropic / stiff spectrum | `specmuon` should be more stable |
| `matrix_factorization` | nonconvex low-rank recovery | close race; compare LR robustness |
| `matrix_completion` | sparse observations, low-rank recovery | `specmuon` often better on stability |
| `sylvester_equation` | two-sided linear operator stiffness | `specmuon` favored on difficult settings |
| `orthogonal_procrustes` | geometric matrix fitting with regularization | `muon` often competitive |
| `shakespeare` | realistic deep mixed-parameter training | must benchmark both |

## Recommended evaluation protocol

1. Start with `--compare-best-lr` for each experiment.
2. Enable `--log-grad-svd` and inspect spectrum evolution.
3. Label each problem by spectral behavior:
   - Low stiffness / broad spectrum: favor `muon`.
   - High stiffness / top-mode dominance: favor `specmuon`.
4. Confirm with at least 3 seeds before concluding.

## Practical selection rules

- If you want maximum simplicity and spectra are mild: choose `muon`.
- If you see instability, sharp anisotropy, or strong LR sensitivity: move to `specmuon`.
- If compute budget is tight: use `muon` for screening, `specmuon` only for hard/stiff cases.
