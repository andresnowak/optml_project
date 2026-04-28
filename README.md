# Optimization Benchmark Project

Benchmarks optimizer convergence speed across controlled problems using PyTorch.
Compares `adam`, `adamw`, `sgd`, `muon` (via `torch.optim.Muon`), and `specmuon` (WIP).

Managed with `uv`.

## Setup

```bash
uv sync
```

## Experiments

| name | description |
|---|---|
| `linear_regression` | Synthetic linear dataset, MSE loss |
| `matrix_factorization` | Recover a low-rank matrix via two factor matrices |

## Modes

### Single run

```bash
uv run optml-bench --experiment linear_regression --optimizer muon --steps 300 --plot
```

### Compare all optimizers at a fixed lr

```bash
uv run optml-bench --experiment matrix_factorization --compare-all --lr 1e-2 --log-scale
```

### Sweep learning rates for one optimizer

```bash
uv run optml-bench --optimizer adam --sweep-lr --lr-min 1e-4 --lr-max 1.0 --lr-n 10 --log-scale
```

### Compare all optimizers each at their best lr

Sweeps the lr grid per optimizer and plots each one at the lr that achieved the lowest final loss.

```bash
uv run optml-bench --experiment linear_regression --compare-best-lr \
    --lr-min 1e-4 --lr-max 1.0 --lr-n 10 --steps 300 --log-scale
```

## Optimizer hyperparameters

| flag | applies to |
|---|---|
| `--lr` | all |
| `--weight-decay` | all except muon |
| `--momentum` | sgd, muon |
| `--beta1`, `--beta2` | adam, adamw |
| `--eps` | adam, adamw |
| `--ns-steps` | muon (Newton-Schulz iterations) |

## Plot options

| flag | effect |
|---|---|
| `--plot` | show plot after a single run |
| `--log-scale` | log y-axis |
| `--smooth N` | rolling-mean window of N steps |
| `--save-plot PATH` | save figure to file |

## Device

`--device auto` (default) picks `cuda` → `mps` → `cpu`. Pass `cpu`, `cuda`, or `mps` to override.

## Notes

- `Muon` is subclassed from `torch.optim.Muon` to allow custom `adjust_lr` shape-scaling functions (coming).
- `SpecMuon` is a placeholder; update rule is TBD.
