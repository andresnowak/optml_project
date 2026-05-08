# Optimization Benchmark Project

Benchmarks optimizer convergence speed across controlled problems using PyTorch.
Compares `adam`, `adamw`, `sgd`, `muon` (via `torch.optim.Muon`), and `specmuon`.

Managed with `uv`. Run `main.py` from the project root — no package install needed.

## Setup

```bash
uv sync
```

Optional `.env`:

```bash
WANDB_PROJECT=optml-bench
```

## Running

```bash
uv run python main.py [options]
```

## Experiments

| name | description |
|---|---|
| `linear_regression` | Synthetic linear dataset, MSE loss |
| `matrix_factorization` | Recover a low-rank matrix via two factor matrices |

## Modes

### Single run

```bash
uv run python main.py --experiment linear_regression --optimizer muon --steps 300
```

### Compare all optimizers at a fixed lr

```bash
uv run python main.py --experiment matrix_factorization --compare-all --lr 1e-2 --log-scale
```

### Sweep one hyperparameter for one optimizer

```bash
uv run python main.py --optimizer adam --sweep-lr --lr-min 1e-4 --lr-max 1.0 --lr-n 10 --log-scale
```

```bash
uv run python main.py --optimizer specmuon --sweep top_k=2,4,8 --log-scale
```

### Grid sweep multiple hyperparameters

```bash
uv run python main.py --optimizer specmuon \
  --sweep lr=1e-3,1e-2 \
  --sweep top_k=2,4,8 \
  --backend wandb
```

## Optimizer hyperparameters

| flag | applies to |
|---|---|
| `--lr` | all |
| `--weight-decay` | all except muon |
| `--momentum` | sgd, muon, specmuon |
| `--beta1`, `--beta2` | adam, adamw |
| `--eps` | adam, adamw, specmuon |
| `--ns-steps` | muon (Newton-Schulz iterations) |
| `--top-k` | specmuon (SAV singular directions, default 5) |
| `--sav-smooth` | specmuon (smoothing factor ξ, default 0.1) |

`--sweep PARAM=v1,v2,...` supports `lr`, `weight_decay`, `momentum`, `beta1`, `beta2`, `eps`, `ns_steps`, `top_k`, `sav_smooth`, and `adjust_lr_fn`.

Repeat `--sweep` to run a Cartesian-product grid sweep.

## Logging options

| flag | effect |
|---|---|
| `--backend matplotlib` | show plot after run |
| `--backend wandb` | log to Weights & Biases |
| `--log-scale` | log y-axis (matplotlib) |
| `--smooth N` | rolling-mean window of N steps |
| `--save-plot PATH` | save figure to file |
| `--log-grad-svd` | log singular values of parameter gradients |
| `--svd-every N` | SVD logging frequency (default: same as `--log-every`) |
| `--svd-top-k K` | only show top-k singular values |

`--wandb-project` defaults to `WANDB_PROJECT` from `.env`, then falls back to `optml-bench`.

## Device

`--device auto` (default) picks `cuda` → `mps` → `cpu`. Pass `cpu`, `cuda`, or `mps` to override.

## Optimizers

- **Muon** — subclassed from `torch.optim.Muon` for future shape-scaling extensions.
- **SpecMuon** — Muon with SAV (Scalar Auxiliary Variable) adaptive scaling for the top-k singular directions. The remaining directions receive the standard Muon update. [Paper](https://www.arxiv.org/abs/2602.16167)
