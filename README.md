# Optimization Benchmark Project

Benchmarks optimizer convergence across controlled problems using PyTorch.
Compares `adam`, `adamw`, `sgd`, `muon`, and **`specmuon`** — Muon with the
Scalar Auxiliary Variable (SAV) mechanism from
*Muon with Spectral Guidance* (Lu, Zhang & Lin, arXiv:2602.16167).

Managed with `uv`. Run `main.py` from the project root — no package install needed.

## Setup

```bash
uv sync
```

Optional `.env`:

```bash
WANDB_PROJECT=<wandb-project>
WANDB_ENTITY=<wandb-entity>
```

## Running

```bash
uv run python main.py [options]
```

YAML configs in `configs/` cover the canonical setups:

```bash
uv run python main.py --config configs/matrix_factorization.yaml
```

`extends:` is supported; CLI flags override anything in the config.

## Experiments

| name | description |
|---|---|
| `linear_regression` | Synthetic linear dataset, MSE loss |
| `ill_conditioned_linear_regression` | Linear regression with controlled condition number |
| `matrix_factorization` | Recover a low-rank matrix via two factor matrices |
| `shakespeare` | Char-level mini-GPT on tinyshakespeare |

`src/experiments/` holds one file per problem. Each experiment subclasses
`BaseExperiment` (`src/experiments/base.py`) and may implement an optional
`metric(model)` hook for held-out evaluation logging.

## Modes

### Single run

```bash
uv run python main.py --experiment linear_regression --optimizer muon --steps 300
```

### Compare all optimizers at a fixed lr

```bash
uv run python main.py --experiment matrix_factorization --compare-all --lr 1e-2 --log-scale
```

### Per-optimizer best-lr comparison

Runs `--lr-n` log-spaced learning rates for each optimizer (silent), then
re-runs each at its winning lr through the configured logger:

```bash
uv run python main.py --experiment matrix_factorization --compare-best-lr --lr-min 1e-4 --lr-max 1
```

### Sweep one or more hyperparameters

```bash
uv run python main.py --optimizer adam --sweep-lr --lr-min 1e-4 --lr-max 1.0 --lr-n 10 --log-scale
uv run python main.py --optimizer specmuon --sweep top_k=2,4,8 --log-scale
```

Repeat `--sweep` for a Cartesian-product grid sweep:

```bash
uv run python main.py --optimizer specmuon \
  --sweep sigma_mode=baseline,clip \
  --sweep gate_threshold=0,0.05,0.1 \
  --backend wandb
```

Shakespeare SpecMuon tail/momentum ablation:

```bash
uv run python main.py --config configs/shakespeare_specmuon_tail_ablation.yaml \
  --sweep top_k=0,1,8,16,32 \
  --sweep tail_mode=gradient,muon \
  --sweep momentum_mode=post_spectral,post_spectral_nesterov \
  --sweep min_lr=0,1e-5 \
  --sweep scheduler=linear,cosine
```

## Optimizer hyperparameters

| flag | applies to |
|---|---|
| `--lr` | all |
| `--min-lr` | all (minimum LR for `linear`/`cosine` schedulers; default 0) |
| `--weight-decay` | all except muon, specmuon |
| `--scheduler` | all (`linear` default, `cosine`, or `none`; torch LR scheduler stepped once per train step) |
| `--momentum` | sgd, muon, specmuon |
| `--beta1`, `--beta2` | adam, adamw |
| `--eps` | adam, adamw, specmuon |
| `--ns-steps` | muon (Newton-Schulz iterations) |
| `--adjust-lr-fn shape_scaling` | muon, specmuon (Keller/Jordan `sqrt(max(1, m/n))`) |
| `--top-k` | specmuon (SAV singular directions, default 6 — paper-faithful) |
| `--sav-smooth` | specmuon (smoothing factor ξ, default 0.2) |
| `--kappa` | specmuon (loss-shift constant κ ≥ 0, paper §2.1) |
| `--sigma-mode` | specmuon (`baseline`/`sqrt`/`power`/`clip`/`truncate`/`energy`) |
| `--power-beta` | specmuon (`power` mode: η/(σ^β+ε); β=1 ≡ baseline, β=0.5 ≡ sqrt) |
| `--sigma-clip` | specmuon (`clip` mode: η/(σ+sigma_clip)) |
| `--sigma-truncate` | specmuon (`truncate` mode: drop σ < threshold·σ_max) |
| `--energy-threshold` | specmuon (`energy` mode: dynamic k by cumulative σ² ≥ τ) |
| `--gate-window`, `--gate-threshold` | specmuon (SAV-gating; τ=0 disables — paper default) |
| `--tail-mode` | specmuon tail update after top-k (`gradient` = paper `U diag(S) V^T`, `muon` = `U V^T`) |
| `--momentum-mode` | specmuon momentum placement (`post_spectral`, `post_spectral_nesterov`, `pre_svd`, `pre_svd_nesterov`) |
| `--specmuon-target` | shakespeare layer-selectivity ablation (`all`, `mlp`, `attention`) |

`--sweep PARAM=v1,v2,...` accepts all of the above keyword names (e.g.
`--sweep sigma_mode=baseline,clip`).

## Logging options

| flag | effect |
|---|---|
| `--backend matplotlib` | show plot after run |
| `--backend wandb` | log to Weights & Biases |
| `--log-scale` | log y-axis (matplotlib) |
| `--smooth N` | rolling-mean window of N steps |
| `--save-plot PATH` | save figure to file |
| `--log-grad-svd` | log singular values of parameter gradients |
| `--log-sav-r` | log SpecMuon SAV diagnostics: top-k `sav_r`, `sav_sigma_scale`, `sav_iota`, `sav_iota_w`, `sav_sigma_min`, `sav_sigma_max` |
| `--log-grad-norms` | log per-parameter gradient norms + total |
| `--log-weight-norms` | log per-parameter weight norms |
| `--svd-every N` | SVD logging frequency (default: same as `--log-every`) |
| `--svd-top-k K` | only show top-k singular values |

`--wandb-project` and `--wandb-entity` default to `WANDB_PROJECT` / `WANDB_ENTITY` from `.env`, then fall back to `mlo-specmuon` / `cs-439-project`.
`--wandb-group` optionally groups related runs in the W&B UI; it defaults to
`WANDB_GROUP` if set.
WandB runs also receive the resolved CLI/config settings, optimizer kwargs,
experiment kwargs, run tag, SpecMuon target, and model parameter count in the
run config.

## RunAI cluster workflow

Sync the local checkout to the RunAI submit host, then submit from there:

```bash
./scripts/sync_to_rcp.sh
```

```bash
CONFIG=configs/shakespeare_specmuon_tail_ablation.yaml \
OPTIMIZER=specmuon \
BACKEND=wandb \
LOG_SAV_R=1 \
WANDB_PROJECT=mlo-specmuon-shakespeare \
WANDB_ENTITY=<wandb-entity> \
WANDB_GROUP=tail_ablation_20260607 \
SWEEP="top_k=0,1,8,16,32 tail_mode=gradient,muon momentum_mode=post_spectral,post_spectral_nesterov" \
./scripts/run_job.sh sweep
```

`scripts/run_job.sh` supports these subcommands:

```text
single, sanity, sweep, compare-best-lr, sav-isolation, gate-sensitivity,
selective-sav, shakespeare-long, interactive, logs, delete, list
```

RunAI / cluster environment variables:

| env var | effect |
|---|---|
| `ENV_FILE` | dotenv file loaded before defaults; default `.env` |
| `IMAGE` / `RUNAI_IMAGE` | container image; repo default is the MLO uv image |
| `CLUSTER_HOME` | home PVC path inside the RunAI container |
| `PROJECT_DIR` | project path inside the RunAI container |
| `LDAP_UID`, `LDAP_GID` | use explicit RunAI UID/GID instead of `--run-as-user` |
| `NODE_POOLS` | optional RunAI `--node-pools` value |
| `WANDB_API_KEY` | forwarded into the container |
| `WANDB_DIR` | persistent WandB cache/log directory; default `${CLUSTER_HOME}/.wandb` |
| `UV_CACHE_DIR` | persistent uv cache; default `${CLUSTER_HOME}/.cache/uv` |
| `UV_PYTHON_INSTALL_DIR` | persistent uv Python install dir; default `${CLUSTER_HOME}/.uv` |
| `JOB_STAMP` | override timestamp suffix used in RunAI job names |

Training/config environment variables forwarded to `main.py`:

| env var | forwarded as |
|---|---|
| `CONFIG` | `--config` |
| `EXPERIMENT` | `--experiment` when `CONFIG` is unset |
| `OPTIMIZER` | `--optimizer` |
| `STEPS`, `LR`, `MIN_LR`, `SCHEDULER`, `SEED`, `LOG_EVERY` | training controls |
| `TOP_K`, `SIGMA_MODE`, `TAIL_MODE`, `GATE_THRESHOLD`, `GATE_WINDOW`, `KAPPA` | SpecMuon controls |
| `CONDITION_NUMBER` | ill-conditioned linear-regression override |
| `BACKEND`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_GROUP`, `LOG_SAV_R` | logging controls |
| `CHECKPOINT_DIR`, `CHECKPOINT_EVERY` | checkpoint controls |
| `SPECMUON_TARGET` | `--specmuon-target` |
| `COMPARE_OPTIMIZERS` | optimizer subset for comparison modes |
| `RUN_TAG` | suffix for RunAI job names and WandB run names |
| `LR_MIN`, `LR_MAX`, `LR_N` | best-LR comparison grid |
| `SWEEP` | space-separated sweep clauses for `run_job.sh sweep` |

`scripts/sync_to_rcp.sh` supports `LOCAL_DIR`, `REMOTE_HOST`, `REMOTE_DIR`,
`USER`, and `DRY_RUN=1`.

Fetch WandB runs back into local artifacts:

```bash
uv run python scripts/fetch_wandb_results.py \
  --entity <wandb-entity> \
  --project mlo-specmuon-shakespeare \
  --filter "sweep" \
  --log-y
```

## Mechanism-isolation scripts

Two scripts probe SpecMuon's SAV mechanism on the three benchmark problems
without depending on extra experiments. Both write to
`results/<name>/<experiment>/`.

```bash
# SAV isolation: top_k ∈ {0, 1, 6, 32} + gated variant on a fixed (experiment, lr).
# top_k=0 is the no-SAV ablation (paper-tail update on every direction).
uv run python scripts/sav_isolation.py --experiment matrix_factorization --lr 1e-1

# Gate sensitivity: τ ∈ {0, .01, .05, .1, .2} × window ∈ {5, 10, 20}.
# τ=0 reproduces the paper-default (always-on SAV) — useful baseline cell.
uv run python scripts/gate_sensitivity.py --experiment shakespeare --steps 600
```

## Device

`--device auto` (default) picks `cuda` → `mps` → `cpu`. Pass `cpu`, `cuda`,
or `mps` to override.

## Optimizers

- **Muon** — subclassed from `torch.optim.Muon`. Torch's internal
  `adjust_lr_fn` is disabled in the subclass so the only LR shape-scaling
  active is the one we explicitly select via `--adjust-lr-fn`.
- **SpecMuon** — Muon with SAV (Scalar Auxiliary Variable) adaptive scaling
  for the top-k singular directions. The remaining directions receive the
  paper-tail update `U_{k:} diag(S_{k:}) V_{k:}^T`. Defaults match the paper
  grid-search winners (`μ=0.9`, `top_k=6`, `sav_smooth=0.2`).
  [Paper](https://www.arxiv.org/abs/2602.16167).
  - **σ-mode knobs** (`power/clip/truncate/energy`) generalize the per-direction
    step size beyond the paper's `η/(σ+ε)`.
  - **SAV-gating** (`gate_threshold`/`gate_window`) bypasses SAV when the loss
    is dropping fast over a rolling window, collapsing to the tail-only update —
    the §4.2 regime where Muon-tail outperforms SAV. `gate_threshold=0`
    reproduces the paper default (always-on SAV) byte-identically.
  - **Tail-mode ablation** (`tail_mode=gradient|muon`) separates the paper
    gradient tail from a true Muon all-ones tail for the non-top-k directions.
  - **Momentum-mode ablation** compares paper post-spectral momentum against
    Nesterov variants and pre-SVD momentum placement.
  - **Layer-target ablation** (`specmuon_target=mlp|attention`) applies the
    SAV branch only to one transformer block type while routing the remaining
    matrix weights through the top-k=0 tail update.
  - **`last_iota` diagnostic** measures per-step SAV-intervention magnitude;
    log it with `--log-sav-r`.

## Tests

```bash
uv run pytest tests/
```

Short smoke checks:

```bash
uv run python main.py --experiment linear_regression --optimizer adamw --steps 5
uv run python main.py --experiment linear_regression --optimizer specmuon --steps 5 --adjust-lr-fn shape_scaling
```
