# DynMuon-Route

Dynamic **layer-wise spectral-exponent routing** for the Muon optimizer, built on
top of the [DynMuon](https://github.com/fzwark/DynMuon) baseline.

Muon-family methods shape the momentum-averaged gradient `M = U Σ Vᵀ` with a
spectral operator `D(p) = U Σ^p Vᵀ`:

- `p = 1.0` → SGD (raw singular values)
- `p = 0.0` → Muon (polar factor, all singular values → 1)
- `p = -0.25` → late-stage outlier suppression (strength reallocated to flat directions)

**DynMuon** drives a single global logistic *time* schedule `p_t : 1 → -0.25` for
every layer (reference `get_p`: `p = p_min + (p_max-p_min)/(1+exp((q_t-τ)/w))`,
with `q_t = step/total_steps`, `τ = w = 0.04`). **DynMuon-Route** (this repo)
replaces it with a *local* per-parameter proxy mapped through a per-layer-type
logistic to a parameter-specific exponent `p_{t,l}`.

## Routing proxies

| Metric | Definition | Routing logic |
|--------|------------|---------------|
| Stable rank `sr` (default) | `‖M‖_F² / σ_max²` | low sr (anisotropic) → p → -0.25; high sr → p ≥ 0 |
| SNR proxy `γ` | `‖M‖_F / ‖G - M‖_F` | low γ (noisy) → p ≥ 0; high γ → negative allowed |
| Alignment `α` | `|tr(Wᵀ M)| / (‖W‖_F‖M‖_F)` | high α → p → 1; low α → p → 0 |

Mapping: `p = p_min + (p_max - p_min) / (1 + exp(-(x - μ)/ω))`, per-layer-type
`μ, ω` (sign of `ω` sets orientation). `p_min = -0.25`, `p_max = 1.0`.

## Compute backends

- `compute_mode="svd"` — exact `U Σ^p Vᵀ` via SVD (validation / debugging).
- `compute_mode="ns"` — fast path using `U Σ^p Vᵀ = A^{p/2} Y_μ`, where `Y_μ` is the
  Newton-Schulz polar factor (`ns_variant`: `quintic`, the reference DynMuon tuned
  5-step iteration, or `cubic`, the textbook `1.5X - 0.5XXᵀX`) and `A = X_n X_nᵀ` is
  the small Gram matrix whose symmetric eigendecomposition gives `A^{p/2}` (and
  `λ_max` for the stable rank).

LR scaling follows DynMuon's spectral-norm rule `sqrt(fan_out/fan_in)`.

## Layout

```
dynmuon/                 # library package
  optimizer.py           # DynMuonRoute, schedule, Newton-Schulz, routing math
  models.py              # 124M GPT (nanoGPT naming) + gated-MLP block
  data.py                # WikiText-103 memmap batch loading
  trainer.py             # train loop, optimizer wiring, routing logging, noise hook
  config.py              # YAML loading (extends) + device selection
  analysis.py            # history dumps, steps-to-target, series helpers
configs/
  base.yaml              # all defaults (every knob lives here)
  gpt124m.yaml           # 124M reference (extends base)
  small.yaml             # fast local / smoke config (extends base)
  adamw / muon / dynmuon / route .yaml   # the four methods (extend gpt124m)
  exp1_spectral.yaml     # experiment 1 (extends small)
  exp2_noise.yaml        # experiment 2 (extends small)
train.py                 # thin CLI entry point
validate_math.py         # pytest: factorization == exact SVD, NS polar factor, ns≈svd
experiments/
  baselines_step_efficiency.py # AdamW / Muon / DynMuon / Route step-efficiency
  exp1_spectral_evolution.py   # per-layer p_{t,l} trajectories
  exp2_noise_injection.py      # anisotropic-noise robustness
scripts/
  prepare_wikitext.py    # WikiText-103 -> GPT-2-BPE train.bin/val.bin
  run_job.sh             # RunAI cluster submission (+ container_entry.sh)
```

## Methods (one config each)

| config | method | how |
|--------|--------|-----|
| `configs/adamw.yaml` | AdamW | `matrix_optimizer: adamw` (all params to AdamW) |
| `configs/muon.yaml` | Muon | `routing_mode: fixed`, `fixed_p: 0.0` (constant p=0) |
| `configs/dynmuon.yaml` | DynMuon | `routing_mode: global_schedule` (logistic p_t) |
| `configs/route.yaml` | **DynMuon-Route** | `routing_mode: stable_rank` (per-layer router) |

Run any single method: `python train.py --config configs/<method>.yaml [--model small]`.

All customization is done through `configs/*.yaml`; a config `extends:` another and
overrides selected keys. CLI flags (e.g. `--routing-mode`, `--max-steps`) override
the YAML for ad-hoc runs.

## Quickstart

```bash
uv sync
pytest validate_math.py                          # spectral-math unit tests
python scripts/prepare_wikitext.py               # tokenize WikiText-103
python train.py --config configs/small.yaml --max-steps 50    # smoke test
python experiments/baselines_step_efficiency.py  # 4-method comparison (small model)
python experiments/exp1_spectral_evolution.py    # spectral-evolution plot (small model)
python experiments/exp2_noise_injection.py       # noise-robustness plot (small model)
```

Cluster (124M):

```bash
scripts/run_job.sh sanity
scripts/run_job.sh baselines --model gpt124m --max-steps 20000
scripts/run_job.sh exp1 --model gpt124m --max-steps 4000
scripts/run_job.sh exp2 --model gpt124m
```

## Experiments

Each experiment script trains the relevant runs, writes plots **and** dumps the raw
metric trajectories to `results/<exp>/history_*.json` for numerical analysis.

- **Baselines (`results/baselines/`)** — AdamW vs Muon (p=0) vs DynMuon vs
  DynMuon-Route. Reports validation-loss curves and step efficiency (fewer steps to
  a common target loss); the project target is DynMuon reaching it ~10–26% faster
  than Muon. Outputs `loss_curves.png` + `summary.md`.
- **Exp 1 — spectral evolution (`results/exp1_spectral_evolution/`)** — does
  Attention reject negative `p` (stays `p ≥ 0`) while MLP routes toward `p = -0.25`?
  Global schedule vs Stable-Rank router; plots `p_{t,l}` for `c_attn`, `c_proj`,
  `mlp.c_fc`, `mlp.c_proj`.
- **Exp 2 — noise injection (`results/exp2_noise_injection/`)** — inject
  `M += λ·σ₁·z·u₁v₁ᵀ` and check the Stable-Rank router detects the anisotropic spike,
  drops `p` negative, and stays stable while the global schedule destabilizes.
