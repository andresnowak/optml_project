# DynMuon-Route

Dynamic **layer-wise spectral-exponent routing** for the Muon optimizer, built on
top of the [DynMuon](https://github.com/fzwark/DynMuon) baseline.

## Intuition

Muon-style updates reshape the momentum gradient's singular values with one knob `p`,
via `D(p) = U Σ^p Vᵀ`:

- `p = 1` → keep raw curvature (SGD-like; good for fast early progress)
- `p = 0` → flatten all singular values to 1 (Muon)
- `p = -0.25` → *suppress* the dominant directions (good late, when the top
  directions are mostly noise / curvature traps)

**DynMuon** anneals one shared clock `p_t : 1 → -0.25` over training: trust curvature
early, suppress it late. **DynMuon-Route** keeps that clock but lets each layer lean:

```
p_{t,l} = clip( p_t  +  clip(beta · (gₗ − ḡₜ)/σ̃ₜ, ±lean_max) ,  −0.25, 1.0 )
          └──┬──┘             └──────┬──────┘
        shared clock           personal lean (z-scored)
```

where `gₗ` is layer `l`'s gradient concentration (stable rank), `ḡₜ` is the
network's **running average** of it, and `σ̃ₜ` the running cross-layer spread
(`lean_norm: zscore`). In one breath: *everyone follows the same clock, but a layer
whose gradient is more lopsided than its peers right now leans `p` down (suppress
that direction harder); a more balanced layer leans up.* The z-scoring and
`lean_max` exist because stable rank is heavy-tailed (deviations of +50 on early
layers): with a raw gain the router saturates into a bang-bang controller pinned at
the clip boundaries (observed in the 20k marathon run: 7/72 matrices stuck at p=+1
while the schedule said −0.25).

Why the **deviation** `gₗ − ḡₜ` and not `gₗ` itself: stable rank drifts globally over
training (~1 → 4), so a layer's absolute value mostly encodes *what time it is* — which
the clock already captures. Subtracting the running mean removes that shared trend and
leaves the only genuinely per-layer signal. So the **clock carries what's universal
(time); the lean carries what's local (geometry).**

Where it should shine (Exp 2): inject anisotropic noise into one layer → its stable
rank collapses below the network average → the lean drives that layer's `p` negative
and suppresses the spike. A fixed clock has no feedback to react; a deviation-driven
lean does.

**Calibrate before long runs:** `python experiments/probe_proxies.py` prints the
per-layer-type proxy distributions and suggested `ref`/`beta`. The pure-logistic modes
(`stable_rank`/`snr`/`alignment`, no clock) are available for ablation.

## Routing proxies

| Metric | Definition | Routing logic |
|--------|------------|---------------|
| Stable rank `sr` (default) | `‖M‖_F² / σ_max²` | low sr (anisotropic) → p → -0.25; high sr → p ≥ 0 |
| SNR proxy `γ` | `‖M‖_F / ‖G - M‖_F` | low γ (noisy) → p → p_max (raw momentum downweights weak dirs); high γ → negative allowed |
| EMA SNR `γ̂` (`snr_ema`) | bias-corrected `‖EMA(G)‖ / std(G)` | same orientation as γ, but a real signal-to-noise estimate |
| Alignment `α` | `|tr(Wᵀ M)| / (‖W‖_F‖M‖_F)` | high α → p → 1; low α → p → 0 |

Mapping: `p = p_min + (p_max - p_min) / (1 + exp(-(x - μ)/ω))`, per-layer-type
`μ, ω` (sign of `ω` sets orientation). `p_min = -0.25`, `p_max = 1.0`.

## Compute backends

- `compute_mode="reference"` (default) — the **released DynMuon transform,
  bit-matched** (verified in `validate_reference.py` against vendored reference
  code): 3-phase switch (`p ≥ 0.25` → raw momentum; `0 ≤ p < 0.25` → bf16 quintic
  Newton-Schulz polar; `p < 0` → `fast_spectral` order-2 Taylor with the `‖M‖^p`
  back-scale), bf16 update quantization, decoupled weight decay at the base LR.
- `compute_mode="svd"` — exact continuous `U Σ^p Vᵀ` on the **raw** spectrum
  (float32, pseudo-power rank tolerance; validation / spectrum controls).
- `compute_mode="ns"` — continuous `U Σ^p Vᵀ` via `‖M‖^p · A^{p/2} Y_μ`, where
  `Y_μ` is the Newton-Schulz polar factor (`ns_variant`: `quintic` or `cubic`) and
  `A = X_n X_nᵀ` is the small Gram matrix (eigendecomposition gives `A^{p/2}` and
  `λ_max` for the stable rank).

Orthogonal experiment knobs (see `math.tex` §3.1/§6): `magnitude: polar_fro`
rescales every update to `‖D‖_F = sqrt(min(m,n))` so `p` changes only the spectrum
*shape* (without it, moving `p` from 1 to 0 multiplies the update norm by ~27 at
768-dim — the p-schedule doubles as an implicit LR schedule); `spectrum: random |
inverted` are Frobenius-norm-preserving Kaon-style controls (`compute_mode: svd`).

Shape-aware LR scaling is controlled by `adjust_lr_fn` in YAML. The default is
`spectral_norm`, matching DynMuon's `sqrt(fan_out/fan_in)` rule. DynMuon also
supports `rms_norm`; Muon also supports `keller_jordan`.

## Layout

```
src/                     # project package
  optimizers/dynmuon.py  # DynMuonRoute, schedule, Newton-Schulz, routing math
  optimizers/            # optimizer registry + parameter grouping
  models/gpt.py          # Track-3-inspired GPT: q/k/v, RMSNorm, RoPE, ReLU^2
  data/                  # cached-token loading
  trainer.py             # train loop, routing logging, validation, noise hook
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
  sync_to_rcp.sh         # rsync local checkout to the RunAI submit host
  run_job.sh             # RunAI cluster submission (+ container_entry.sh)
data/
  prepare_wikitext.py    # WikiText-103 -> GPT-2-BPE train.bin/val.bin
  prepare_fineweb.py     # FineWeb10B GPT-2-BPE shards by requested token count
```

## Methods (one config each)

| config | method | how |
|--------|--------|-----|
| `configs/adamw.yaml` | AdamW | `matrix_optimizer: adamw` (matrix params use `weight_decay`; aux params use `scalar_weight_decay`) |
| `configs/muon.yaml` | Muon | `matrix_optimizer: muon` (Track-3-style Muon) |
| `configs/muon_svd.yaml` | exact-polar Muon | `orthogonalize: svd`, sets every live singular value to 1 |
| `configs/gated_muon.yaml` | GatedMuon | `matrix_optimizer: gated_muon`, Muon with a small-singular-value gate (`gate_tau`) |
| `configs/dynmuon.yaml` | DynMuon | `routing_mode: global_schedule` (reference-exact) |
| `configs/route.yaml` | **DynMuon-Route** | `routing_mode: schedule_modulated` (z-scored per-layer router) |
| `configs/route_decoupled.yaml` | Route, decoupled | route + `magnitude: polar_fro` (pure spectrum-shape routing) |
| `configs/relmuon_*.yaml` | RelMuon | weight-spectrum scales (`log1p`, `rms`, `complete`, plus `log1p_aligned` via `relmuon_scale_mode`) |

Run any single method: `python train.py --config configs/<method>.yaml`.

### Ablation knobs & sweeps

`train.py` exposes a small set of config-key overrides for ad-hoc runs:
`--seed`, `--routing-mode`, `--compute-mode`, `--ns-variant`, `--magnitude`,
`--spectrum`, `--track-proxies/--no-track-proxies`, `--snr-ema-decay`,
`--train-steps`, `--batch-size`, `--sequence-length`, `--mbs`, `--val-tokens`,
`--val-loss-every`, `--warmup-steps`, `--min-lr-ratio`, `--muon-lr`, `--adam-lr`,
`--embed-lr`, `--weight-decay`, `--relmuon-scale-mode`, `--relmuon-scale-cap`,
`--beta`, `--lean-norm`, `--lean-max`,
`--modulate-metric`, `--dynamic-ref/--no-dynamic-ref`, `--noise-lambda`,
`--run-name`, `--wandb-group`, `--wandb-project`, `--wandb-entity`, `--device`,
and `--wandb`.

`scripts/sweep.sh` submits a one-parameter sweep as separate W&B runs in one group:

```bash
scripts/sweep.sh beta_sweep --beta 0,0.25,0.5,1.0 --config configs/route.yaml
scripts/sweep.sh seed_route --seed 0,1,2          --config configs/route.yaml
scripts/sweep.sh proxy --modulate-metric stable_rank,alignment --config configs/route.yaml
```

For boolean knobs (e.g. `--dynamic-ref` vs `--no-dynamic-ref`) just submit the two
`single` runs directly with a shared `--wandb-group`.

### LR sweep bowl plots

Use the preset wrapper to regenerate the main five-project LR bowl. It pulls the
W&B sweep projects, drops AdamW `1e-4` through `1e-3`, marks each method's best
plotted point with a star, and writes the CSV/PNG under
`results/lr_bowls/adamw_filtered_star_best/`.
It also writes AdamW-target speed plots: the fastest LR per method to reach
AdamW's best plotted `val/loss` and `train/loss`, once by wall-clock training
time and once by optimizer steps.

```bash
scripts/plot_lr_sweep_bowl.sh
```

For custom W&B projects, call the plotting script directly. Pass projects as a
list with `--sweep-projects`; pass matching display names with `--labels` in the
same order.

```bash
uv run python experiments/lr_bowl.py \
  --sweep-projects relmuon-rms-lr-sweep adam-lr-sweep muon-lr-sweep \
  --labels RelMuon-RMS AdamW Muon \
  --entity cs-439-project \
  --selection final \
  --exclude-lr-range AdamW:1e-4:1e-3 \
  --out-dir results/lr_bowls/custom
```

Comma-separated lists also work:

```bash
uv run python experiments/lr_bowl.py \
  --sweep-projects relmuon-rms-lr-sweep,adam-lr-sweep,muon-lr-sweep \
  --labels RelMuon-RMS,AdamW,Muon \
  --out-dir results/lr_bowls/custom
```

Use `--selection best` to plot best-seen losses instead of final losses. The
legacy grouped-run workflow is still available with `--project <wandb-project>`
and `--group <wandb-group>`.

By default, the speed plots use `--target-series AdamW`; override this with
`--target-series <label>` or pass `--no-target-plots` to only regenerate the LR
bowl.

All customization is done through `configs/*.yaml`; a config `extends:` another and
overrides selected keys. CLI flags (e.g. `--routing-mode`, `--train-steps`) override
the YAML for ad-hoc runs.

`batch_size` and `mbs` are both sequence counts. The actual token budget per
optimizer step is `batch_size * sequence_length`.

## Reproducing the report

`python run.py` reproduces every figure and table in `report/main.tex`: it
pulls all sweep histories from W&B (`experiments/pull_wandb.py`) into
`results/wandb/` and regenerates the plots (`experiments/report_figures.py`)
into `report/figures/` — LR bowls, loss curves, depth-resolved routing, the
beta sweep, the proxy comparison, the empirical SVD≡Newton-Schulz check, and
the step-time/steps-to-target cost comparison. `--skip-pull` reuses local
dumps. The training runs themselves are submitted with
`scripts/sweeps.sh bowls`, `scripts/sweeps.sh route 0.02`, and
`scripts/sweeps.sh final 0.02` (see `notes/timeline.md`).

## Quickstart

```bash
uv sync
uv run pytest validate_math.py validate_reference.py   # spectral math + reference parity
uv run python data/prepare_wikitext.py                  # default small/base data
uv run python data/prepare_fineweb.py 500M              # gpt124m data: 5 shards ~= 1 GB
uv run python train.py --config configs/small.yaml --train-steps 50   # smoke test
uv run python experiments/baselines_step_efficiency.py  # 4-method comparison (124M configs)
uv run python experiments/exp1_spectral_evolution.py    # spectral-evolution plot (small model)
uv run python experiments/exp2_noise_injection.py       # noise-robustness plot (small model)
```

## RunAI / RCP workflow

The cluster workflow is:

1. Sync this local checkout to RCP.
2. Submit RunAI jobs from the synced checkout.
3. Use `run_job.sh logs/list/delete` to inspect or clean up jobs.

From the local machine:

```bash
scripts/sync_to_rcp.sh
```

By default this syncs the repo to
`jhrcp:/home/nowak/developer/optml_project`, excludes local caches (`.git`,
`.venv`, `wandb`, `__pycache__`, etc.), deletes remote files that no longer exist
locally, and makes the cluster scripts executable. Useful overrides:

```bash
DRY_RUN=1 scripts/sync_to_rcp.sh
REMOTE_HOST=myhost REMOTE_USER=me REMOTE_DIR=/home/me/developer/optml_project scripts/sync_to_rcp.sh
```

Then SSH to the synced checkout on RCP and submit jobs:

```bash
scripts/run_job.sh prep
scripts/run_job.sh prep-fineweb 500M
scripts/run_job.sh sanity
scripts/run_job.sh single --config configs/route.yaml --wandb
scripts/run_job.sh baselines --train-steps 20000
scripts/run_job.sh exp1 --config configs/gpt124m.yaml --train-steps 4000
scripts/run_job.sh exp2 --config configs/gpt124m.yaml
scripts/run_job.sh logs <job-name>
scripts/run_job.sh list
scripts/run_job.sh delete <job-name>
```

`run_job.sh` resolves the project path as seen inside the RunAI pod and invokes
`scripts/container_entry.sh`. The entry script `cd`s into the project, sets
`PYTHONPATH`, installs `uv` if needed, runs `uv sync --locked`, and finally executes
the requested Python script through `uv run --no-sync`.

Common environment overrides can live in `.env` or be passed inline:

```bash
IMAGE=ic-registry.epfl.ch/mlo/mlo-base:uv1 GPUS=1 scripts/run_job.sh sanity
NODE_POOLS=h100 scripts/run_job.sh single --config configs/muon.yaml --wandb
UV_SYNC=0 scripts/run_job.sh single --config configs/small.yaml --train-steps 20
UV_SYNC_ARGS="--locked --extra dev" scripts/run_job.sh sanity
```

By default, `run_job.sh` submits with RunAI's `--run-as-user`, so files written on
the home PVC should be owned by your cluster user rather than by root inside the
container. If your RunAI setup needs explicit numeric IDs instead, set both
`LDAP_UID` and `LDAP_GID`; the script will use
`--run-as-uid ${LDAP_UID} --run-as-gid ${LDAP_GID}` in place of `--run-as-user`.

The most useful knobs are `IMAGE`/`RUNAI_IMAGE`, `GPUS`, `CLUSTER_HOME`,
`PROJECT_DIR`, `REMOTE_USER`, `NODE_POOLS`, `LDAP_UID`/`LDAP_GID`, `UV_SYNC`,
`UV_SYNC_ARGS`, `HF_TOKEN`, and the `WANDB_*` variables. If `uv sync --locked`
fails in the pod after dependency changes, update the lockfile locally, sync again,
and resubmit.

## Experiments

Each experiment script trains the relevant runs, writes plots **and** dumps the raw
metric trajectories to `results/<exp>/history_*.json` for numerical analysis. Pass
`--wandb` to additionally log every arm as its own W&B run inside a shared group
(`--wandb-group`), so the arms overlay in the W&B UI.

- **Baselines (`results/baselines/`)** — AdamW vs Muon (p=0) vs DynMuon vs
  DynMuon-Route. Reports validation-loss curves and step efficiency (fewer steps to
  a common target loss); the project target is DynMuon reaching it ~10–26% faster
  than Muon. Outputs `loss_curves.png` + `summary.md`.
- **Exp 1 — spectral evolution (`results/exp1_spectral_evolution/`)** — does
  Attention reject negative `p` (stays `p ≥ 0`) while MLP routes toward `p = -0.25`?
  Global schedule vs Stable-Rank router; plots `p_{t,l}` for attention `q/k/v/proj`
  and `mlp.fc` / `mlp.proj`.
- **Exp 2 — noise injection (`results/exp2_noise_injection/`)** — inject
  `M += λ·σ₁·z·u₁v₁ᵀ` and check the Stable-Rank router detects the anisotropic spike,
  drops `p` negative, and stays stable while the global schedule destabilizes.

The two scripts below are pure analysis over the `history_*.json` dumps (no GPU,
no training env) — run them after the experiments above:

- **Depth routing (`depth_routing.png` + `.json`)** — depth-resolved view of how far
  each layer's routed exponent departs from the shared clock,
  `mean_t (p_{t,l} − p_t)` vs. block index, one series per matrix type. The clock
  `p_t` comes from the `global_schedule` run:
  `python experiments/depth_routing.py --routed results/exp1_spectral_evolution/history_schedule_modulated.json --baseline results/exp1_spectral_evolution/history_global_schedule.json`
- **Cost table (`cost.md` + `cost.png`)** — per-optimizer best/final val loss, total
  wall-clock time, ms/step, and both steps- and time-to-target (with `%` vs a
  reference). Separates "fewer steps" from "cheaper per step":
  `python experiments/cost_table.py --results-dir results/baselines --reference muon`

## References

```bibtex
@misc{wu2026dynmuondynamicspectralshaping,
      title={DynMuon: A Dynamic Spectral Shaping View of Muon},
      author={Fangzhou Wu and Rikhav Shah and Sandeep Silwal and Qiuyi Zhang},
      year={2026},
      eprint={2605.17109},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.17109},
}

@misc{modded_nanogpt_2024,
  author       = {Keller Jordan and Jeremy Bernstein and Brendan Rappazzo and
                  @fernbear.bsky.social and Boza Vlado and You Jiacheng and
                  Franz Cesista and Braden Koszarsky and @Grad62304977},
  title        = {modded-nanogpt: Speedrunning the NanoGPT baseline},
  year         = {2024},
  url          = {https://github.com/KellerJordan/modded-nanogpt}
}
```
