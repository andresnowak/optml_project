# Project Guide: Singular-Value Control in Muon-Style Optimizers

This document summarizes what this repository is doing, how the code is organized,
what is already implemented, what is still missing, and how to run the experiments.

The current project direction is:

- Put SpecMuon aside for now.
- Work on GPT training experiments.
- Study how to handle the singular values of the matrix update in Muon-style optimizers.
- Compare Muon, DynMuon, and a routed layer-wise DynMuon variant.
- Add random or alternative singular-value spectra as explicit ablations before making claims about them.
- Analyze the route contribution by measuring how each layer's exponent differs from the global schedule.

## 1. Research Question

Muon applies a matrix update based on the polar factor of the momentum gradient.
If the momentum update is

$$
M = U \Sigma V^T,
$$

then standard Muon effectively uses

$$
D(0) = U V^T,
$$

which means all singular values of the update direction are set to 1.

This project asks a broader question:

> Instead of always flattening the singular values to 1, can we choose a better
> singular-value transformation for the update direction?

The main implemented transformation is the DynMuon-style spectral exponent:

$$
D(p) = U \Sigma^p V^T.
$$

Important cases:

- `p = 1`: keep the raw momentum spectrum, close to SGD with momentum.
- `p = 0`: flatten the spectrum, standard Muon.
- `p < 0`: suppress dominant singular directions and amplify smaller ones.

The active hypothesis is that a single global value of `p` may be too crude.
Different GPT layers can have different gradient geometry, so every layer should
be allowed to deviate from the shared global schedule.

## 2. Current Main Idea

The current mainline is not SpecMuon. The repo is centered on DynMuon-Route:

$$
p_{t,l} = \mathrm{clip}(p_t + \beta (g_l - \tilde{g}_t), -0.25, 1.0).
$$

Where:

- `p_t` is the global DynMuon clock shared by every layer.
- `l` is the layer or parameter group.
- `g_l` is a routing proxy measured from that layer's update geometry.
- `\tilde{g}_t` is the running network-wide proxy average.
- `beta` controls how much a layer can deviate from the clock.
- The clipped range is `[p_min, p_max] = [-0.25, 1.0]`.

Interpretation:

- Everyone follows the same training-time clock.
- Each layer gets a local correction based on its gradient geometry.
- The correction is centered by the running network average, so the router does
  not duplicate the global time trend.

This is exactly the route contribution we want to analyze:

$$
p_{t,l} - p_t.
$$

The desired plot is:

- x-axis: layer depth or parameter name.
- y-axis: average `p_{t,l} - p_t`.
- interpretation: which layers are being routed above or below the global schedule.

The current code logs enough information to build this plot, but there is not yet
a dedicated plotting script for `p_{t,l} - p_t` by depth. The closest implemented
script is `experiments/exp1_spectral_evolution.py`, which plots per-layer `p`
trajectories.

## 3. What Is Implemented

### GPT Model

The experiments train a GPT model implemented in `src/models/gpt.py`.

The architecture is Track-3 / modded-nanoGPT inspired:

- separate attention `q`, `k`, `v`, and `proj` matrices,
- RMSNorm,
- RoPE,
- ReLU-squared MLP,
- tied output embedding.

This matters because parameter names make it possible to separate attention and
MLP matrix updates during routing and logging.

### Training Loop

The main training loop is in `src/trainer.py`.

It handles:

- model construction,
- token cache loading,
- optimizer construction,
- warmup plus cosine learning-rate schedule,
- gradient accumulation through microbatches,
- validation loss,
- W&B logging,
- in-memory history logging for experiment scripts,
- route diagnostics logging,
- anisotropic noise injection for Experiment 2.

Training entrypoint:

```bash
uv run python train.py --config configs/<config>.yaml
```

`train.py` only calls `src.cli.main()`, which loads config and CLI overrides.

### Optimizers

Optimizer construction is in `src/optimizers/registry.py`.

The repo supports these matrix optimizer modes:

| Config | Matrix optimizer | Meaning |
| --- | --- | --- |
| `configs/adamw.yaml` | `adamw` | AdamW for matrix and auxiliary parameters |
| `configs/muon.yaml` | `muon` | Standard Muon, fixed polar update |
| `configs/dynmuon.yaml` | `dynmuon` + `global_schedule` | DynMuon global spectral schedule |
| `configs/route.yaml` | `dynmuon` + `schedule_modulated` | DynMuon-Route, per-layer routed exponent |

Non-matrix parameters are handled by AdamW:

- embeddings,
- biases,
- RMSNorm gains,
- other scalar parameters.

Matrix parameters are handled by Muon or DynMuonRoute.

### Parameter Grouping

Parameter grouping is in `src/optimizers/param_groups.py`.

For routed runs, matrix parameters are split into:

- `attn`,
- `mlp`,
- `other`.

For non-routed runs, all matrix parameters go into one `matrix` group.

The layer type is inferred from parameter names:

- names containing `.attn.` become `attn`,
- names containing `.mlp.` become `mlp`,
- everything else becomes `other`.

### Muon

Standard Muon is implemented in `src/optimizers/muon.py`.

It uses:

- momentum,
- optional Nesterov momentum,
- Newton-Schulz polar update,
- shape-aware LR scaling,
- decoupled weight decay on matrix parameters.

This matches the project requirement that if Muon uses Nesterov momentum, the other
methods should also use Nesterov and apply singular-value changes to the proposed
momentum update.

### DynMuon and DynMuon-Route

DynMuonRoute is implemented in `src/optimizers/dynmuon.py`.

It supports these routing modes:

| `routing_mode` | Meaning |
| --- | --- |
| `fixed` | fixed exponent `fixed_p`; `fixed_p=0` is Muon-like |
| `global_schedule` | one DynMuon schedule `p_t` for every layer |
| `schedule_modulated` | global schedule plus per-layer route correction |
| `stable_rank` | direct logistic map from stable rank to `p` |
| `snr` | direct logistic map from SNR proxy to `p` |
| `alignment` | direct logistic map from alignment proxy to `p` |

The active project method is:

```yaml
matrix_optimizer: dynmuon
routing_mode: schedule_modulated
```

### Compute Backends

`src/optimizers/dynmuon.py` supports two ways to compute `D(p)`:

| `compute_mode` | Meaning |
| --- | --- |
| `svd` | exact SVD computation of `U Sigma^p V^T` |
| `ns` | fast Newton-Schulz plus Gram eigendecomposition path |

The math validation is in `validate_math.py`.

Run it with:

```bash
uv run pytest validate_math.py
```

This validates:

- exact SVD factorization,
- Newton-Schulz polar factor behavior,
- `ns` vs `svd` optimizer update agreement,
- fixed `p` behavior,
- schedule and clipping behavior.

## 4. Routing Proxies

The implemented proxies are:

| Proxy | Logged key | Meaning |
| --- | --- | --- |
| Stable rank | `route/sr/<param>` | `||M||_F^2 / sigma_max(M)^2`; low means anisotropic |
| SNR proxy | `route/gamma/<param>` | `||M||_F / ||G - M||_F` |
| Alignment | `route/alpha/<param>` | normalized `|tr(W^T M)|` |

For the current route method, the default proxy is stable rank:

```yaml
route:
  schedule_modulated:
    metric: stable_rank
```

The current default uses:

```yaml
dynamic_ref: true
ref_decay: 0.9
beta: 0.5
```

With `dynamic_ref: true`, the reference is a running cross-layer mean. This is
important because stable rank can drift globally during training; subtracting the
running mean makes the route focus on per-layer deviation.

## 5. Random Singular Values and Other Spectra

The project notes mention:

> Muon is Not That Special: Random or Inverted Spectra Work Just as Well.

This is conceptually aligned with the repo goal, but it is not implemented yet in
the current codebase.

Current status:

- random singular values are not a live optimizer mode,
- inverted spectra are not a live optimizer mode,
- RelMuon is not a live optimizer mode,
- top-k or SpecMuon-style singular-value editing is not a live optimizer mode.

The repo is ready to host these ablations because all matrix updates already pass
through a centralized spectral shaping point in `DynMuonRoute._update_param`.

Recommended implementation plan for random-spectrum ablations:

1. Add a new config key such as `spectrum_mode`.
2. Keep `p` routing unchanged when `spectrum_mode: power`.
3. Add `spectrum_mode: random` to replace `S^p` with sampled singular values.
4. Add `spectrum_mode: inverted` to test reversed or reciprocal spectra.
5. Log the sampled or transformed spectrum statistics.
6. Add method configs such as `configs/random_spectrum.yaml`.
7. Add the method to `experiments/baselines_step_efficiency.py`.

This separation matters. We should not claim experimental evidence for random
singular values until this mode is actually implemented and run.

## 6. Config System

Configs live in `configs/`.

The loader in `src/config.py` supports:

- `extends: <file>.yaml`,
- recursive YAML merging,
- CLI overrides applied last.

Important configs:

| File | Purpose |
| --- | --- |
| `configs/base.yaml` | shared defaults |
| `configs/small.yaml` | small GPT for local smoke tests |
| `configs/gpt124m.yaml` | reference 124M GPT config |
| `configs/adamw.yaml` | AdamW baseline |
| `configs/muon.yaml` | Muon baseline |
| `configs/dynmuon.yaml` | global DynMuon schedule |
| `configs/route.yaml` | DynMuon-Route |
| `configs/exp1_spectral.yaml` | spectral evolution experiment |
| `configs/exp2_noise.yaml` | noise injection experiment |

Important training knobs:

| Key | Meaning |
| --- | --- |
| `train_steps` | number of optimizer steps |
| `batch_size` | sequences per optimizer step |
| `mbs` | sequences per microbatch |
| `sequence_length` | tokens per sequence |
| `warmup_steps` | linear LR warmup |
| `min_lr_ratio` | final cosine LR ratio |
| `muon_lr` | matrix optimizer LR |
| `adam_lr` | auxiliary AdamW LR |
| `weight_decay` | matrix weight decay |
| `scalar_weight_decay` | auxiliary weight decay |
| `momentum` | Muon/DynMuon momentum |
| `nesterov` | use Nesterov momentum |
| `beta` | route strength |
| `modulate_metric` | route proxy |
| `compute_mode` | `ns` or `svd` |

Token budget per optimizer step:

$$
\text{tokens per step} = \text{batch_size} \times \text{sequence_length}.
$$

## 7. Data

### WikiText-103

WikiText is the default local data source in `configs/base.yaml`.

Prepare it with:

```bash
uv run python data/prepare_wikitext.py
```

This downloads `Salesforce/wikitext`, config `wikitext-103-raw-v1`, tokenizes with
GPT-2 BPE, and writes:

- `data/wikitext103/train.bin`
- `data/wikitext103/val.bin`
- `data/wikitext103/metadata.json`

### FineWeb

The 124M config uses FineWeb token shards:

```yaml
data_dir: data/fineweb10B
train_bin: fineweb_train_*.bin
val_bin: fineweb_val_000000.bin
```

Prepare 500M training tokens with:

```bash
uv run python data/prepare_fineweb.py 500M
```

This downloads from `kjj0/fineweb10B-gpt2`.

## 8. Local Setup

Install dependencies:

```bash
uv sync
```

Validate spectral math:

```bash
uv run pytest validate_math.py
```

Prepare WikiText:

```bash
uv run python data/prepare_wikitext.py
```

Run a smoke test:

```bash
uv run python train.py --config configs/small.yaml --train-steps 50
```

Run one method:

```bash
uv run python train.py --config configs/route.yaml
```

Useful overrides:

```bash
uv run python train.py --config configs/route.yaml --train-steps 400
uv run python train.py --config configs/route.yaml --beta 0.25
uv run python train.py --config configs/route.yaml --modulate-metric alignment
uv run python train.py --config configs/dynmuon.yaml --compute-mode svd
uv run python train.py --config configs/route.yaml --wandb --wandb-group beta_sweep
```

## 9. Experiments

### Experiment A: Baseline Step Efficiency

Script:

```bash
uv run python experiments/baselines_step_efficiency.py
```

Compares:

- AdamW,
- Muon,
- DynMuon global schedule,
- DynMuon-Route.

Outputs:

- `results/baselines/history_adamw.json`
- `results/baselines/history_muon.json`
- `results/baselines/history_dynmuon.json`
- `results/baselines/history_dynmuon_route.json`
- `results/baselines/loss_curves.png`
- `results/baselines/summary.md`

With overrides:

```bash
uv run python experiments/baselines_step_efficiency.py --train-steps 20000
uv run python experiments/baselines_step_efficiency.py --target-loss 4.0
uv run python experiments/baselines_step_efficiency.py --wandb --wandb-group baselines
```

This gives the loss curves and step-efficiency comparison requested in the project
notes.

### Experiment B: Spectral Evolution / Route Contribution

Script:

```bash
uv run python experiments/exp1_spectral_evolution.py
```

Compares:

- `global_schedule`,
- `schedule_modulated`.

Outputs:

- `results/exp1_spectral_evolution/history_global_schedule.json`
- `results/exp1_spectral_evolution/history_schedule_modulated.json`
- `results/exp1_spectral_evolution/p_trajectories.png`

With overrides:

```bash
uv run python experiments/exp1_spectral_evolution.py --config configs/exp1_spectral.yaml --train-steps 400
uv run python experiments/exp1_spectral_evolution.py --beta 0.25
uv run python experiments/exp1_spectral_evolution.py --modulate-metric alignment
```

This is the current implemented route-analysis experiment.

Missing but recommended:

- add a plot of average `p_{t,l} - p_t` by layer depth,
- aggregate separately for attention and MLP,
- report mean and standard deviation across seeds.

The raw histories already include per-parameter route values, so this is a plotting
addition rather than an optimizer change.

### Experiment C: Noise Injection

Script:

```bash
uv run python experiments/exp2_noise_injection.py
```

The trainer injects anisotropic noise into the momentum matrix:

$$
M \leftarrow M + \lambda \sigma_1 z u_1 v_1^T.
$$

Compares:

- global schedule,
- routed schedule.

Outputs:

- `results/exp2_noise_injection/history_global_schedule.json`
- `results/exp2_noise_injection/history_schedule_modulated.json`
- `results/exp2_noise_injection/noise_stability.png`

With overrides:

```bash
uv run python experiments/exp2_noise_injection.py --config configs/exp2_noise.yaml --train-steps 300 --noise-lambda 3.0
uv run python experiments/exp2_noise_injection.py --beta 0.5
uv run python experiments/exp2_noise_injection.py --modulate-metric stable_rank
```

The intended result is that the router detects the anisotropic spike through stable
rank, lowers `p`, and stabilizes training better than the global schedule.

### Experiment D: Proxy Calibration

Script:

```bash
uv run python experiments/probe_proxies.py --config configs/small.yaml --steps 150
```

This prints distributions for:

- stable rank,
- SNR proxy,
- alignment.

Use it before long runs to set:

- route `ref`,
- `beta`,
- logistic `mu`,
- logistic `omega`.

### Experiment E: Learning-Rate Bowls

The repo has a W&B plotting utility:

```bash
uv run python experiments/lr_bowl.py --group <group-name>
```

Examples:

```bash
uv run python experiments/lr_bowl.py --group route_lr_sweep --project dynmuon-route --entity cs-439-project
uv run python experiments/lr_bowl.py --group muon_lr_sweep --group route_lr_sweep --label muon --label route
```

Outputs:

- CSV summary in `results/lr_bowls/`,
- bowl plot in `results/lr_bowls/`.

This script assumes the sweep runs already exist in W&B. It does not launch the
sweep by itself.

## 10. Cluster / RunAI Workflow

The cluster scripts are in `scripts/`.

Sync local checkout:

```bash
scripts/sync_to_rcp.sh
```

Submit data prep:

```bash
scripts/run_job.sh prep
scripts/run_job.sh prep-fineweb 500M
```

Submit smoke test:

```bash
scripts/run_job.sh sanity
```

Submit one run:

```bash
scripts/run_job.sh single --config configs/route.yaml --wandb
```

Submit experiments:

```bash
scripts/run_job.sh baselines --train-steps 20000
scripts/run_job.sh exp1 --config configs/gpt124m.yaml --train-steps 4000
scripts/run_job.sh exp2 --config configs/gpt124m.yaml
scripts/run_job.sh probe --config configs/small.yaml --steps 150
```

Inspect jobs:

```bash
scripts/run_job.sh list
scripts/run_job.sh logs <job-name>
scripts/run_job.sh delete <job-name>
```

Environment variables supported by `scripts/run_job.sh` include:

- `IMAGE` or `RUNAI_IMAGE`,
- `GPUS`,
- `CLUSTER_HOME`,
- `PROJECT_DIR`,
- `NODE_POOLS`,
- `HF_TOKEN`,
- `WANDB_API_KEY`,
- `WANDB_PROJECT`,
- `WANDB_ENTITY`,
- `UV_SYNC`,
- `UV_SYNC_ARGS`.

## 11. Sweeps

The repo has a helper for one-parameter cluster sweeps:

```bash
scripts/sweep.sh <group> <flag> <v1,v2,...> [common train.py args]
```

Beta sweep:

```bash
scripts/sweep.sh beta_sweep --beta 0,0.25,0.5,1.0 --config configs/route.yaml
```

Seed sweep:

```bash
scripts/sweep.sh seed_route --seed 0,1,2 --config configs/route.yaml
```

Proxy sweep:

```bash
scripts/sweep.sh proxy_sweep --modulate-metric stable_rank,alignment,snr --config configs/route.yaml
```

Learning-rate sweeps can be launched similarly:

```bash
scripts/sweep.sh route_muon_lr --muon-lr 0.005,0.01,0.02,0.04,0.08 --config configs/route.yaml
scripts/sweep.sh dynmuon_muon_lr --muon-lr 0.005,0.01,0.02,0.04,0.08 --config configs/dynmuon.yaml
scripts/sweep.sh adamw_lr --adam-lr 0.00015,0.0003,0.0006,0.0012 --config configs/adamw.yaml
```

After runs finish, plot bowls:

```bash
uv run python experiments/lr_bowl.py --group route_muon_lr --group dynmuon_muon_lr --label route --label dynmuon
```

## 12. What To Report

The project report should include:

- loss curves for AdamW, Muon, DynMuon, and DynMuon-Route,
- learning-rate bowl plots,
- beta sweep results,
- route contribution plot of average `p_{t,l} - p_t` by depth,
- step efficiency table,
- wall-clock or per-step timing comparison,
- validation that `compute_mode: ns` and `compute_mode: svd` match,
- discussion of random/inverted spectra after those ablations are implemented.

Current implemented outputs already cover:

- loss curves,
- step efficiency,
- per-layer `p` trajectories,
- proxy logging,
- noise robustness,
- W&B LR bowl plotting,
- `ns` vs `svd` validation.

Current missing outputs:

- dedicated `p_{t,l} - p_t` by layer-depth plot,
- random singular-value optimizer mode,
- inverted-spectrum optimizer mode,
- RelMuon optimizer mode,
- automated timing summary per optimizer,
- report generator that combines all plots/tables.

## 13. Recommended Execution Order

Start with local correctness:

```bash
uv sync
uv run pytest validate_math.py
uv run python data/prepare_wikitext.py
uv run python train.py --config configs/small.yaml --train-steps 50
```

Then calibrate routing:

```bash
uv run python experiments/probe_proxies.py --config configs/small.yaml --steps 150
```

Then run small experiments:

```bash
uv run python experiments/baselines_step_efficiency.py --train-steps 400
uv run python experiments/exp1_spectral_evolution.py --train-steps 400
uv run python experiments/exp2_noise_injection.py --train-steps 300 --noise-lambda 3.0
```

Then scale to GPT-124M / FineWeb:

```bash
uv run python data/prepare_fineweb.py 500M
uv run python train.py --config configs/gpt124m.yaml --wandb
```

Then run cluster sweeps:

```bash
scripts/sweep.sh beta_sweep --beta 0,0.25,0.5,1.0 --config configs/route.yaml
scripts/sweep.sh route_muon_lr --muon-lr 0.005,0.01,0.02,0.04,0.08 --config configs/route.yaml
```

## 14. Repo Map

```text
configs/
  base.yaml                  shared defaults
  small.yaml                 small GPT smoke config
  gpt124m.yaml               reference GPT-124M config
  adamw.yaml                 AdamW baseline
  muon.yaml                  Muon baseline
  dynmuon.yaml               DynMuon global schedule
  route.yaml                 DynMuon-Route
  exp1_spectral.yaml         spectral evolution config
  exp2_noise.yaml            noise injection config

data/
  prepare_wikitext.py        WikiText-103 token cache
  prepare_fineweb.py         FineWeb token shard download

docs/
  research_guide.md          previous working guide
  project_guide.md           this guide

experiments/
  baselines_step_efficiency.py
  exp1_spectral_evolution.py
  exp2_noise_injection.py
  probe_proxies.py
  lr_bowl.py

scripts/
  sync_to_rcp.sh
  run_job.sh
  sweep.sh
  container_entry.sh

src/
  cli.py
  config.py
  trainer.py
  analysis.py
  data/cached_tokens.py
  models/gpt.py
  optimizers/muon.py
  optimizers/dynmuon.py
  optimizers/registry.py
  optimizers/param_groups.py

train.py
validate_math.py
pyproject.toml
uv.lock
```

## 15. Practical Interpretation

Right now, the repo is a solid framework for testing whether layer-wise control of
the spectral exponent improves Muon-style GPT training.

The strongest current story is:

1. Muon fixes all update singular values to 1.
2. DynMuon replaces that with a global time-dependent exponent.
3. DynMuon-Route keeps the global clock but gives each layer a geometry-dependent correction.
4. The route correction is measured by `p_{t,l} - p_t`.
5. The relevant evidence is loss, step efficiency, LR robustness, beta sensitivity,
   route trajectories, and noise robustness.

The main gap is that random and alternative singular-value spectra are still
research ideas in this repo, not implemented baselines. They should be added before
being positioned as final empirical results.
