# Research Guide: Muon, DynMuon, and Layer-wise Routing

This document is the working guide for the current project direction.

The short version is:

- SpecMuon is not the main line right now.
- The active line is Muon-style spectral shaping of matrix updates on GPT.
- We are studying how the singular values of the update matrix should be handled.
- The main comparison is between a global spectral schedule and a per-layer routed version.
- We also keep proxy calibration, step-efficiency, and noise-robustness experiments in the loop.

## 1. What we are trying to test

The optimizer acts on a matrix update $M = U \Sigma V^T$ by changing the singular values through a spectral exponent $p$:

$$
D(p) = U \Sigma^p V^T
$$

The project question is not just whether to flatten the spectrum completely.
Instead, we want to test whether a controlled choice of $p$ can better preserve useful geometry while still suppressing harmful directions.

The current codebase supports these ideas:

- $p = 0$ gives the Muon polar factor.
- A global schedule moves $p$ from $1$ toward $-0.25$ over training.
- A routed variant lets each layer deviate from that shared schedule.
- Different proxies can drive routing: stable rank, SNR, or alignment.

The conceptual research direction is:

- compare fixed, scheduled, and routed spectral shaping,
- understand whether per-layer routing is actually useful,
- measure whether the layer-specific deviation in $p$ aligns with gradient geometry,
- and compare against alternative spectral treatments, including random or inverted singular-value baselines when those are added.

## 2. What is implemented in the repo

The live code path is centered on a GPT training loop in `src/trainer.py`, a GPT model in `src/models/gpt.py`, and optimizer construction in `src/optimizers/registry.py`.

### Training entry point

- `train.py` calls `src.cli.main()`.
- `src/cli.py` loads a YAML config and applies command-line overrides.
- `src/config.py` resolves `extends:` chains between YAML files and picks the device.
- `src/trainer.py` builds the model, loads token data, creates the optimizers, runs training, validation, and logging.

### Model

The GPT is a Track-3-style transformer:

- separate `q`, `k`, `v`, and `proj` matrices in attention,
- RMSNorm,
- RoPE,
- ReLU-squared MLP,
- tied output embedding.

This matters because the parameter names are stable and routing can distinguish attention and MLP matrices.

### Optimizers

The project has three matrix-optimizer modes:

- `adamw`: matrix parameters use AdamW.
- `muon`: fixed Muon behavior, equivalent to `routing_mode: fixed` with `fixed_p: 0`.
- `dynmuon`: spectral exponent follows a global time schedule.
- `route`: the scheduled version with layer-wise modulation.

The matrix optimizer is selected through `matrix_optimizer` and `routing_mode` in YAML.

## 3. How the spectral routing works

### Global schedule

The reference DynMuon schedule is a logistic anneal from $p = 1$ toward $p = -0.25$.

In code, that is the `global_schedule` routing mode in `src/optimizers/dynmuon.py`.

### Routed version

The routed version keeps the same global clock and adds a per-layer correction:

$$
p_{t,l} = \mathrm{clip}(p_t + \beta \cdot (g_l - \tilde g_t), -0.25, 1.0)
$$

The code implements this as `schedule_modulated`.

The important details are:

- `p_t` is the shared clock.
- `g_l` is a layer-local proxy.
- `\tilde g_t` is the running network average when `dynamic_ref: true`.
- `beta` controls how strongly a layer can deviate from the clock.

The routed optimizer logs the actual per-parameter values in `state[p]` so the training loop can record trajectories for later plots.

### Proxies

The implemented proxies are:

- stable rank,
- SNR proxy,
- alignment proxy.

The default routing proxy is stable rank, but the config also exposes logistic modes for the other two.

### Exact vs fast spectral backend

The optimizer can shape the update in two ways:

- `compute_mode: svd` uses exact SVD.
- `compute_mode: ns` uses the Newton-Schulz / eigendecomposition factorization.

This is validated by `validate_math.py`, which checks that the two paths agree numerically for the supported cases.

## 4. What the experiments are meant to answer

### Baseline step efficiency

File: `experiments/baselines_step_efficiency.py`

This compares:

- AdamW,
- Muon,
- DynMuon,
- DynMuon-Route.

The script:

- trains all four methods on the same config and seed,
- writes `results/baselines/history_*.json`,
- generates `results/baselines/loss_curves.png`,
- writes `results/baselines/summary.md`.

This is the main step-efficiency comparison.

### Experiment 1: spectral evolution

File: `experiments/exp1_spectral_evolution.py`

This compares two routing modes on GPT:

- `global_schedule`,
- `schedule_modulated`.

It plots the mean routed exponent over time for:

- attention `q`, `k`, `v`, and `proj`,
- MLP `fc` and `proj`.

The key question is whether attention resists negative $p$ while MLP layers move more aggressively toward the low-$p$ regime.

Outputs:

- `results/exp1_spectral_evolution/history_global_schedule.json`
- `results/exp1_spectral_evolution/history_schedule_modulated.json`
- `results/exp1_spectral_evolution/p_trajectories.png`

### Experiment 2: noise injection

File: `experiments/exp2_noise_injection.py`

This injects anisotropic noise into the momentum buffer:

$$
M \leftarrow M + \lambda \sigma_1 z u_1 v_1^T
$$

The idea is to create a spike along the dominant singular direction and see whether the router reacts by pushing $p$ downward.

It compares:

- `global_schedule`,
- `schedule_modulated`.

Outputs:

- `results/exp2_noise_injection/history_global_schedule.json`
- `results/exp2_noise_injection/history_schedule_modulated.json`
- `results/exp2_noise_injection/noise_stability.png`

### Proxy calibration

File: `experiments/probe_proxies.py`

This is a short run used to inspect the proxy distributions before launching long jobs.

It prints suggested calibration values for:

- `ref` in `schedule_modulated`,
- `mu` and `omega` for logistic proxy routing.

Use this before tuning `beta` or changing the proxy metric.

### Learning-rate bowls

File: `experiments/lr_bowl.py`

This is the plotting tool for LR sweep groups stored in W&B.

It pulls runs by group, extracts `adam_lr`, `muon_lr`, or `lr`, and writes a CSV plus a bowl plot.

This is the right tool for the LR sweep graphs requested in the project notes.

## 5. How the configs are organized

The repo uses `extends:` in YAML.

### Base config

File: `configs/base.yaml`

This contains the shared defaults:

- model shape,
- optimizer choice,
- routing parameters,
- data paths,
- batch sizes,
- warmup and cosine decay,
- logging settings.

### Model-specific configs

- `configs/gpt124m.yaml` for the reference GPT setup on FineWeb.
- `configs/small.yaml` for smoke tests and fast iteration.

### Method configs

- `configs/adamw.yaml`
- `configs/muon.yaml`
- `configs/dynmuon.yaml`
- `configs/route.yaml`

### Experiment configs

- `configs/exp1_spectral.yaml`
- `configs/exp2_noise.yaml`

## 6. How to run the experiments locally

The repo is set up to run with `uv`.

### Install and verify

```bash
uv sync
uv run pytest validate_math.py
```

### Prepare data

WikiText-103:

```bash
uv run python data/prepare_wikitext.py
```

FineWeb 500M cache:

```bash
uv run python data/prepare_fineweb.py 500M
```

### Smoke test

```bash
uv run python train.py --config configs/small.yaml --train-steps 50
```

### Baselines

```bash
uv run python experiments/baselines_step_efficiency.py
```

Optional overrides:

```bash
uv run python experiments/baselines_step_efficiency.py --train-steps 20000
uv run python experiments/baselines_step_efficiency.py --target-loss 4.0
```

### Spectral evolution

```bash
uv run python experiments/exp1_spectral_evolution.py
```

Example with overrides:

```bash
uv run python experiments/exp1_spectral_evolution.py --config configs/exp1_spectral.yaml --train-steps 400
```

### Noise injection

```bash
uv run python experiments/exp2_noise_injection.py
```

Example with overrides:

```bash
uv run python experiments/exp2_noise_injection.py --config configs/exp2_noise.yaml --train-steps 300 --noise-lambda 3.0
```

### Proxy probe

```bash
uv run python experiments/probe_proxies.py --config configs/small.yaml --steps 150
```

### LR bowls from W&B

```bash
uv run python experiments/lr_bowl.py --group <group-name> --project dynmuon-route-sweeps --entity cs-439-project
```

## 7. How to run on RunAI / RCP

The repo already ships with cluster scripts.

### Sync the checkout

```bash
scripts/sync_to_rcp.sh
```

Dry run:

```bash
DRY_RUN=1 scripts/sync_to_rcp.sh
```

### Submit jobs

```bash
scripts/run_job.sh prep
scripts/run_job.sh prep-fineweb 500M
scripts/run_job.sh sanity
scripts/run_job.sh single --config configs/route.yaml --wandb
scripts/run_job.sh baselines --train-steps 20000
scripts/run_job.sh exp1 --config configs/gpt124m.yaml --train-steps 4000
scripts/run_job.sh exp2 --config configs/gpt124m.yaml
scripts/run_job.sh probe
scripts/run_job.sh logs <job-name>
scripts/run_job.sh list
scripts/run_job.sh delete <job-name>
```

The submission flow is:

1. Sync the repository to the remote host.
2. Submit the RunAI job.
3. Let `scripts/container_entry.sh` resolve the project directory, sync dependencies with `uv`, and execute the requested Python script.

## 8. What to look at in the outputs

The important artifacts are:

- `results/baselines/summary.md` for step-efficiency numbers,
- `results/baselines/loss_curves.png` for the validation curves,
- `results/exp1_spectral_evolution/p_trajectories.png` for layer-wise $p_{t,l}$ behavior,
- `results/exp2_noise_injection/noise_stability.png` for the noise stress test,
- `results/lr_bowls/*.csv` and `*.png` for sweep bowls,
- raw `history_*.json` files for offline analysis.

The logged routing metrics are:

- `route/p/<parameter>`,
- `route/sr/<parameter>`,
- `route/gamma/<parameter>`,
- `route/alpha/<parameter>`.

## 9. What is archival, not the current mainline

There are older result folders under `results/gate_sensitivity/` and `results/sav_isolation/`.

These are useful as background evidence for singular-value handling, especially around top-k style or SAV-style ideas, but they are not wired into the current GPT training pipeline.

Likewise, `docs/new_proposals.md` is a proposal note, not executable code.

The current codebase does not contain a live SpecMuon implementation, so any SpecMuon-specific conclusions should be treated as historical or conceptual unless they are reintroduced into `src/`.

## 10. Current working hypothesis

The hypothesis we are testing is:

- the shared global clock captures the universal training-time trend,
- the per-layer deviation from the proxy average captures local geometry,
- and that local correction improves behavior on GPT without overcomplicating the optimizer.

The practical test is whether the routed method is better than the global schedule on:

- step efficiency,
- loss curves,
- proxy separation,
- and noise robustness.

If the routed method does not outperform the global schedule, the next question is whether the proxy, the calibration, or the route design itself is the wrong abstraction.

## 11. Recommended execution order

If you are starting fresh, use this order:

1. Run `uv run pytest validate_math.py`.
2. Prepare WikiText with `uv run python data/prepare_wikitext.py`.
3. Run the smoke test with `uv run python train.py --config configs/small.yaml --train-steps 50`.
4. Run `uv run python experiments/probe_proxies.py --config configs/small.yaml --steps 150`.
5. Run the baseline comparison.
6. Run Exp 1 and Exp 2.
7. Only then start the larger W&B sweeps or RCP jobs.

## 12. Open implementation gaps

These are the main gaps between the current code and the broader research agenda:

- no live SpecMuon branch in `src/` yet,
- no live random-singular-value ablation yet,
- no live inverted-spectrum baseline yet,
- no dedicated top-k SAV route in the current trainer,
- no automated report generator yet,
- no end-to-end sweep orchestration beyond the W&B/LR-bowl utilities.

That means the repo is already good for the Muon/DynMuon/Route study on GPT, but the more exotic singular-value variants still need to be added explicitly if we want them compared in the same framework.
