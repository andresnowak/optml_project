# Code Changes

This file tracks implementation changes made during result diagnosis, separate
from the report. Each entry records what changed, why, and how it was checked.

## 2026-06-11 - W&B pull and route-control cleanup

### Added remote W&B pull helper

- File: `experiments/pull_wandb.py`
- Change: added `pull_wandb_logs(...)` as the programmatic entry point.
- Change: added `--url` parsing for W&B workspace/run URLs.
- Change: summary CSV rows now include `run_id`, `run_url`, and effective route
  defaults such as `beta`, `lean_norm`, `lean_max`, `modulate_metric`,
  `dynamic_ref`, and `ref_decay`.
- Why: remote W&B runs need to be reproducibly pulled into the local
  `MemoryLogger` history format so existing analysis scripts work unchanged.
  The old summary left default route knobs blank when they came from nested YAML
  config rather than CLI overrides.

### Made beta-zero route arm a cleaner DynMuon control

- File: `src/optimizers/registry.py`
- Change: `routing_mode: schedule_modulated` with effective `beta == 0.0` now
  builds the same single matrix parameter group as `global_schedule`.
- Why: the `route_beta0` sweep arm is intended as a DynMuon schedule control.
  With `beta=0`, the optimizer math already reduces to the global schedule, but
  using routed attn/mlp/other groups made the control less identical at the
  builder level and harder to reason about.

### Added regression coverage

- File: `validate_math.py`
- Change: added `test_beta_zero_uses_global_schedule_grouping`.
- Why: preserve the intended sweep invariant: `beta=0` uses global-schedule
  grouping, while nonzero beta keeps routed groups.

### Verification

```bash
uv run pytest validate_math.py -q
```

Result observed: `85 passed`.

## 2026-06-11 - Integration-test merge and report pass

### Merged Luca's integration-test branch

- Source branch: `origin/luca/integration-test`
- Change: fast-forwarded the current branch to include the spatial ablation
  runner, new RelMuon/spectrum configs, and optimizer/trainer updates.
- Why: the report now needs to account for the additional experiment that
  applies RelMuon only to attention matrices.

### Hardened attention-only RelMuon experiment

- File: `experiments/spatial_ablation.py`
- Change: made the spatial ablation select attention matrices via the shared
  `layer_type()` helper (`.attn.`), remove empty matrix-optimizer parameter
  groups, sort parameter lists deterministically, and preserve the normal
  optimizer builder before filtering.
- Why: the merged experiment had the right intent but used a loose
  `'attn' in name` check and could leave empty optimizer groups. The updated
  runner is safer and keeps the full RelMuon optimizer settings intact.

- File: `configs/relmuon_attention.yaml`
- Change: added a named config for RelMuon-log1p on attention matrices only.
- Why: gives the experiment a stable, artifact-friendly name and prevents
  accidental use of the normal `single` entry point.

- File: `scripts/sweeps.sh`
- Change: added a `relmuon-attention` phase and included it in `final`.
- Why: makes the attention-only efficiency ablation reproducible alongside the
  full RelMuon bowls.

### Relaxed and corrected report narrative

- File: `report/main.tex`
- Change: rewrote the report around completed W&B results, removed `TBD`
  placeholders, reduced failure-forensics detail in the main text, and framed
  RelMuon as a fixed-budget spectral bias.
- Why: the previous version was dense and contained unsupported placeholders.
  The new version only draws numerical conclusions from completed runs and
  explicitly marks RelMuon/attention-only RelMuon as integrated experiments
  awaiting full-length W&B results.

- File: `report/literature.bib`
- Change: added Transformer and LoRA references.
- Why: these support the attention-only ablation motivation: attention is the
  transformer token-mixing mechanism, and targeted low-dimensional updates are
  a standard efficiency idea in large-model adaptation.

### Confirmed W&B project separation

- File: `experiments/pull_wandb.py`
- Change: kept the default W&B project as `dynmuon-route-sweeps`.
- Why: `dynmuon-route` contains another person's runs and should not be mixed
  into this report's evidence base.

- File: `scripts/sweeps.sh`
- Change: kept the default `SWEEP_PROJECT` as `dynmuon-route-sweeps`.
- Why: future submitted runs should land in the project used by this report.

- File: `report/main.tex`
- Change: updated the RelMuon-status paragraph to refer only to
  `dynmuon-route-sweeps`.
- Why: cross-project runs are not comparable evidence for this report.

## 2026-06-11 - LR sweep figure correction

- File: `experiments/report_figures.py`
- Change: renamed the main LR plot to `lr_sweep` while keeping `bowls` and
  `lr_bowls` as backward-compatible aliases; boundary best points are now
  marked with hollow edge markers and annotated as boundary optima.
- Why: the completed Muon and DynMuon points are not true bowls yet: their best
  sampled LR is the lowest sampled matrix LR (`0.02`).

- File: `scripts/sweeps.sh`
- Change: expanded `bowls-left` to run Muon/DynMuon at `0.001`, `0.002`,
  `0.005`, and `0.01`.
- Why: these points are needed to demonstrate whether validation loss rises
  again on the low-LR side and to bracket the optimum.

- File: `report/main.tex`
- Change: changed the language from "learning-rate bowls" to
  "learning-rate sweeps" and made the boundary limitation explicit.
- Why: the report should not imply that a sampled boundary minimum is a
  demonstrated bowl.

## 2026-06-11 - Loss and routing figures

- File: `experiments/report_figures.py`
- Change: added `losses`, which generates separate late-training train and
  validation loss plots with log-scaled y axes.
- Why: the old single validation plot was visually dominated by the shared
  initialization spike and made optimizer differences hard to read.

- File: `report/main.tex`
- Change: replaced the single loss-curve figure with `train_loss_late` and
  `val_loss_late`, and added the `depth_routing` figure.
- Why: separate train/validation plots make the Muon/DynMuon/decoupled
  trajectories easier to compare, and the depth plot shows how routing varies
  across block depth and matrix type.

## 2026-06-11 - Numeric tables and RelMuon comparison command

- File: `report/main.tex`
- Change: added a numeric comparison table and a step/wall-clock time-to-target
  table in the main body.
- Why: the strongest comparisons are clearer as numbers than as additional
  plots, especially for final validation loss and seconds-per-step.

- File: `report/main.tex`
- Change: expanded the appendix with the full mathematical framework:
  momentum conventions, spectral update family, magnitude decoupling, routing
  interpretation, RelMuon, and attention-only RelMuon.
- Why: the appendix can carry the detailed derivation and numerical-check
  figure without crowding the main narrative.

- File: `scripts/sweeps.sh`
- Change: added `relmuon-compare`, a matched full-vs-attention-only
  RelMuon-log1p sweep over the same LR grid, W&B project, and train budget.
- Why: this is the clean cluster command for comparing full RelMuon against
  attention-only RelMuon.
