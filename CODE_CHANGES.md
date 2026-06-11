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
