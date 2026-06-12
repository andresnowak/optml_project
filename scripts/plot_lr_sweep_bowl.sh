#!/usr/bin/env bash
# Plot the main LR sweep bowl and AdamW-target speed charts from the five W&B sweep projects.
#
# Usage:
#   scripts/plot_lr_sweep_bowl.sh [extra experiments/lr_bowl.py args]
#
# Example:
#   scripts/plot_lr_sweep_bowl.sh
#   scripts/plot_lr_sweep_bowl.sh --selection best --out-dir results/lr_bowls/best_marked
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

uv run python experiments/lr_bowl.py \
  --sweep-projects \
    relmuon-rms-lr-sweep \
    adam-lr-sweep \
    relmuon-log1p-lr-sweep \
    dynmuon-route-lr-sweep \
    muon-lr-sweep \
  --labels \
    RelMuon-RMS \
    AdamW \
    RelMuon-log1p \
    DynMuon-Route \
    Muon \
  --selection final \
  --exclude-lr-range AdamW:1e-4:1e-3 \
  --out-dir results/lr_bowls/adamw_filtered_star_best \
  "$@"
