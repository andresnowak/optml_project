#!/usr/bin/env bash
# Submit a one-parameter sweep as separate W&B runs in a shared group.
#
# Usage:
#   scripts/sweep.sh <group> <flag> <v1,v2,...> [common train.py args]
#
# Examples:
#   scripts/sweep.sh beta_sweep --beta 0,0.25,0.5,1.0 --config configs/route.yaml --model gpt124m
#   scripts/sweep.sh seed_route --seed 0,1,2 --config configs/route.yaml --model gpt124m
#   scripts/sweep.sh seed_dynmuon --seed 0,1,2 --config configs/dynmuon.yaml --model gpt124m
#   scripts/sweep.sh proxy --modulate-metric stable_rank,alignment --config configs/route.yaml --model gpt124m
#
# Each value becomes one `run_job.sh single` job: --wandb, --wandb-group <group>,
# --run-name <group>_<value>, plus the flag and the common args.
set -euo pipefail

GROUP="${1:?usage: $0 <group> <flag> <v1,v2,...> [common args]}"
FLAG="${2:?missing flag, e.g. --beta}"
VALUES="${3:?missing comma-separated values}"
shift 3

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IFS=',' read -ra VS <<< "${VALUES}"
for v in "${VS[@]}"; do
    echo ">>> ${GROUP}: ${FLAG} ${v}"
    "${HERE}/run_job.sh" single --wandb --wandb-group "${GROUP}" \
        --run-name "${GROUP}_${v}" "${FLAG}" "${v}" "$@"
done
