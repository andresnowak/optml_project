#!/usr/bin/env bash
# RunAI submission wrapper for DynMuon-Route.
#
# Subcommands:
#   sanity   50-step small-model smoke test (ns mode)
#   single   one full train.py run (pass extra args after the subcommand)
#   exp1     spectral-evolution experiment
#   exp2     anisotropic noise-injection experiment
#
# Examples:
#   scripts/run_job.sh sanity
#   scripts/run_job.sh single --config configs/gpt124m.yaml --routing-mode stable_rank --wandb
#   scripts/run_job.sh exp1 --model gpt124m --max-steps 4000
set -euo pipefail

IMAGE="${IMAGE:-pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime}"
GPUS="${GPUS:-1}"
PROJECT="${WANDB_PROJECT:-dynmuon-route}"
ENTITY="${WANDB_ENTITY:-}"
JOB_PREFIX="${JOB_PREFIX:-dynmuon}"

CMD="${1:-}"; shift || true
EXTRA=("$@")

case "$CMD" in
  sanity)
    PY=(python train.py --config configs/small.yaml --max-steps 50)
    ;;
  single)
    PY=(python train.py "${EXTRA[@]}")
    ;;
  exp1)
    PY=(python experiments/exp1_spectral_evolution.py "${EXTRA[@]}")
    ;;
  exp2)
    PY=(python experiments/exp2_noise_injection.py "${EXTRA[@]}")
    ;;
  baselines)
    PY=(python experiments/baselines_step_efficiency.py "${EXTRA[@]}")
    ;;
  *)
    echo "usage: $0 {sanity|single|exp1|exp2|baselines} [extra args]" >&2
    exit 1
    ;;
esac

JOB_NAME="${JOB_PREFIX}-${CMD}-$(date +%s)"
echo "[run_job] submitting $JOB_NAME : ${PY[*]}"

runai submit "$JOB_NAME" \
  --image "$IMAGE" \
  --gpu "$GPUS" \
  --environment WANDB_PROJECT="$PROJECT" \
  --environment WANDB_ENTITY="$ENTITY" \
  --environment PROJECT_DIR=/workspace/project \
  --command -- bash scripts/container_entry.sh "${PY[@]}"
