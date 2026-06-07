#!/usr/bin/env bash
# ============================================================
# RunAI submission wrapper for DynMuon-Route.
#
# The repo must live on the home PVC (the same filesystem the cluster mounts
# at $HOME inside the pod). This script resolves its OWN absolute path and
# hands RunAI a shell-free, absolute argv:
#     [bash, /abs/scripts/container_entry.sh, /abs/<script>.py, --flag, value, ...]
# so paths resolve regardless of the container's working directory. The entry
# script cd's into the project, sets PYTHONPATH, installs the few missing deps,
# then execs `python "$@"`.
#
# Subcommands:
#   prep        tokenize WikiText-103 -> data/wikitext103/{train,val}.bin (run once)
#   sanity      50-step small-model smoke test
#   single      one train.py run (pass train.py args after the subcommand)
#   exp1        spectral-evolution experiment
#   exp2        anisotropic noise-injection experiment
#   baselines   AdamW / Muon / DynMuon / DynMuon-Route step-efficiency
#   logs <job> | delete <job> | list
#
# Env: WANDB_API_KEY (forwarded for wandb.init), WANDB_PROJECT, WANDB_ENTITY,
#      IMAGE, GPUS, PROJECT_DIR (override the auto-resolved repo path).
#
# Examples:
#   scripts/run_job.sh prep
#   scripts/run_job.sh baselines --model gpt124m --max-steps 20000
#   scripts/run_job.sh single --config configs/route.yaml --model gpt124m --wandb
# ============================================================
set -euo pipefail

IMAGE="${IMAGE:-pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime}"
GPUS="${GPUS:-1}"

# Resolve the repo root from THIS script's location (scripts/run_job.sh), so we
# never hard-code a path that differs from where the repo lives on the PVC.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${_SELF_DIR}/.." && pwd)}"
ENTRY_SH="${PROJECT_DIR}/scripts/container_entry.sh"
STAMP="$(date +%Y%m%d-%H%M%S)-${RANDOM}"   # random suffix avoids name clashes on rapid submits

# Mount the home PVC so the repo + data + wandb cache persist and are visible
# inside the pod at the same absolute path as on the submit node.
BASE_FLAGS="--image ${IMAGE} --pvc home:${HOME} -e HOME=${HOME} --run-as-user --gpu ${GPUS}"
[ -n "${HF_TOKEN:-}" ]      && BASE_FLAGS="${BASE_FLAGS} -e HF_TOKEN=${HF_TOKEN}"
[ -n "${WANDB_API_KEY:-}" ] && BASE_FLAGS="${BASE_FLAGS} -e WANDB_API_KEY=${WANDB_API_KEY}"
[ -n "${WANDB_PROJECT:-}" ] && BASE_FLAGS="${BASE_FLAGS} -e WANDB_PROJECT=${WANDB_PROJECT}"
[ -n "${WANDB_ENTITY:-}" ]  && BASE_FLAGS="${BASE_FLAGS} -e WANDB_ENTITY=${WANDB_ENTITY}"
BASE_FLAGS="${BASE_FLAGS} -e WANDB_DIR=${WANDB_DIR:-${HOME}/.wandb}"

# _submit <job-name> <script-abs-path> [args...]
_submit() {
    local job="$1"; shift
    echo "submitting ${job}:"
    printf '  bash %s' "${ENTRY_SH}"; printf ' %q' "$@"; printf '\n'
    runai submit ${BASE_FLAGS} --name "${job}" \
        --command -- bash "${ENTRY_SH}" "$@"
}

CMD="${1:-}"; shift || true
case "${CMD}" in
  prep)
    _submit "dynmuon-prep-${STAMP}" "${PROJECT_DIR}/scripts/prepare_wikitext.py" ;;
  probe)
    _submit "dynmuon-probe-${STAMP}" "${PROJECT_DIR}/experiments/probe_proxies.py" "$@" ;;
  sanity)
    _submit "dynmuon-sanity-${STAMP}" "${PROJECT_DIR}/train.py" --config configs/small.yaml --max-steps 50 ;;
  single)
    _submit "dynmuon-single-${STAMP}" "${PROJECT_DIR}/train.py" "$@" ;;
  exp1)
    _submit "dynmuon-exp1-${STAMP}" "${PROJECT_DIR}/experiments/exp1_spectral_evolution.py" "$@" ;;
  exp2)
    _submit "dynmuon-exp2-${STAMP}" "${PROJECT_DIR}/experiments/exp2_noise_injection.py" "$@" ;;
  baselines)
    _submit "dynmuon-baselines-${STAMP}" "${PROJECT_DIR}/experiments/baselines_step_efficiency.py" "$@" ;;
  logs)
    runai logs "${1:?usage: $0 logs <job-name>}" ;;
  delete)
    runai delete job "${1:?usage: $0 delete <job-name>}" ;;
  list)
    runai list ;;
  *)
    echo "usage: $0 {prep|probe|sanity|single|exp1|exp2|baselines|logs|delete|list} [args]" >&2
    exit 1 ;;
esac
