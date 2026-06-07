#!/usr/bin/env bash
# ============================================================
# RunAI submission wrapper for optimizer experiments.
#
# The repo must live on the home PVC as seen by the RunAI pod. This script can
# be run from the cluster checkout directly, or from a local machine after
# syncing to RCP:
#   CLUSTER_HOME=/home/nowak PROJECT_DIR=/home/nowak/developer/optml_project
#
# It resolves its OWN absolute path and hands RunAI a shell-free, absolute argv:
#     [bash, /abs/scripts/container_entry.sh, /abs/<script>.py, --flag, value, ...]
# so paths resolve regardless of the container's working directory. The entry
# script cd's into the project, sets PYTHONPATH, syncs deps with uv, then execs
# `uv run --no-sync python "$@"`.
#
# Subcommands:
#   prep        tokenize WikiText-103 -> data/wikitext103/{train,val}.bin
#   prep-fineweb download FineWeb10B GPT-2 token shards, e.g. prep-fineweb 500M
#   sanity      50-step small-model smoke test
#   single      one train.py run (pass train.py args after the subcommand)
#   exp1        spectral-evolution experiment
#   exp2        anisotropic noise-injection experiment
#   baselines   AdamW / Muon / DynMuon / DynMuon-Route step-efficiency
#   logs <job> | delete <job> | list
#
# Env: .env is loaded if present. Common overrides:
#      JOB_PREFIX, IMAGE/RUNAI_IMAGE, GPUS, CLUSTER_HOME, PROJECT_DIR, NODE_POOLS,
#      LDAP_UID/LDAP_GID, PYTHONUNBUFFERED, UV_SYNC, UV_SYNC_ARGS, HF_TOKEN,
#      WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY.
#
# Examples:
#   scripts/run_job.sh prep
#   scripts/run_job.sh prep-fineweb 500M
#   scripts/run_job.sh baselines --train-steps 20000
#   scripts/run_job.sh single --config configs/route.yaml --wandb
# ============================================================
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env}"
if [ -f "${ENV_FILE}" ]; then
    while IFS='=' read -r key value; do
        case "${key}" in
            ''|\#*) continue ;;
        esac
        key="${key%%[[:space:]]*}"
        value="${value%%[[:space:]]#*}"
        value="${value%"${value##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        if [ -n "${key}" ] && [ -z "${!key+x}" ]; then
            export "${key}=${value}"
        fi
    done < "${ENV_FILE}"
fi

# Resolve the repo root from THIS script's location (scripts/run_job.sh), so we
# can submit from the cluster checkout while still allowing PROJECT_DIR to point
# at the path as seen inside the pod.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_DIR="$(cd "${_SELF_DIR}/.." && pwd)"

IMAGE="${IMAGE:-${RUNAI_IMAGE:-ic-registry.epfl.ch/mlo/mlo-base:uv1}}"
GPUS="${GPUS:-1}"
REMOTE_USER="${REMOTE_USER:-nowak}"
if [ -z "${CLUSTER_HOME+x}" ] && [[ "${LOCAL_PROJECT_DIR}" == /Users/* ]]; then
    CLUSTER_HOME="/home/${REMOTE_USER}"
else
    CLUSTER_HOME="${CLUSTER_HOME:-${HOME}}"
fi
if [ -z "${PROJECT_DIR+x}" ] && [[ "${LOCAL_PROJECT_DIR}" == /Users/* ]]; then
    PROJECT_DIR="${CLUSTER_HOME}/developer/$(basename "${LOCAL_PROJECT_DIR}")"
else
    PROJECT_DIR="${PROJECT_DIR:-${LOCAL_PROJECT_DIR}}"
fi
ENTRY_SH="${PROJECT_DIR}/scripts/container_entry.sh"
JOB_PREFIX="${JOB_PREFIX:-optml}"
STAMP="$(date +%Y%m%d-%H%M%S)-${RANDOM}"   # random suffix avoids name clashes on rapid submits

# Mount the home PVC so the repo + data + wandb cache persist and are visible
# inside the pod.
if [ -n "${LDAP_UID:-}" ] && [ -n "${LDAP_GID:-}" ]; then
    RUN_AS_FLAGS="--run-as-uid ${LDAP_UID} --run-as-gid ${LDAP_GID}"
else
    RUN_AS_FLAGS="--run-as-user"
fi
BASE_FLAGS="--image ${IMAGE} --pvc home:${CLUSTER_HOME} -e HOME=${CLUSTER_HOME} ${RUN_AS_FLAGS} --gpu ${GPUS}"
if [ -n "${NODE_POOLS:-}" ]; then
    BASE_FLAGS="${BASE_FLAGS} --node-pools ${NODE_POOLS}"
fi
[ -n "${HF_TOKEN:-}" ]      && BASE_FLAGS="${BASE_FLAGS} -e HF_TOKEN=${HF_TOKEN}"
[ -n "${WANDB_API_KEY:-}" ] && BASE_FLAGS="${BASE_FLAGS} -e WANDB_API_KEY=${WANDB_API_KEY}"
[ -n "${WANDB_PROJECT:-}" ] && BASE_FLAGS="${BASE_FLAGS} -e WANDB_PROJECT=${WANDB_PROJECT}"
[ -n "${WANDB_ENTITY:-}" ]  && BASE_FLAGS="${BASE_FLAGS} -e WANDB_ENTITY=${WANDB_ENTITY}"
BASE_FLAGS="${BASE_FLAGS} -e WANDB_DIR=${WANDB_DIR:-${CLUSTER_HOME}/.wandb}"
BASE_FLAGS="${BASE_FLAGS} -e PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}"
BASE_FLAGS="${BASE_FLAGS} -e UV_CACHE_DIR=${UV_CACHE_DIR:-${CLUSTER_HOME}/.cache/uv}"
BASE_FLAGS="${BASE_FLAGS} -e UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-${CLUSTER_HOME}/.uv}"
[ -n "${UV_SYNC:-}" ]      && BASE_FLAGS="${BASE_FLAGS} -e UV_SYNC=${UV_SYNC}"
[ -n "${UV_SYNC_ARGS:-}" ] && BASE_FLAGS="${BASE_FLAGS} -e UV_SYNC_ARGS=${UV_SYNC_ARGS}"

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
    _submit "${JOB_PREFIX}-prep-${STAMP}" "${PROJECT_DIR}/data/prepare_wikitext.py" ;;
  prep-fineweb)
    _submit "${JOB_PREFIX}-prep-fineweb-${STAMP}" "${PROJECT_DIR}/data/prepare_fineweb.py" "${1:-500M}" ;;
  probe)
    _submit "${JOB_PREFIX}-probe-${STAMP}" "${PROJECT_DIR}/experiments/probe_proxies.py" "$@" ;;
  sanity)
    _submit "${JOB_PREFIX}-sanity-${STAMP}" "${PROJECT_DIR}/train.py" --config configs/small.yaml --train-steps 50 ;;
  single)
    _submit "${JOB_PREFIX}-single-${STAMP}" "${PROJECT_DIR}/train.py" "$@" ;;
  exp1)
    _submit "${JOB_PREFIX}-exp1-${STAMP}" "${PROJECT_DIR}/experiments/exp1_spectral_evolution.py" "$@" ;;
  exp2)
    _submit "${JOB_PREFIX}-exp2-${STAMP}" "${PROJECT_DIR}/experiments/exp2_noise_injection.py" "$@" ;;
  baselines)
    _submit "${JOB_PREFIX}-baselines-${STAMP}" "${PROJECT_DIR}/experiments/baselines_step_efficiency.py" "$@" ;;
  logs)
    runai logs "${1:?usage: $0 logs <job-name>}" ;;
  delete)
    runai delete job "${1:?usage: $0 delete <job-name>}" ;;
  list)
    runai list ;;
  *)
    echo "usage: $0 {prep|prep-fineweb|probe|sanity|single|exp1|exp2|baselines|logs|delete|list} [args]" >&2
    exit 1 ;;
esac
