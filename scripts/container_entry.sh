#!/bin/bash
# ============================================================
# Cluster container entry. Run by run_job.sh — not invoked locally.
#
# This script makes the cluster image behave like the local dev setup:
# install `uv` if the image does not already provide it, sync from uv.lock,
# set PYTHONPATH so `from src.cli import main` resolves regardless of cwd,
# then exec `uv run --no-sync python "$@"` with the rest of the argv.
#
# Invocation pattern (from run_job.sh):
#   runai submit ... --command -- bash /path/to/container_entry.sh \
#       /path/to/main.py --experiment shakespeare ...
#
# argv after k8s flattens it stays clean (no quoting, no metacharacters):
#   [bash, .../container_entry.sh, .../main.py, --experiment, shakespeare, ...]
# ============================================================
set -euo pipefail

# PROJECT_DIR = the parent of this script's directory.
# Resolves correctly whether the script is invoked by absolute or relative path.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PATH="${HOME}/.local/bin:${PATH}"
export USER="${USER:-nowak}"
export LOGNAME="${LOGNAME:-${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}" # NOTE: It will use your home as the cache dir.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"

cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --quiet --no-cache-dir --user --disable-pip-version-check uv
fi

# Keep the virtualenv on the PVC-backed checkout so later jobs can reuse it.
# Set UV_SYNC=0 to skip this when using a pre-synced custom image.
UV_SYNC="${UV_SYNC:-1}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---locked}"
if [ "${UV_SYNC}" = "1" ]; then
    uv sync ${UV_SYNC_ARGS}
fi

exec uv run --no-sync python "$@"
