#!/usr/bin/env bash
# ============================================================
# Cluster container entry. Invoked by run_job.sh, not locally.
#
# Invocation (after RunAI/k8s flattens argv):
#   bash /abs/scripts/container_entry.sh /abs/<script>.py --flag value ...
#
# Resolves the project dir from this script's own location, cd's in so relative
# paths the scripts use (configs/, data/, results/) resolve, sets PYTHONPATH so
# `import src` works, syncs deps with uv, then execs
# `uv run --no-sync python "$@"`.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PATH="${HOME}/.local/bin:${PATH}"
export USER="${USER:-nowak}"
export LOGNAME="${LOGNAME:-${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"

cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --quiet --no-cache-dir --user --disable-pip-version-check uv
fi

UV_SYNC="${UV_SYNC:-1}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---locked}"
if [ "${UV_SYNC}" = "1" ]; then
    uv sync ${UV_SYNC_ARGS}
fi

exec uv run --no-sync python "$@"
