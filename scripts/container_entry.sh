#!/usr/bin/env bash
# ============================================================
# Cluster container entry. Invoked by run_job.sh, not locally.
#
# Invocation (after RunAI/k8s flattens argv):
#   bash /abs/scripts/container_entry.sh /abs/<script>.py --flag value ...
#
# Resolves the project dir from this script's own location, cd's in so relative
# paths the scripts use (configs/, data/, results/) resolve, sets PYTHONPATH so
# `import dynmuon` works, installs the few deps the pytorch base image lacks,
# then execs `python "$@"`. Deps install to $HOME/.local (on the PVC), so the
# next job on the same image finds them already present.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

python -m pip install --quiet --no-cache-dir --user --disable-pip-version-check \
    pyyaml tiktoken datasets wandb matplotlib

exec python "$@"
