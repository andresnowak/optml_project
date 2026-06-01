#!/bin/bash
# ============================================================
# Cluster container entry. Run by run_job.sh — not invoked locally.
#
# The pytorch base image ships torch + numpy; this script adds the three
# small pure-Python deps (`wandb`, `pyyaml`, `matplotlib`) we actually need,
# sets PYTHONPATH so `from src.cli import main` resolves regardless of cwd,
# then execs `python "$@"` with the rest of the argv. metalcore is NOT
# installed: it's only needed on MPS, and src/{optimizers,experiments}.py
# already import it inside a try/except so the cuda code path is clean.
#
# Invocation pattern (from run_job.sh):
#   runai submit ... --command -- bash /path/to/container_entry.sh \
#       /path/to/main.py --experiment shakespeare ...
#
# argv after k8s flattens it stays clean (no quoting, no metacharacters):
#   [bash, .../container_entry.sh, .../main.py, --experiment, shakespeare, ...]
# ============================================================
set -e

# PROJECT_DIR = the parent of this script's directory.
# Resolves correctly whether the script is invoked by absolute or relative path.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Install the few deps the pytorch image doesn't have. `--user` keeps them
# in $HOME/.local (on the PVC), so subsequent jobs on the same image find
# them already installed and the next call is a no-op.
python -m pip install --quiet --no-cache-dir --user --disable-pip-version-check \
    wandb pyyaml matplotlib

exec python "$@"
