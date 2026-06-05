#!/usr/bin/env bash
# Container entrypoint for cluster runs. Installs deps into the base PyTorch
# image, sets PYTHONPATH, cd's to the project dir, then execs the given command.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/project}"
cd "$PROJECT_DIR"

pip install --no-cache-dir numpy pyyaml tiktoken datasets wandb matplotlib >/dev/null

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "[container_entry] running: $*"
exec "$@"
