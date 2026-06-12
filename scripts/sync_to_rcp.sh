#!/bin/bash
# Sync this checkout to the RunAI submit host.
#
# Defaults match Andres' jhrcp setup:
#   local repo: directory containing this script's parent
#   remote:     jhrcp:/home/nowak/developer/optml_project
#
# Override examples:
#   REMOTE_HOST=myhost REMOTE_DIR=/home/me/developer/optml_project ./scripts/sync_to_rcp.sh
#   DRY_RUN=1 ./scripts/sync_to_rcp.sh

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${LOCAL_DIR:-$(cd "${SELF_DIR}/.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-jhrcp}"
REMOTE_USER="${REMOTE_USER:-nowak}"
REMOTE_DIR="${REMOTE_DIR:-/home/${REMOTE_USER}/developer/optml_project}"
DRY_RUN="${DRY_RUN:-0}"

RSYNC_ARGS=(-az --delete)
if [ "${DRY_RUN}" = "1" ]; then
    RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

EXCLUDES=(
    --exclude .git
    --exclude .venv
    --exclude .DS_Store
    --exclude .pytest_cache
    --exclude __pycache__
    --exclude wandb
    --exclude data/fineweb10B
    --exclude data/wikitext103
    --exclude '*.pyc'
)

echo "Syncing ${LOCAL_DIR}/ -> ${REMOTE_HOST}:${REMOTE_DIR}/"
rsync "${RSYNC_ARGS[@]}" "${EXCLUDES[@]}" "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Ensuring cluster scripts are executable"
ssh "${REMOTE_HOST}" "chmod +x '${REMOTE_DIR}/scripts/container_entry.sh' '${REMOTE_DIR}/scripts/run_job.sh' '${REMOTE_DIR}/scripts/sync_to_rcp.sh'"

echo "Done."
