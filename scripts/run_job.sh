#!/bin/bash
# ============================================================
# Submit SpecMuon experiments to the RunAI cluster.
#
# Usage: ./scripts/run_job.sh <subcommand> [args]
#
# Subcommands:
#   single           one main.py run (uses CONFIG or per-flag env vars).
#   sanity           short smoke test (STEPS=50) — confirm the container
#                    boots and the project imports before launching a real job.
#   sweep            grid sweep via main.py --sweep. Pass --sweep clauses
#                    through the SWEEP env var (space-separated).
#   compare-best-lr  per-optimizer best-lr comparison through main.py.
#   sav-isolation    scripts/sav_isolation.py on EXPERIMENT.
#   gate-sensitivity scripts/gate_sensitivity.py on EXPERIMENT.
#   shakespeare-long the Priority-1 follow-up: SpecMuon @ top_k=32, lr=3e-4,
#                    10 000 steps on shakespeare. Drives the §4.2 dichotomy.
#   interactive      `runai submit --interactive --attach` for shell access.
#   logs <name>      tail logs of a submitted job.
#   delete <name>    delete a job from the namespace.
#   list             list jobs in the namespace.
#
# Every run parameter can be overridden via env vars (see "Defaults" below).
# A YAML config file under `configs/` can also be passed via CONFIG=… and
# is forwarded to main.py's `--config` flag, which then becomes the base
# for argparse defaults.
#
# Example:
#   CONFIG=configs/shakespeare.yaml OPTIMIZER=specmuon TOP_K=32 STEPS=10000 \
#     ./scripts/run_job.sh single
# ============================================================

set -euo pipefail

USERNAME=$(whoami)

# --- Image -------------------------------------------------------------
# Generic CUDA-enabled PyTorch image. The container must have `uv` on PATH
# and the project synced (`uv sync`) before training launches. Easiest path:
# bake a small derived image that runs `pip install uv && uv sync` once.
IMAGE="${IMAGE:-pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime}"

# --- Project layout (on the home PVC, same path inside the container) -
PROJECT_DIR="${PROJECT_DIR:-${HOME}/MDS/MLO/project}"
MAIN_PY="${PROJECT_DIR}/main.py"
SAV_ISO_PY="${PROJECT_DIR}/scripts/sav_isolation.py"
GATE_SENS_PY="${PROJECT_DIR}/scripts/gate_sensitivity.py"

# Entry shim that pip-installs wandb/pyyaml/matplotlib into ~/.local on the
# PVC, sets PYTHONPATH=${PROJECT_DIR}, then execs `python "$@"`. We use this
# instead of `uv` because the pytorch base image doesn't ship uv.
ENTRY_SH="${PROJECT_DIR}/scripts/container_entry.sh"

# --- RunAI base flags --------------------------------------------------
# WANDB_API_KEY is forwarded from the submitting shell so wandb.init works
# inside the container without baking secrets into the image.
BASE_FLAGS="--image ${IMAGE} --pvc home:${HOME} -e HOME=${HOME} --run-as-user --gpu 1"
if [ -n "${WANDB_API_KEY:-}" ]; then
    BASE_FLAGS="${BASE_FLAGS} -e WANDB_API_KEY=${WANDB_API_KEY}"
fi

# --- Per-invocation timestamp for unique RunAI job names ----------------
# RunAI rejects a `runai submit --name X` if a job with that name still
# exists in the namespace (even completed/failed ones). Stamp every job
# in this invocation with a shared suffix to guarantee no collisions.
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d-%H%M%S)}"

# --- Defaults (override via env vars) ----------------------------------
# Experiment / training
CONFIG="${CONFIG:-}"                          # path to a YAML config; optional
EXPERIMENT="${EXPERIMENT:-shakespeare}"        # linear_regression | matrix_factorization | shakespeare
OPTIMIZER="${OPTIMIZER:-specmuon}"             # adam | adamw | sgd | muon | specmuon
STEPS="${STEPS:-2000}"
LR="${LR:-3e-4}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-50}"

# SpecMuon v2 knobs
TOP_K="${TOP_K:-}"                            # leave empty to use paper default (6)
SIGMA_MODE="${SIGMA_MODE:-}"                  # baseline | sqrt | power | clip | truncate | energy
GATE_THRESHOLD="${GATE_THRESHOLD:-}"          # 0 disables (paper default)
GATE_WINDOW="${GATE_WINDOW:-}"
KAPPA="${KAPPA:-}"

# Logging
BACKEND="${BACKEND:-wandb}"                   # null | matplotlib | wandb
WANDB_PROJECT="${WANDB_PROJECT:-mlo-specmuon}"
WANDB_ENTITY="${WANDB_ENTITY:-cs-439-project}"
LOG_SAV_R="${LOG_SAV_R:-0}"                   # 1 ⇒ --log-sav-r

# Optional run tag — appears in the WandB run name AND the RunAI job name
# so back-to-back jobs with otherwise identical configs don't visually merge.
RUN_TAG="${RUN_TAG:-}"
if [ -n "${RUN_TAG}" ]; then TAG_SUFFIX="-${RUN_TAG}"; else TAG_SUFFIX=""; fi

# --- Helpers -----------------------------------------------------------
# Append `--flag value` to MAIN_ARGS (a bash array) iff value is non-empty.
_append_if_set() {
    local flag="$1" value="$2"
    if [ -n "${value}" ]; then MAIN_ARGS+=("${flag}" "${value}"); fi
}

# Build the common arg list for `main.py` as a bash ARRAY (MAIN_ARGS).
# Arrays preserve token boundaries through `"$@"` — string concatenation
# would re-collapse into a single token under RunAI's argv flattening.
_build_main_args() {
    MAIN_ARGS=()
    if [ -n "${CONFIG}" ]; then
        MAIN_ARGS+=(--config "${CONFIG}")
    else
        MAIN_ARGS+=(--experiment "${EXPERIMENT}")
    fi
    MAIN_ARGS+=(--optimizer "${OPTIMIZER}" --lr "${LR}" --steps "${STEPS}"
                --seed "${SEED}" --log-every "${LOG_EVERY}"
                --backend "${BACKEND}"
                --wandb-project "${WANDB_PROJECT}" --wandb-entity "${WANDB_ENTITY}")
    _append_if_set "--top-k" "${TOP_K}"
    _append_if_set "--sigma-mode" "${SIGMA_MODE}"
    _append_if_set "--gate-threshold" "${GATE_THRESHOLD}"
    _append_if_set "--gate-window" "${GATE_WINDOW}"
    _append_if_set "--kappa" "${KAPPA}"
    if [ "${LOG_SAV_R}" = "1" ]; then MAIN_ARGS+=(--log-sav-r); fi
}

# `runai submit --command --` flattens argv into a single string that
# Kubernetes then re-tokenizes by whitespace — so `bash -lc "cd X && cmd"`
# does NOT survive (quotes vanish, `&&` becomes a literal positional arg).
# Workaround: hand RunAI a shell-free, absolute-path argv list:
#   [bash, /abs/path/container_entry.sh, /abs/path/main.py, --flag, value, …]
# The entry script sets PYTHONPATH + installs missing deps, then execs python.
_submit() {
    local job_name="$1"
    shift
    echo "Submitting: ${job_name}"
    printf '  command: bash %s' "${ENTRY_SH}"
    printf ' %q' "$@"; printf '\n'
    runai submit ${BASE_FLAGS} \
        --name "${job_name}" \
        --command -- bash "${ENTRY_SH}" "$@"
}

# --- Subcommands -------------------------------------------------------
case "${1:-}" in

    single)
        _build_main_args
        job_name="single-${EXPERIMENT}-${OPTIMIZER}${TAG_SUFFIX}-${JOB_STAMP}"
        _submit "${job_name}" "${MAIN_PY}" "${MAIN_ARGS[@]}"
        ;;

    sanity)
        # 50-step smoke test on whatever config is selected. Confirms the
        # container, paths, and WandB credentials before a real job.
        STEPS=50
        LOG_EVERY=25
        BACKEND="${BACKEND:-null}"   # silent by default; pass BACKEND=wandb to confirm logging
        _build_main_args
        job_name="sanity-${EXPERIMENT}-${OPTIMIZER}${TAG_SUFFIX}-${JOB_STAMP}"
        _submit "${job_name}" "${MAIN_PY}" "${MAIN_ARGS[@]}"
        ;;

    sweep)
        # Pass --sweep clauses through SWEEP env var, space-separated:
        #   SWEEP="top_k=2,6,32 sigma_mode=baseline,clip" ./run_job.sh sweep
        : "${SWEEP:?Set SWEEP='param1=v1,v2 param2=v1,v2 ...'}"
        _build_main_args
        for clause in ${SWEEP}; do
            MAIN_ARGS+=(--sweep "${clause}")
        done
        # Sweep can be slow; let WandB see the descriptive run names.
        BACKEND=wandb
        job_name="sweep-${EXPERIMENT}-${OPTIMIZER}${TAG_SUFFIX}-${JOB_STAMP}"
        _submit "${job_name}" "${MAIN_PY}" "${MAIN_ARGS[@]}"
        ;;

    compare-best-lr)
        # Per-optimizer lr sweep, then re-run each at its winner.
        LR_MIN="${LR_MIN:-1e-5}"
        LR_MAX="${LR_MAX:-1e-2}"
        LR_N="${LR_N:-8}"
        _build_main_args
        MAIN_ARGS+=(--compare-best-lr --lr-min "${LR_MIN}" --lr-max "${LR_MAX}" --lr-n "${LR_N}")
        job_name="cbestlr-${EXPERIMENT}${TAG_SUFFIX}-${JOB_STAMP}"
        _submit "${job_name}" "${MAIN_PY}" "${MAIN_ARGS[@]}"
        ;;

    sav-isolation)
        # scripts/sav_isolation.py — top_k ∈ {0, 1, 6, 32} + gated variant.
        # Writes results/sav_isolation/<experiment>/{trajectories.png, summary.md, results.json}.
        OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/sav_isolation/${EXPERIMENT}}"
        job_name="savisol-${EXPERIMENT}${TAG_SUFFIX}-${JOB_STAMP}"
        _submit "${job_name}" "${SAV_ISO_PY}" --experiment "${EXPERIMENT}" --lr "${LR}" --steps "${STEPS}" --seed "${SEED}" --out-dir "${OUT_DIR}"
        ;;

    gate-sensitivity)
        OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/gate_sensitivity/${EXPERIMENT}}"
        GATE_ARGS=(--experiment "${EXPERIMENT}" --lr "${LR}" --steps "${STEPS}" --seed "${SEED}" --out-dir "${OUT_DIR}")
        if [ -n "${TOP_K}" ]; then GATE_ARGS+=(--top-k "${TOP_K}"); fi
        job_name="gatesens-${EXPERIMENT}${TAG_SUFFIX}-${JOB_STAMP}"
        _submit "${job_name}" "${GATE_SENS_PY}" "${GATE_ARGS[@]}"
        ;;

    shakespeare-long)
        # Priority 1(a) of the experimental program: does top_k=32 keep
        # descending past 2000 steps? Push 5× further with --log-sav-r so
        # the WandB curves answer the warmup-vs-sustained-mechanism question.
        CONFIG="${CONFIG:-${PROJECT_DIR}/configs/shakespeare.yaml}"
        EXPERIMENT=shakespeare
        OPTIMIZER=specmuon
        TOP_K="${TOP_K:-32}"
        LR="${LR:-3e-4}"
        STEPS="${STEPS:-10000}"
        LOG_EVERY="${LOG_EVERY:-100}"
        LOG_SAV_R=1
        BACKEND=wandb
        _build_main_args
        job_name="shakelong-top${TOP_K}${TAG_SUFFIX}-${JOB_STAMP}"
        _submit "${job_name}" "${MAIN_PY}" "${MAIN_ARGS[@]}"
        ;;

    interactive)
        runai submit ${BASE_FLAGS} --interactive --attach \
            --name "interactive-${USERNAME}-${JOB_STAMP}"
        ;;

    logs)
        runai logs "${2:?Usage: $0 logs <job-name>}"
        ;;

    delete)
        runai delete job "${2:?Usage: $0 delete <job-name>}"
        ;;

    list)
        runai list
        ;;

    *)
        cat <<EOF
Usage: $0 <subcommand>

  single | sanity | sweep | compare-best-lr
  sav-isolation | gate-sensitivity | shakespeare-long
  interactive | logs <job> | delete <job> | list

See header comment for env vars and examples.
EOF
        exit 1
        ;;
esac
