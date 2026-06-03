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
# Auto-resolve PROJECT_DIR from THIS script's location (run_job.sh lives in
# ${PROJECT_DIR}/scripts/), so we don't hard-code a path that may differ
# from the actual repo location on the cluster PVC. The submitter's CWD is
# irrelevant; what matters is where the repo lives on disk.
# Override via the env var if you ever need to point at a different copy.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${_SELF_DIR}/.." && pwd)}"
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
# Persistent local logs on the PVC (otherwise wandb falls back to /tmp and
# its run cache is wiped when the pod terminates). The directory is created
# lazily by wandb on first use.
WANDB_DIR="${WANDB_DIR:-${HOME}/.wandb}"
BASE_FLAGS="${BASE_FLAGS} -e WANDB_DIR=${WANDB_DIR}"

# --- Per-invocation timestamp for unique RunAI job names ----------------
# RunAI rejects a `runai submit --name X` if a job with that name still
# exists in the namespace (even completed/failed ones). Stamp every job
# in this invocation with a shared suffix to guarantee no collisions.
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d-%H%M%S)}"

# --- Capture which vars the user explicitly set in the environment ----
# Used by subcommands to decide whether to apply their own defaults vs.
# defer to the user's override. `${X+set}` expands to "set" iff X is set
# (even to empty), so we lock the answer BEFORE the defaults block below.
for _v in CONFIG EXPERIMENT OPTIMIZER STEPS LR SEED LOG_EVERY \
          TOP_K SIGMA_MODE GATE_THRESHOLD GATE_WINDOW KAPPA \
          BACKEND LOG_SAV_R CHECKPOINT_DIR CHECKPOINT_EVERY \
          SPECMUON_TARGET COMPARE_OPTIMIZERS LR_MIN LR_MAX LR_N; do
    eval "_USER_${_v}=\"\${${_v}+set}\""
done

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

# Checkpointing (on the PVC so checkpoints survive the pod). Off by default;
# set CHECKPOINT_DIR to opt in, or use one of the subcommands that pins it.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"           # e.g. ${HOME}/optml_checkpoints
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-}"       # intermediate save cadence; empty ⇒ end-only

# Selective-SAV ablation: which 2-D weights receive the SAV branch.
SPECMUON_TARGET="${SPECMUON_TARGET:-}"         # all | mlp | attention; empty ⇒ CLI default ('all')

# Comma-separated optimizer subset for --compare-all / --compare-best-lr.
COMPARE_OPTIMIZERS="${COMPARE_OPTIMIZERS:-}"   # e.g. adamw,muon,specmuon

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
    _append_if_set "--checkpoint-dir" "${CHECKPOINT_DIR}"
    _append_if_set "--checkpoint-every" "${CHECKPOINT_EVERY}"
    _append_if_set "--specmuon-target" "${SPECMUON_TARGET}"
    _append_if_set "--compare-optimizers" "${COMPARE_OPTIMIZERS}"
    # Forward RUN_TAG into the WandB run name too (it already appears in
    # the RunAI job name via TAG_SUFFIX). Without this, back-to-back jobs
    # with identical configs overlap in the WandB dashboard.
    _append_if_set "--run-tag" "${RUN_TAG}"
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

    selective-sav)
        # Layer-selectivity ablation: TWO parallel jobs that vary which 2-D
        # weights receive the SAV branch (the rest get SpecMuon top_k=0,
        # i.e. paper-tail) — isolates SAV's per-layer-type contribution while
        # holding everything else constant. Pair the resulting WandB curves
        # with the existing shakespeare-long run (specmuon_target=all).
        [ -z "${_USER_CONFIG}"           ] && CONFIG="${PROJECT_DIR}/configs/shakespeare.yaml"
        EXPERIMENT=shakespeare
        OPTIMIZER=specmuon
        [ -z "${_USER_TOP_K}"            ] && TOP_K=32
        [ -z "${_USER_LR}"               ] && LR=3e-4
        [ -z "${_USER_STEPS}"            ] && STEPS=10000
        [ -z "${_USER_LOG_EVERY}"        ] && LOG_EVERY=100
        LOG_SAV_R=1
        BACKEND=wandb
        [ -z "${_USER_CHECKPOINT_EVERY}" ] && CHECKPOINT_EVERY=2000
        # Fire MLP-only first, then attention-only — share JOB_STAMP so they
        # sort together in `runai list`. The original SPECMUON_TARGET env
        # var (if any) is ignored; both targets are fixed by this subcommand.
        for tgt in mlp attention; do
            SPECMUON_TARGET="${tgt}"
            CHECKPOINT_DIR="${HOME}/optml_checkpoints/selective_sav_${tgt}"
            _build_main_args
            job_name="selsav-${tgt}${TAG_SUFFIX}-${JOB_STAMP}"
            _submit "${job_name}" "${MAIN_PY}" "${MAIN_ARGS[@]}"
        done
        ;;

    shakespeare-long)
        # Priority 1(a) of the experimental program: does top_k=32 keep
        # descending past 2000 steps? Push 5× further with --log-sav-r so
        # the WandB curves answer the warmup-vs-sustained-mechanism question.
        # Apply subcommand defaults ONLY when the user didn't set the var
        # explicitly — the top-level defaults above don't count.
        [ -z "${_USER_CONFIG}"           ] && CONFIG="${PROJECT_DIR}/configs/shakespeare.yaml"
        EXPERIMENT=shakespeare
        OPTIMIZER=specmuon
        [ -z "${_USER_TOP_K}"            ] && TOP_K=32
        [ -z "${_USER_LR}"               ] && LR=3e-4
        [ -z "${_USER_STEPS}"            ] && STEPS=10000
        [ -z "${_USER_LOG_EVERY}"        ] && LOG_EVERY=100
        LOG_SAV_R=1
        BACKEND=wandb
        [ -z "${_USER_CHECKPOINT_DIR}"   ] && CHECKPOINT_DIR="${HOME}/optml_checkpoints/shakespeare_long"
        [ -z "${_USER_CHECKPOINT_EVERY}" ] && CHECKPOINT_EVERY=2000
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
  sav-isolation | gate-sensitivity | shakespeare-long | selective-sav
  interactive | logs <job> | delete <job> | list

See header comment for env vars and examples.
EOF
        exit 1
        ;;
esac
