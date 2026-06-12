#!/usr/bin/env bash
# ============================================================
# Sweep plan for the post-rewrite evaluation (2026-06-11).
#
# Everything runs on FineWeb at the gpt124m scale (configs/gpt124m.yaml:
# 1526 steps = 400M tokens unless STEPS overrides it) and logs to the
# dynmuon-route-sweeps W&B project so experiments/lr_bowl.py can plot the
# bowls per group.
#
# Phases (run them IN ORDER; phases 2-3 need the best LR from phase 1):
#   scripts/sweeps.sh prep            # size the FineWeb cache for STEPS
#   scripts/sweeps.sh bowls           # LR sweeps: dynmuon / muon / adamw
#   scripts/sweeps.sh route 0.2       # router arms at the best dynmuon LR
#   scripts/sweeps.sh controls 0.2    # spectrum-shape controls at that LR
#   scripts/sweeps.sh seeds 0.2       # seed replicates of dynmuon vs route
#
# Safety:
#   * DRY_RUN=1 scripts/sweeps.sh bowls   prints jobs without submitting.
#   * Run names are unique per arm -> unique checkpoint dirs; RunAI
#     preemption resumes exactly (optimizer schedule state is checkpointed).
#   * The embedding LR is pinned (configs embed_lr) so LR sweeps move ONLY the
#     matrix LR; the adamw sweep moves adam_lr, which is its matrix LR.
#   * DATA BUDGET: gpt124m's batch is 262k tokens/step. The default 500M-token
#     FineWeb cache covers ~1900 steps without repetition. `prep` downloads
#     ceil(STEPS*262k) tokens so no sweep silently trains multiple epochs.
#
# Env overrides: STEPS (default 1526), SWEEP_PROJECT, SLEEP_BETWEEN (default 5).
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="${STEPS:-1526}"
SWEEP_PROJECT="${SWEEP_PROJECT:-dynmuon-route-sweeps}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"

submit() {
    # submit <group> <run_name> <config> [extra train.py flags...]
    local group="$1" name="$2" config="$3"
    shift 3
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY: run_job.sh single --config ${config} --wandb --wandb-project ${SWEEP_PROJECT}" \
             "--wandb-group ${group} --run-name ${name} --train-steps ${STEPS} $*"
        return
    fi
    echo ">>> ${group} / ${name}"
    "${HERE}/run_job.sh" single --config "${config}" --wandb \
        --wandb-project "${SWEEP_PROJECT}" --wandb-group "${group}" \
        --run-name "${name}" --train-steps "${STEPS}" "$@"
    sleep "${SLEEP_BETWEEN}"
}

lr_tag() { echo "$1" | tr '.' 'p'; }   # 0.05 -> 0p05 (job-name safe)

cmd="${1:?usage: $0 prep|bowls|bowls-left|route <lr>|controls <lr>|seeds <lr>|relmuon-bowls|relmuon-attention|relmuon-compare|kaon <lr>|proxies <lr>|final <lr>}"
shift || true

case "${cmd}" in
prep)
    # 262144 tokens/step; round up to the next 100M shard.
    tokens=$(( STEPS * 262144 ))
    millions=$(( (tokens + 99999999) / 100000000 * 100 ))
    echo "ensuring FineWeb cache covers ${STEPS} steps (~${millions}M tokens)"
    "${HERE}/run_job.sh" prep-fineweb "${millions}M"
    ;;

bowls)
    # Phase 1 — LR sweeps (one epoch max; see DATA BUDGET above).
    for lr in 0.02 0.05 0.1 0.2 0.4 0.8; do
        submit bowl_dynmuon "bowl_dynmuon_mlr$(lr_tag ${lr})" configs/dynmuon.yaml \
            --muon-lr "${lr}"
    done
    for lr in 0.02 0.05 0.1 0.2 0.4; do
        submit bowl_muon "bowl_muon_mlr$(lr_tag ${lr})" configs/muon.yaml \
            --muon-lr "${lr}"
    done
    # adamw: adam_lr IS the matrix LR for this arm (embeddings stay on embed_lr).
    for lr in 0.0003 0.0006 0.0012 0.0024; do
        submit bowl_adamw "bowl_adamw_alr$(lr_tag ${lr})" configs/adamw.yaml \
            --adam-lr "${lr}"
    done
    ;;

bowls-left)
    # Phase 1b — close the matrix-optimizer sweeps on the LEFT. The current
    # completed Muon/DynMuon curves have their best sampled point at the 0.02
    # boundary, so these points are required before calling them "bowls" or
    # claiming the optimum is bracketed.
    for lr in 0.001 0.002 0.005 0.01; do
        submit bowl_dynmuon "bowl_dynmuon_mlr$(lr_tag ${lr})" configs/dynmuon.yaml \
            --muon-lr "${lr}"
        submit bowl_muon "bowl_muon_mlr$(lr_tag ${lr})" configs/muon.yaml \
            --muon-lr "${lr}"
    done
    ;;

route)
    best="${1:?usage: $0 route <best_dynmuon_muon_lr>}"
    t="$(lr_tag "${best}")"
    # control: the schedule-only baseline at the same LR (same group for overlay)
    submit route_arms "route_ctrl_dynmuon_${t}" configs/dynmuon.yaml --muon-lr "${best}"
    # the router (zscore lean, beta/lean_max from configs/base.yaml)
    submit route_arms "route_${t}" configs/route.yaml --muon-lr "${best}"
    # beta=0 sanity: must match route_ctrl_dynmuon to step-level noise
    submit route_arms "route_beta0_${t}" configs/route.yaml --muon-lr "${best}" --beta 0
    # stronger lean
    submit route_arms "route_beta0p3_${t}" configs/route.yaml --muon-lr "${best}" --beta 0.3
    # magnitude-decoupled router (pure spectrum-shape routing)
    submit route_arms "route_decoupled_${t}" configs/route_decoupled.yaml --muon-lr "${best}"
    ;;

controls)
    best="${1:?usage: $0 controls <best_dynmuon_muon_lr>}"
    t="$(lr_tag "${best}")"
    # Kaon-style question: at matched update magnitude (polar_fro), does the
    # spectrum SHAPE matter at all? power(p=0) = Muon shape, vs random/inverted.
    # NOTE: must go through matrix_optimizer=dynmuon (configs/dynmuon.yaml);
    # the Track-3 Muon class ignores routing/spectrum/magnitude flags.
    for spec in power random inverted; do
        submit spectrum_controls "ctrl_${spec}_${t}" configs/dynmuon.yaml \
            --muon-lr "${best}" \
            --routing-mode fixed --compute-mode svd --magnitude polar_fro \
            --spectrum "${spec}"
    done
    ;;

seeds)
    best="${1:?usage: $0 seeds <best_dynmuon_muon_lr>}"
    t="$(lr_tag "${best}")"
    for seed in 1 2; do
        submit seed_replicates "seed${seed}_muon_${t}" configs/muon.yaml \
            --muon-lr "${best}" --seed "${seed}"
        submit seed_replicates "seed${seed}_dynmuon_${t}" configs/dynmuon.yaml \
            --muon-lr "${best}" --seed "${seed}"
        submit seed_replicates "seed${seed}_route_${t}" configs/route.yaml \
            --muon-lr "${best}" --seed "${seed}"
    done
    ;;

relmuon-bowls)
    # RelMuon LR bowls on the CURRENT code (the 06-10 RelMuon sweep ran the
    # old-init era and is not comparable with the 06-11 bowls).
    for lr in 0.02 0.1 0.3 0.5; do
        submit bowl_relmuon "bowl_relmuon_log1p_mlr$(lr_tag ${lr})" configs/relmuon_log1p.yaml \
            --muon-lr "${lr}"
    done
    for lr in 0.1 0.5; do
        submit bowl_relmuon "bowl_relmuon_rms_mlr$(lr_tag ${lr})" configs/relmuon_rms.yaml \
            --muon-lr "${lr}"
    done
    ;;

relmuon-attention)
    # Spatial ablation: keep RelMuon-log1p only on attention projection
    # matrices, and optimize MLP/embedding/scalar parameters with AdamW.
    for lr in 0.02 0.1 0.3 0.5; do
        group="relmuon_attention"
        name="relmuon_attention_log1p_mlr$(lr_tag ${lr})"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY: run_job.sh spatial --config configs/relmuon_attention.yaml --wandb" \
                 "--wandb-project ${SWEEP_PROJECT} --wandb-group ${group}" \
                 "--run-name ${name} --train-steps ${STEPS} --muon-lr ${lr}"
        else
            echo ">>> ${group} / ${name}"
            "${HERE}/run_job.sh" spatial --config configs/relmuon_attention.yaml --wandb \
                --wandb-project "${SWEEP_PROJECT}" --wandb-group "${group}" \
                --run-name "${name}" --train-steps "${STEPS}" --muon-lr "${lr}"
            sleep "${SLEEP_BETWEEN}"
        fi
    done
    ;;

relmuon-compare)
    # Matched full-vs-attention-only RelMuon-log1p comparison.
    # This is the clean comparison for the report: same W&B project, same
    # training budget, same LRs, same naming convention.
    for lr in 0.02 0.1 0.3 0.5; do
        submit relmuon_compare "relmuon_full_log1p_mlr$(lr_tag ${lr})" configs/relmuon_log1p.yaml \
            --muon-lr "${lr}"
        group="relmuon_compare"
        name="relmuon_attention_log1p_mlr$(lr_tag ${lr})"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY: run_job.sh spatial --config configs/relmuon_attention.yaml --wandb" \
                 "--wandb-project ${SWEEP_PROJECT} --wandb-group ${group}" \
                 "--run-name ${name} --train-steps ${STEPS} --muon-lr ${lr}"
        else
            echo ">>> ${group} / ${name}"
            "${HERE}/run_job.sh" spatial --config configs/relmuon_attention.yaml --wandb \
                --wandb-project "${SWEEP_PROJECT}" --wandb-group "${group}" \
                --run-name "${name}" --train-steps "${STEPS}" --muon-lr "${lr}"
            sleep "${SLEEP_BETWEEN}"
        fi
    done
    ;;

kaon)
    best="${1:?usage: $0 kaon <muon_lr>}"
    t="$(lr_tag "${best}")"
    submit spectrum_controls "ctrl_kaon_${t}" configs/kaon.yaml --muon-lr "${best}"
    ;;

proxies)
    # Ablation over "unbalancedness" proxies at fixed beta magnitude and LR.
    # stable_rank and alignment arms are covered by the route_fill /
    # route_alignment groups (same beta/LR); this adds the SNR proxies.
    # Orientation: the framework routes noisy (low gamma) -> raw momentum,
    # i.e. p decreases with gamma, hence beta = -0.15 is the principled sign;
    # the +0.15 arm is the orientation sanity check.
    best="${1:?usage: $0 proxies <muon_lr>}"
    t="$(lr_tag "${best}")"
    submit route_proxies "proxy_snr_negbeta_${t}" configs/route.yaml \
        --muon-lr "${best}" --modulate-metric snr --beta -0.15
    submit route_proxies "proxy_snr_posbeta_${t}" configs/route.yaml \
        --muon-lr "${best}" --modulate-metric snr --beta 0.15
    submit route_proxies "proxy_snr_ema_negbeta_${t}" configs/route.yaml \
        --muon-lr "${best}" --modulate-metric snr_ema --beta -0.15
    ;;

final)
    # Everything still needed for the report, in one shot (~24 jobs).
    # Usage: scripts/sweeps.sh final 0.02
    best="${1:?usage: $0 final <best_muon_lr>}"
    "$0" bowls-left
    "$0" seeds "${best}"
    "$0" controls "${best}"
    "$0" kaon "${best}"
    "$0" relmuon-bowls
    "$0" relmuon-attention
    "$0" proxies "${best}"
    ;;

*)
    echo "unknown phase: ${cmd}" >&2
    exit 1
    ;;
esac
