#!/usr/bin/env bash
# ============================================================
# Targeted reviewer-response experiments.
#
# This script turns the review gaps into reproducible W&B job batches. The
# default split is deliberately lean: about ten jobs total, using the report's
# existing seed-0 runs as anchors.
#
#   DRY_RUN=1 scripts/reviewer_experiments.sh part-a
#   DRY_RUN=1 scripts/reviewer_experiments.sh part-b
#
# The full reviewer-proof matrix is still available as full-a/full-b, but that
# is intentionally no longer the default.
#
# Budgets:
#   * Full-budget runs use STEPS=1526 (400M tokens).
#   * Noise tests use SHORT_STEPS=760 by default.
#   * Cost profiles use PROFILE_STEPS=220 and log timing every 10 steps.
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_PROJECT="${SWEEP_PROJECT:-dynmuon-route-sweeps}"
STEPS="${STEPS:-1526}"
SHORT_STEPS="${SHORT_STEPS:-760}"
PROFILE_STEPS="${PROFILE_STEPS:-220}"
HORIZON_STEPS="${HORIZON_STEPS:-2289}"   # 600M tokens at the gpt124m batch size.
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"

# Seed defaults extend the current report's seed-0 runs. Override to 0,1,2 for
# a fully independent rerun.
SEEDS="${SEEDS:-1,2}"
PROXY_SEEDS="${PROXY_SEEDS:-${SEEDS}}"
SPECTRUM_SEEDS="${SPECTRUM_SEEDS:-${SEEDS}}"
NOISE_SEEDS="${NOISE_SEEDS:-0,1}"
SAFEGUARD_SEEDS="${SAFEGUARD_SEEDS:-0}"
RELMUON_SEEDS="${RELMUON_SEEDS:-0}"
PROFILE_SEEDS="${PROFILE_SEEDS:-0}"
HORIZON_SEEDS="${HORIZON_SEEDS:-0}"
LEAN_ROUTE_SEED="${LEAN_ROUTE_SEED:-1}"
LEAN_NOISE_SEED="${LEAN_NOISE_SEED:-0}"
LEAN_RELMUON_SEED="${LEAN_RELMUON_SEED:-0}"
LEAN_PROFILE_SEED="${LEAN_PROFILE_SEED:-0}"

ROUTE_LRS="${ROUTE_LRS:-0.01,0.02,0.05,0.2}"
ROUTE_VARIANTS="${ROUTE_VARIANTS:-dynmuon,route_align,route_stable}"
PROXY_LR="${PROXY_LR:-0.02}"
SPECTRUM_LR="${SPECTRUM_LR:-0.02}"
SPECTRA="${SPECTRA:-random}"
SAFEGUARD_LR="${SAFEGUARD_LR:-0.02}"
NOISE_LRS="${NOISE_LRS:-0.02}"
NOISE_LAMBDAS="${NOISE_LAMBDAS:-3.0}"
RELMUON_LRS="${RELMUON_LRS:-0.02,0.1,0.3}"
PROFILE_LR="${PROFILE_LR:-0.1}"
HORIZON_LR="${HORIZON_LR:-0.02}"

usage() {
    cat <<'EOF'
usage: scripts/reviewer_experiments.sh <phase>

phases:
  prep                 prepare FineWeb cache for STEPS
  part-a               lean routing add-on (4 jobs)
  part-b               lean controls/profile add-on (6 jobs)
  full-a               full routing grid + router-safeguard stress tests
  full-b               full proxy/spectrum/noise/RelMuon/cost matrix
  lean-routing         same as part-a
  lean-controls        same as part-b
  routing-grid         seeded route on/off grid across ROUTE_LRS
  proxy-seeds          seed the SNR / EMA-SNR proxy arms at PROXY_LR
  spectrum-seeds       seed random/power/inverted spectrum controls
  safeguards           z-score / lean-cap stress test
  noise-final-scale    anisotropic-noise design-case test at gpt124m scale
  relmuon-confound     RelMuon wd and wd+shape-LR controls
  cost-profile         short timing runs for full vs attention-only RelMuon
  horizon              optional longer-horizon Muon vs DynMuon smoke test

common env overrides:
  DRY_RUN=1, SWEEP_PROJECT, STEPS, SHORT_STEPS, PROFILE_STEPS,
  LEAN_ROUTE_SEED, LEAN_NOISE_SEED, LEAN_RELMUON_SEED,
  LEAN_PROFILE_SEED, SEEDS, ROUTE_LRS, RELMUON_LRS, NOISE_LAMBDAS
EOF
}

each_csv() {
    local csv="$1"
    local old_ifs="${IFS}"
    IFS=','
    # shellcheck disable=SC2206
    _CSV_ITEMS=(${csv})
    IFS="${old_ifs}"
}

lr_tag() { echo "$1" | tr '.' 'p'; }

submit_single() {
    # submit_single <group> <run_name> <config> <steps> [train.py flags...]
    local group="$1" name="$2" config="$3" steps="$4"
    shift 4
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY: run_job.sh single --config ${config} --wandb --wandb-project ${SWEEP_PROJECT}" \
             "--wandb-group ${group} --run-name ${name} --train-steps ${steps} $*"
        return
    fi
    echo ">>> ${group} / ${name}"
    "${HERE}/run_job.sh" single --config "${config}" --wandb \
        --wandb-project "${SWEEP_PROJECT}" --wandb-group "${group}" \
        --run-name "${name}" --train-steps "${steps}" "$@"
    sleep "${SLEEP_BETWEEN}"
}

submit_spatial() {
    # submit_spatial <group> <run_name> <config> <steps> [train.py flags...]
    local group="$1" name="$2" config="$3" steps="$4"
    shift 4
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY: run_job.sh spatial --config ${config} --wandb --wandb-project ${SWEEP_PROJECT}" \
             "--wandb-group ${group} --run-name ${name} --train-steps ${steps} $*"
        return
    fi
    echo ">>> ${group} / ${name}"
    "${HERE}/run_job.sh" spatial --config "${config}" --wandb \
        --wandb-project "${SWEEP_PROJECT}" --wandb-group "${group}" \
        --run-name "${name}" --train-steps "${steps}" "$@"
    sleep "${SLEEP_BETWEEN}"
}

submit_route_variant() {
    local variant="$1" lr="$2" seed="$3" group="$4"
    local tag
    tag="$(lr_tag "${lr}")"
    case "${variant}" in
        dynmuon)
            submit_single "${group}" "review_route_dynmuon_mlr${tag}_seed${seed}" \
                configs/dynmuon.yaml "${STEPS}" --muon-lr "${lr}" --seed "${seed}"
            ;;
        route_align)
            submit_single "${group}" "review_route_align_mlr${tag}_seed${seed}" \
                configs/route.yaml "${STEPS}" --muon-lr "${lr}" --seed "${seed}" \
                --modulate-metric alignment --beta 0.15
            ;;
        route_stable)
            submit_single "${group}" "review_route_stable_mlr${tag}_seed${seed}" \
                configs/route.yaml "${STEPS}" --muon-lr "${lr}" --seed "${seed}" \
                --modulate-metric stable_rank --beta 0.15
            ;;
        *)
            echo "unknown route variant: ${variant}" >&2
            exit 1
            ;;
    esac
}

phase_prep() {
    local prep_steps="${PREP_STEPS:-${STEPS}}"
    local tokens=$(( prep_steps * 262144 ))
    local millions=$(( (tokens + 99999999) / 100000000 * 100 ))
    echo "ensuring FineWeb cache covers ${prep_steps} steps (~${millions}M tokens)"
    "${HERE}/run_job.sh" prep-fineweb "${millions}M"
}

phase_routing_grid() {
    local group="review_route_grid"
    each_csv "${ROUTE_LRS}"; local lrs=("${_CSV_ITEMS[@]}")
    each_csv "${SEEDS}"; local seeds=("${_CSV_ITEMS[@]}")
    each_csv "${ROUTE_VARIANTS}"; local variants=("${_CSV_ITEMS[@]}")
    for lr in "${lrs[@]}"; do
        for seed in "${seeds[@]}"; do
            for variant in "${variants[@]}"; do
                submit_route_variant "${variant}" "${lr}" "${seed}" "${group}"
            done
        done
    done
}

phase_lean_routing() {
    # Four full-budget jobs. These extend the report's existing seed-0 points
    # without launching the full 4-LR x 3-variant x multi-seed grid.
    #
    # eta=0.05: checks the reported one-sided "helps above optimum" claim for
    # the winning alignment proxy.
    # eta=0.2: checks the safety-net claim at 10x the tuned LR, using the
    # stable-rank router that already has the seed-0 high-LR report point.
    local group="review_lean_route"
    local seed="${LEAN_ROUTE_SEED}"
    submit_route_variant dynmuon 0.05 "${seed}" "${group}"
    submit_route_variant route_align 0.05 "${seed}" "${group}"
    submit_route_variant dynmuon 0.2 "${seed}" "${group}"
    submit_route_variant route_stable 0.2 "${seed}" "${group}"
}

phase_proxy_seeds() {
    local group="review_proxy_seeds"
    local tag
    tag="$(lr_tag "${PROXY_LR}")"
    each_csv "${PROXY_SEEDS}"; local seeds=("${_CSV_ITEMS[@]}")
    for seed in "${seeds[@]}"; do
        submit_single "${group}" "review_proxy_snr_pos_mlr${tag}_seed${seed}" \
            configs/route.yaml "${STEPS}" --muon-lr "${PROXY_LR}" --seed "${seed}" \
            --modulate-metric snr --beta 0.15
        submit_single "${group}" "review_proxy_snr_neg_mlr${tag}_seed${seed}" \
            configs/route.yaml "${STEPS}" --muon-lr "${PROXY_LR}" --seed "${seed}" \
            --modulate-metric snr --beta -0.15
        submit_single "${group}" "review_proxy_snr_ema_neg_mlr${tag}_seed${seed}" \
            configs/route.yaml "${STEPS}" --muon-lr "${PROXY_LR}" --seed "${seed}" \
            --modulate-metric snr_ema --beta -0.15
    done
}

phase_spectrum_seeds() {
    local group="review_spectrum_seeds"
    local tag
    tag="$(lr_tag "${SPECTRUM_LR}")"
    each_csv "${SPECTRUM_SEEDS}"; local seeds=("${_CSV_ITEMS[@]}")
    each_csv "${SPECTRA}"; local spectra=("${_CSV_ITEMS[@]}")
    for seed in "${seeds[@]}"; do
        for spectrum in "${spectra[@]}"; do
            submit_single "${group}" "review_spectrum_${spectrum}_mlr${tag}_seed${seed}" \
                configs/dynmuon.yaml "${STEPS}" --muon-lr "${SPECTRUM_LR}" --seed "${seed}" \
                --routing-mode fixed --compute-mode svd --magnitude polar_fro \
                --spectrum "${spectrum}"
        done
    done
}

phase_safeguards() {
    local group="review_router_safeguards"
    local tag
    tag="$(lr_tag "${SAFEGUARD_LR}")"
    each_csv "${SAFEGUARD_SEEDS}"; local seeds=("${_CSV_ITEMS[@]}")
    for seed in "${seeds[@]}"; do
        submit_single "${group}" "review_safeguard_zscore_cap_mlr${tag}_seed${seed}" \
            configs/route.yaml "${SHORT_STEPS}" --muon-lr "${SAFEGUARD_LR}" --seed "${seed}" \
            --modulate-metric stable_rank --beta 0.15
        submit_single "${group}" "review_safeguard_raw_cap_mlr${tag}_seed${seed}" \
            configs/route.yaml "${SHORT_STEPS}" --muon-lr "${SAFEGUARD_LR}" --seed "${seed}" \
            --modulate-metric stable_rank --beta 0.15 --lean-norm raw
        submit_single "${group}" "review_safeguard_zscore_uncapped_mlr${tag}_seed${seed}" \
            configs/route_zscore_uncapped.yaml "${SHORT_STEPS}" --muon-lr "${SAFEGUARD_LR}" --seed "${seed}" \
            --modulate-metric stable_rank --beta 0.15
        submit_single "${group}" "review_safeguard_raw_uncapped_mlr${tag}_seed${seed}" \
            configs/route_raw_uncapped.yaml "${SHORT_STEPS}" --muon-lr "${SAFEGUARD_LR}" --seed "${seed}" \
            --modulate-metric stable_rank --beta 0.15
    done
}

phase_noise_final_scale() {
    local group="review_noise_finalscale"
    each_csv "${NOISE_LRS}"; local lrs=("${_CSV_ITEMS[@]}")
    each_csv "${NOISE_LAMBDAS}"; local lambdas=("${_CSV_ITEMS[@]}")
    each_csv "${NOISE_SEEDS}"; local seeds=("${_CSV_ITEMS[@]}")
    for lr in "${lrs[@]}"; do
        local tag
        tag="$(lr_tag "${lr}")"
        for lambda in "${lambdas[@]}"; do
            local ltag
            ltag="$(lr_tag "${lambda}")"
            for seed in "${seeds[@]}"; do
                submit_single "${group}" "review_noise_dynmuon_mlr${tag}_lam${ltag}_seed${seed}" \
                    configs/dynmuon.yaml "${SHORT_STEPS}" --muon-lr "${lr}" --seed "${seed}" \
                    --noise-lambda "${lambda}"
                submit_single "${group}" "review_noise_route_stable_mlr${tag}_lam${ltag}_seed${seed}" \
                    configs/route.yaml "${SHORT_STEPS}" --muon-lr "${lr}" --seed "${seed}" \
                    --modulate-metric stable_rank --beta 0.15 --noise-lambda "${lambda}"
            done
        done
    done
}

phase_relmuon_confound() {
    local group="review_relmuon_confound"
    each_csv "${RELMUON_LRS}"; local lrs=("${_CSV_ITEMS[@]}")
    each_csv "${RELMUON_SEEDS}"; local seeds=("${_CSV_ITEMS[@]}")
    for lr in "${lrs[@]}"; do
        local tag
        tag="$(lr_tag "${lr}")"
        for seed in "${seeds[@]}"; do
            submit_single "${group}" "review_relmuon_log1p_wd_mlr${tag}_seed${seed}" \
                configs/relmuon_log1p_wd.yaml "${STEPS}" --muon-lr "${lr}" --seed "${seed}"
            submit_single "${group}" "review_relmuon_log1p_matched_mlr${tag}_seed${seed}" \
                configs/relmuon_log1p_matched.yaml "${STEPS}" --muon-lr "${lr}" --seed "${seed}"
        done
    done
}

phase_cost_profile() {
    local group="review_cost_profile"
    local tag
    tag="$(lr_tag "${PROFILE_LR}")"
    each_csv "${PROFILE_SEEDS}"; local seeds=("${_CSV_ITEMS[@]}")
    for seed in "${seeds[@]}"; do
        submit_single "${group}" "review_profile_relmuon_full_mlr${tag}_seed${seed}" \
            configs/relmuon_log1p.yaml "${PROFILE_STEPS}" --muon-lr "${PROFILE_LR}" --seed "${seed}" \
            --log-every 10 --val-loss-every 110
        submit_spatial "${group}" "review_profile_relmuon_attention_mlr${tag}_seed${seed}" \
            configs/relmuon_attention.yaml "${PROFILE_STEPS}" --muon-lr "${PROFILE_LR}" --seed "${seed}" \
            --log-every 10 --val-loss-every 110
        submit_single "${group}" "review_profile_muon_mlr0p02_seed${seed}" \
            configs/muon.yaml "${PROFILE_STEPS}" --muon-lr 0.02 --seed "${seed}" \
            --log-every 10 --val-loss-every 110
        submit_single "${group}" "review_profile_adamw_alr0p0012_seed${seed}" \
            configs/adamw.yaml "${PROFILE_STEPS}" --adam-lr 0.0012 --seed "${seed}" \
            --log-every 10 --val-loss-every 110
    done
}

phase_lean_controls() {
    # Six jobs total. These cover the high-risk non-routing objections:
    # designed-for anisotropic noise, RelMuon protocol confound, and the
    # attention-only cost oddity.
    local noise_group="review_lean_noise"
    local rel_group="review_lean_relmuon"
    local cost_group="review_lean_cost"
    local noise_seed="${LEAN_NOISE_SEED}"
    local rel_seed="${LEAN_RELMUON_SEED}"
    local profile_seed="${LEAN_PROFILE_SEED}"
    local profile_tag
    profile_tag="$(lr_tag "${PROFILE_LR}")"

    submit_single "${noise_group}" "review_lean_noise_dynmuon_mlr0p02_lam3p0_seed${noise_seed}" \
        configs/dynmuon.yaml "${SHORT_STEPS}" --muon-lr 0.02 --seed "${noise_seed}" \
        --noise-lambda 3.0
    submit_single "${noise_group}" "review_lean_noise_route_stable_mlr0p02_lam3p0_seed${noise_seed}" \
        configs/route.yaml "${SHORT_STEPS}" --muon-lr 0.02 --seed "${noise_seed}" \
        --modulate-metric stable_rank --beta 0.15 --noise-lambda 3.0

    submit_single "${rel_group}" "review_lean_relmuon_log1p_wd_mlr0p1_seed${rel_seed}" \
        configs/relmuon_log1p_wd.yaml "${STEPS}" --muon-lr 0.1 --seed "${rel_seed}"
    submit_single "${rel_group}" "review_lean_relmuon_log1p_matched_mlr0p1_seed${rel_seed}" \
        configs/relmuon_log1p_matched.yaml "${STEPS}" --muon-lr 0.1 --seed "${rel_seed}"

    submit_single "${cost_group}" "review_lean_profile_relmuon_full_mlr${profile_tag}_seed${profile_seed}" \
        configs/relmuon_log1p.yaml "${PROFILE_STEPS}" --muon-lr "${PROFILE_LR}" --seed "${profile_seed}" \
        --log-every 10 --val-loss-every 110
    submit_spatial "${cost_group}" "review_lean_profile_relmuon_attention_mlr${profile_tag}_seed${profile_seed}" \
        configs/relmuon_attention.yaml "${PROFILE_STEPS}" --muon-lr "${PROFILE_LR}" --seed "${profile_seed}" \
        --log-every 10 --val-loss-every 110
}

phase_horizon() {
    local group="review_horizon"
    local tag
    tag="$(lr_tag "${HORIZON_LR}")"
    each_csv "${HORIZON_SEEDS}"; local seeds=("${_CSV_ITEMS[@]}")
    for seed in "${seeds[@]}"; do
        submit_single "${group}" "review_horizon_muon_mlr${tag}_steps${HORIZON_STEPS}_seed${seed}" \
            configs/muon.yaml "${HORIZON_STEPS}" --muon-lr "${HORIZON_LR}" --seed "${seed}"
        submit_single "${group}" "review_horizon_dynmuon_mlr${tag}_steps${HORIZON_STEPS}_seed${seed}" \
            configs/dynmuon.yaml "${HORIZON_STEPS}" --muon-lr "${HORIZON_LR}" --seed "${seed}"
    done
}

cmd="${1:-}"
case "${cmd}" in
    prep) phase_prep ;;
    routing-grid) phase_routing_grid ;;
    proxy-seeds) phase_proxy_seeds ;;
    spectrum-seeds) phase_spectrum_seeds ;;
    safeguards) phase_safeguards ;;
    noise-final-scale) phase_noise_final_scale ;;
    relmuon-confound) phase_relmuon_confound ;;
    cost-profile) phase_cost_profile ;;
    horizon) phase_horizon ;;
    lean-routing|part-a)
        phase_lean_routing
        ;;
    lean-controls|part-b)
        phase_lean_controls
        ;;
    full-a)
        phase_routing_grid
        phase_safeguards
        ;;
    full-b)
        phase_proxy_seeds
        phase_spectrum_seeds
        phase_noise_final_scale
        phase_relmuon_confound
        phase_cost_profile
        ;;
    ""|-h|--help|help) usage ;;
    *)
        echo "unknown phase: ${cmd}" >&2
        usage >&2
        exit 1
        ;;
esac
