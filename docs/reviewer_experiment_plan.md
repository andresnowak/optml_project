# Reviewer-Response Experiment Plan

This plan is encoded in `scripts/reviewer_experiments.sh`. The defaults extend
the existing report runs without making the response unnecessarily expensive:
seed 0 is assumed to exist for the main report arms, so the launcher submits
seeds `1,2` where extra power is needed. For a clean rerun, set
`SEEDS=0,1,2 PROXY_SEEDS=0,1,2 SPECTRUM_SEEDS=0,1,2`.

## Before Submitting

```bash
scripts/sync_to_rcp.sh
DRY_RUN=1 scripts/reviewer_experiments.sh part-a
DRY_RUN=1 scripts/reviewer_experiments.sh part-b
```

If the FineWeb cache is too small for the chosen budget:

```bash
scripts/reviewer_experiments.sh prep
```

## Part A: Routing Evidence

Recommended owner: one person runs this whole part.

```bash
scripts/reviewer_experiments.sh part-a
```

Default size: 28 jobs.

This submits:

- `review_route_grid`: DynMuon, alignment routing, and stable-rank routing at
  `eta in {0.01, 0.02, 0.05, 0.2}` for seeds `1,2`.
- `review_router_safeguards`: short stress tests comparing z-scored/capped
  routing against raw and uncapped variants.

Reviewer gaps covered:

- Thin seeding on the route-align point.
- The safety-net claim across the LR grid, including the one-sidedness at
  `0.01`, `0.05`, and the mistuned `0.2` point.
- Evidence for why z-scoring and a lean cap are safeguards, not cosmetic
  implementation choices.

Useful overrides:

```bash
# Fresh rerun instead of extending existing seed-0 logs.
SEEDS=0,1,2 scripts/reviewer_experiments.sh part-a

# Cheaper routing-only run.
ROUTE_VARIANTS=dynmuon,route_align scripts/reviewer_experiments.sh routing-grid
```

## Part B: Controls, Design Case, and Cost

Recommended owner: the teammate runs this whole part.

```bash
scripts/reviewer_experiments.sh part-b
```

Default size: 22 jobs.

This submits:

- `review_proxy_seeds`: extra seeds for SNR(+), SNR(-), and EMA-SNR(-) at
  `eta=0.02`.
- `review_spectrum_seeds`: extra seeds for the random spectrum control at
  `eta=0.02`.
- `review_noise_finalscale`: 124M-scale anisotropic-noise design-case runs for
  DynMuon vs stable-rank routing, using `SHORT_STEPS=760`.
- `review_relmuon_confound`: RelMuon-log1p controls with weight decay and with
  weight decay plus spectral-norm shape LR scaling.
- `review_cost_profile`: short timing runs for full RelMuon, attention-only
  RelMuon, Muon, and AdamW. These log optimizer-step time separately from
  forward/backward time.

Reviewer gaps covered:

- Thin seeding on proxy and random-spectrum claims.
- The router's originally designed-for anisotropic-noise case at final model
  scale.
- The RelMuon weight-decay / shape-LR confound.
- The attention-only cost oddity.

Useful overrides:

```bash
# Run full spectrum-control seed coverage.
SPECTRA=power,random,inverted scripts/reviewer_experiments.sh spectrum-seeds

# Stronger noise stress test.
NOISE_LAMBDAS=1.5,3.0 scripts/reviewer_experiments.sh noise-final-scale

# Re-tune matched RelMuon more broadly.
RELMUON_LRS=0.02,0.05,0.1,0.2,0.3 scripts/reviewer_experiments.sh relmuon-confound
```

## Optional Longer-Horizon Check

The single-budget limitation is real. The cheaper default is to keep this
optional because it is the only batch that directly increases wall-clock time.
It runs Muon and DynMuon at `HORIZON_STEPS=2289` (about 600M tokens) for seed 0.

```bash
scripts/reviewer_experiments.sh horizon
```

For a stronger but longer check:

```bash
HORIZON_SEEDS=0,1 HORIZON_STEPS=3052 PREP_STEPS=3052 scripts/reviewer_experiments.sh prep
HORIZON_SEEDS=0,1 HORIZON_STEPS=3052 scripts/reviewer_experiments.sh horizon
```

## Pulling and Analyzing

After jobs finish, pull their W&B groups with the existing puller:

```bash
for group in \
  review_route_grid review_router_safeguards review_proxy_seeds \
  review_spectrum_seeds review_noise_finalscale review_relmuon_confound \
  review_cost_profile review_horizon
do
  uv run python experiments/pull_wandb.py --group "$group"
done
```

Then rerun the report pipeline as usual:

```bash
uv run python run.py --skip-pull
```

