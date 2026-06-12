# Reviewer-Response Experiment Plan

Use the lean split by default. It is about 10 jobs total and assumes the report
already has seed-0 runs for the main bowls and route arms.

```bash
scripts/sync_to_rcp.sh
DRY_RUN=1 scripts/reviewer_experiments.sh part-a
DRY_RUN=1 scripts/reviewer_experiments.sh part-b
```

## Part A: Routing Safety Net

Recommended owner: you.

```bash
scripts/reviewer_experiments.sh part-a
```

Default size: 4 full-budget jobs.

What it runs:

- DynMuon and Route-align at `eta=0.05`, seed `LEAN_ROUTE_SEED=1`.
- DynMuon and stable-rank Route at `eta=0.2`, seed `LEAN_ROUTE_SEED=1`.

Why only these:

- `eta=0.05` checks the one-sided "helps above the optimum" claim without
  rerunning the whole LR grid.
- `eta=0.2` checks the 10x-LR safety-net claim.
- Existing seed-0 report runs provide the first replicate, so this adds the
  missing paired replicate rather than launching every proxy at every LR.

## Part B: Highest-Risk Controls

Recommended owner: teammate.

```bash
scripts/reviewer_experiments.sh part-b
```

Default size: 6 jobs.

What it runs:

- 2 short final-scale anisotropic-noise runs:
  DynMuon vs stable-rank Route at `noise_lambda=3.0`.
- 2 RelMuon confound runs:
  RelMuon-log1p with weight decay, and RelMuon-log1p with weight decay plus
  spectral-norm shape LR scaling, both at `eta=0.1`.
- 2 short cost-profile runs:
  full RelMuon-log1p vs attention-only RelMuon-log1p, logging optimizer time
  separately from forward/backward time.

Why these:

- They answer the reviewer points most likely to undermine the report's current
  claims.
- They avoid extra proxy and spectrum seeds unless the response needs stronger
  statistical wording.

## Full Matrix, Only If Needed

The original 50-job matrix is still available, but it is opt-in:

```bash
scripts/reviewer_experiments.sh full-a
scripts/reviewer_experiments.sh full-b
```

Use this only if there is enough cluster bandwidth or if the lean results are
ambiguous. It adds seeded proxy arms, random-spectrum seeds, safeguard stress
tests, broader RelMuon LR controls, and a broader routing grid.

## Useful Overrides

```bash
# Different added route replicate.
LEAN_ROUTE_SEED=2 scripts/reviewer_experiments.sh part-a

# Stronger final-scale noise check.
SHORT_STEPS=1526 scripts/reviewer_experiments.sh part-b

# Broader matched-RelMuon sweep without running everything else.
RELMUON_LRS=0.02,0.05,0.1,0.2,0.3 scripts/reviewer_experiments.sh relmuon-confound
```

## Pulling Results

```bash
for group in review_lean_route review_lean_noise review_lean_relmuon review_lean_cost
do
  uv run python experiments/pull_wandb.py --group "$group"
done
```
