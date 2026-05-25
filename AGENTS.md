# AGENTS.md

Guidance for coding agents working in this repository.

## Project Context

This repo benchmarks optimizer convergence on small PyTorch problems. The active experiments are:

- `linear_regression`
- `matrix_factorization`
- `shakespeare`

The optimizer implementations live in `src/optimizers.py`; experiment definitions live in `src/experiments.py`; the training loop is in `src/training.py`; CLI orchestration is in `src/cli.py`.

Important result summaries:

- `shakespeare_results.md`
- `other_experiment_results.md`

Full search CSVs from the latest 1000-step searches are tracked under `results/`:

- `results/linear_regression_1000_search_results.csv`
- `results/matrix_factorization_1000_search_results.csv`
- `results/shakespeare_1000_compare_results.csv`

If rerunning searches, write durable CSVs under `results/` rather than `/private/tmp`.

## Environment

Use `uv` from the repository root.

```bash
uv sync
uv run python main.py --help
```

Optional Weights & Biases config can be set in `.env`:

```bash
WANDB_PROJECT=optml-bench
```

The project depends on PyTorch and `metalcore`. On Apple Silicon/MPS, SVD-heavy optimizers may fall back to CPU for `torch.linalg.svd`; this is expected and can make RelMuon/SpecMuon searches slow.

## Common Commands

Single run:

```bash
uv run python main.py --experiment linear_regression --optimizer muon --steps 300
```

Compare all optimizers:

```bash
uv run python main.py --experiment matrix_factorization --compare-all --lr 1e-2 --log-scale
```

Sweep one optimizer:

```bash
uv run python main.py --experiment shakespeare --optimizer relmuon --sweep lr=3e-2,5e-2,8e-2 --scheduler cosine
```

Run the local search helper:

```bash
uv run python tools/search_other_experiments.py --experiment linear_regression --steps 1000 --csv results/linear_regression_1000_search_results.csv
uv run python tools/search_other_experiments.py --experiment matrix_factorization --steps 1000 --focused --csv results/matrix_factorization_1000_search_results.csv
```

## Implementation Notes

- Preserve the distinction between plain `Muon`, `RelativeMuon`/`relmuon`, and `SpecMuon`.
- Muon-family optimizers should use repo-controlled `adjust_lr_fn="shape_scaling"` when comparing against the documented tuned runs.
- `RelativeMuon(spectrum_transform="log1p")` uses normalized `log(1 + singular_value)` scales directly. Do not tune or report `alpha`, `min_scale`, or `max_scale` for this log1p variant.
- For Muon-family optimizers, `src/training.py` splits matrix parameters from embeddings/LayerNorm/non-matrix parameters. Matrix parameters use the Muon-family optimizer; remaining parameters use AdamW.
- `SpecMuon.step()` requires the current loss and is called as `optimizer.step(loss=loss)` in the training loop.
- Keep default experiment sizes stable unless the user explicitly asks for a different benchmark.
- For reproducible comparisons, reset both `torch.manual_seed(0)` and `np.random.seed(0)` before each run.

## Result Interpretation

- Linear regression reaches a synthetic noise floor after tuning, so differences around `1e-10` in final loss are not meaningful.
- Matrix factorization is more discriminative: tuned RelMuon reaches the numerical floor, while plain Muon can have a much worse final loss even when its best-seen loss is lower mid-run.
- Shakespeare results are more sensitive to model size, batch size, scheduler, and context length. Match the settings in `shakespeare_results.md` before comparing new claims.
- For every reported experiment, include the hyperparameters used in the markdown/HTML prose and tables: LR, scheduler, momentum/betas when applicable, weight decay, step count, batch/context/model settings for Shakespeare, Muon-family `adjust_lr_fn`, and optimizer-specific settings such as `alpha`, scale clamps, or `spectrum_transform`.

## Coding Conventions

- Keep changes scoped. Avoid unrelated refactors while running experiments or tuning optimizers.
- Prefer adding small helper scripts under `tools/` for repeatable searches rather than relying on one-off shell history.
- Write scratch outputs to `/private/tmp`, but put result artifacts referenced by markdown under `results/`.
- Update the relevant markdown result file when adding new benchmark conclusions.
- Do not commit or rely on `wandb/` run artifacts for core results; summarize the relevant metrics in markdown or CSV.
- For notebooks and exploratory analysis plots, use matplotlib unless the user explicitly asks for another plotting stack.
- When the user asks for a git commit, prefer a complete multi-line commit message via `git commit -F <message-file>` or an editor flow instead of a single `git commit -m` subject, unless the user explicitly asks for a simple one-line commit.

## HTML Reports

When creating HTML reports for experiment results:

- Before creating a result report, ask whether the user wants markdown, HTML, or both unless they already specified the format.
- Before running new report-oriented experiments or extending an existing report, first read the relevant existing markdown and HTML reports to understand what is already covered. Then ask what additional information, comparisons, diagnostics, or emphasis the user wants added, unless they have already specified it clearly.
- Prefer generating the HTML from durable CSVs or markdown summaries with a small script under `tools/`, rather than hand-editing large result tables.
- Keep reports static and self-contained when practical: inline CSS and SVG are fine; avoid external CDNs unless the user explicitly wants them.
- Store the final report at the repository root or another obvious durable path, and keep source data under `results/`.
- Use clear visual comparisons: log-scale charts for losses, gradient norms, parameter norms, or singular-value summaries when values span many orders of magnitude.
- For norm and singular-value dynamics, prefer checkpoint trajectory line graphs over final-only bar charts when checkpoint data is available.
- When adding a new optimizer or optimizer variant to a dynamics report, include its norm and singular-value diagnostics alongside loss metrics. Do not report only final-loss tables for optimizer dynamics comparisons.
- Always show the hyperparameters used for each experiment in HTML and markdown reports, not only the winning loss. Put concise hyperparameter columns in tables and mention the most important settings in the surrounding review text.
- Include the key conclusion in prose above the charts, then tables below for exact numbers.
- If the report is generated, preserve or add the generator script so future agents can rebuild it after new runs.
- Verify generated HTML in the in-app browser when possible. Direct `file://` may work for the user, but browser automation can block it; if useful, serve the repo temporarily with `uv run python -m http.server <port>` and open `http://127.0.0.1:<port>/<report>.html`.
- Stop any temporary local server after verification.

## Verification

There is no broad test suite documented for this repo. For optimizer changes, run at least one short smoke test:

```bash
uv run python main.py --experiment linear_regression --optimizer adamw --steps 5
uv run python main.py --experiment linear_regression --optimizer relmuon --steps 5 --adjust-lr-fn shape_scaling
```

For benchmark changes, prefer a fixed-seed run with `--scheduler cosine` and include final loss, best-seen loss, LR, scheduler, step count, and key optimizer settings in the result note.
