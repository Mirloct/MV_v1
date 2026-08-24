# CONTEXT.md — Project Memory

This file is persistent memory for future sessions/agents working on this
project. Read it before making changes. Keep it updated as work progresses.

## Purpose

This project is a research framework for anomaly detection in banking-sector
panel data. It combines an Isolation Forest model and a Variational
Autoencoder (VAE) as complementary detectors, with Optuna-driven
hyperparameter tuning, crash-recovery / resiliency for long-running jobs,
exhaustive logging of every phase, and rich reporting (HTML/Markdown, no PDF
-- removed by explicit decision, see `docs/decisiones_de_modelado.md`)
of results, metrics, and interpretability artifacts.

## Directory Structure

**Source at the root, every generated file under `artifacts/`.** The split is
the organising principle: the root holds only things a human writes, and the
whole generated state can be inspected — or deleted — in one place.
`python main.py` rebuilds `artifacts/` from scratch.

```
Modelo-v0.1/
├── main.py                 # orchestrator entry point (12 phases, argparse CLI)
├── setup_validator.py      # environment / dependency check
├── requirements.txt
├── README.md               # user-facing guide
├── CONTEXT.md              # this file — project memory
├── .gitignore              # artifacts/ and caches stay out of version control
├── src/
│   ├── data/               # panel loader + synthetic generator
│   ├── preprocessing/      # pipeline (transforms, panel features, ratios) + diagnostics
│   ├── models/             # Isolation Forest + VAE
│   ├── evaluation/         # OOT split, GT join, metrics, scoring, OOT Excel export
│   ├── interpretability/   # SHAP, path length, latent space, per-feature recon
│   ├── reporting/          # HTML/MD report builder (no PDF)
│   └── utils/              # paths.py (all artifact locations) + logging_config.py
├── docs/                   # *.md sources + generated documentation.html
└── artifacts/              # EVERYTHING the pipeline writes (gitignored)
    ├── data/               # data.csv + ground_truth.parquet (hidden labels)
    ├── logs/               # execution.log
    ├── models/             # iforest.joblib, vae_best.pt, vae/, vae_tuning/
    ├── tuning/             # optuna_*.db, best_params_*.yaml (Optuna outputs)
    └── reports/            # anomaly_report.{html,md}, model_documentation.md, oot_p95_*.xlsx, feature_attribution.xlsx
        └── figures/        # ALL figures, flat (see rule below)
```

`artifacts/models/vae_tuning/` grows one subdirectory per Optuna trial for
crash-resume. That is expected and transient — delete it once a study has
finished; nothing downstream reads it.

## Paths — single source of truth

**Never hardcode an artifact path.** `src/utils/paths.py` holds every one of
them and is the only module that knows the `artifacts/` layout; each consumer
takes its default from there (`_DEFAULT_MODEL_OUT = paths.IFOREST_MODEL`, etc.).
Moving the tree again is a one-file change.

Two properties those values must keep: **relative, never absolute** — because
`tests/conftest.py` chdirs the session into a throwaway sandbox precisely so the
defaults resolve there and never touch the real tree — and built with
`os.path.join`, so separators are the standard library's problem.

## HARD RULE — Figures

**All generated plots/figures/charts, from ANY module, MUST be saved under
`artifacts/reports/figures/`** (i.e. `paths.FIGURES_DIR`). Project-wide
convention, no exceptions: not in `artifacts/data/`, not next to scripts, not in
`artifacts/models/`.

The directory is **flat**. It used to carry one subdirectory per producing
module (`preprocessing/`, `iforest/`, `vae/`, `evaluation/`,
`interpretability/`), but every filename already begins with its producer
(`iforest_*`, `vae_*`, `embedding_*`, `roc_pr_*`), so the nesting bought nothing
and cost five constants. All 24 filenames are unique; keep it that way when
adding a figure.

## Tech Stack

- Python 3.12+
- scikit-learn (Isolation Forest)
- torch (VAE)
- numpy, pandas, scipy, statsmodels (data/panel handling, stats)
- optuna (hyperparameter tuning)
- shap, umap-learn (interpretability)
- matplotlib, plotly, seaborn (visualization)
- tqdm, joblib (progress, parallelism/persistence)
- pyyaml (config files)

## Status

- [x] Directory scaffolding created
- [x] `src/utils/logging_config.py` (`setup_logging`, `log_phase`)
- [x] `setup_validator.py` (environment/dependency check + auto-install)
- [x] `requirements.txt`
- [x] `main.py` stub (no pipeline logic yet)
- [x] `src/data` — panel data loader + synthetic generator (`loader.py`,
  `synthetic.py`); see the Data contract and Scale characteristics sections
  below
- [x] `src/preprocessing` — cleaning / scaling / panel-transform pipeline
  (`pipeline.py`, `statistics.py`); see the Preprocessing contract section
  below. Tests (`tests/test_preprocessing.py`) are being added in parallel.
- [x] `src/models` — Isolation Forest implementation (`iforest.py`:
  `IsolationForestDetector`, `tune_iforest`, `plot_score_distribution`); see
  `docs/models_isolation_forest.md`. Score convention is **higher = more
  anomalous** (`score_samples` = `-sklearn.score_samples`). Optuna studies use
  a SQLite RDBStorage under `artifacts/tuning/` (`sqlite:///artifacts/tuning/optuna_iforest.db`)
  with `load_if_exists=True` for crash-recovery, and best params persist to
  `artifacts/tuning/best_params_iforest.yaml` incrementally (rewritten after every
  trial); the refitted best model lands in `artifacts/models/iforest.joblib` and figures
  in `artifacts/reports/figures/`.
- [x] `src/models` — VAE implementation (`vae.py`: `VAEModel`, `vae_loss`,
  `VAEDetector`, `tune_vae`, `plot_reconstruction_error`, `plot_latent_space`);
  see `docs/models_vae.md`. Score convention matches iForest: **higher = more
  anomalous**, where the VAE score is the per-row **MSE reconstruction error**
  (`score_samples` == `reconstruction_error`; default `score_kl_weight=0.0`).
  Loss is `reconstruction + beta*KL` (ELBO; `beta=1` vanilla, `beta!=1`
  beta-VAE). Consumes the preprocessing matrix directly (sparse input is
  densified to `float32`). Training does **per-epoch `checkpoint.pth`**
  crash-recovery (model+optimizer+epoch+best loss+config+history+numpy/torch RNG;
  atomic write under `artifacts/models/vae/`), `resume=True` continues from `epoch+1` when
  the architecture matches, and best weights are restored at the end. Optuna
  studies use a SQLite RDBStorage `sqlite:///artifacts/tuning/optuna_vae.db` with
  `load_if_exists=True`, best params persist incrementally to
  `artifacts/tuning/best_params_vae.yaml` (rewritten after every trial), the refitted
  best model lands in `artifacts/models/vae_best.pt`, and figures in
  `artifacts/reports/figures/`. NOTE: all internal `torch.load` calls pass
  `weights_only=False` (PyTorch>=2.6 defaults to `True` and refuses the
  checkpoint's config dict + numpy RNG state) — required for checkpoint resume;
  do not remove.
- [x] Optuna tuning integration + `artifacts/tuning/*.yaml` persistence (Isolation
  Forest done; VAE done)
- [x] Crash-recovery / checkpointing / resiliency layer (Isolation Forest
  tuning: SQLite study resume + incremental YAML checkpoint; VAE: per-epoch
  `checkpoint.pth` training resume + SQLite study resume + incremental YAML)
- [ ] `src/evaluation` — metrics + evaluation harness. **Hard deliverable:** an
  Excel export of individuals at/above P90 of the anomaly score in the OOT
  (out-of-time) month(s), sorted descending, table format **ID – SCORE –
  VARIABLES**, each row graded p90/p95/p99. OOT month = the last panel
  period(s) (`n_oot_periods`, default 1), reserved strictly AFTER `test` —
  **not the same block**: `test` is the once-touched block model metrics
  (ROC-AUC/PR-AUC/threshold diagnostics) are reported on, `oot` is data none
  of fitting, tuning, threshold calibration, or test-set reporting ever
  touched. See `chronological_split(..., n_oot_periods=...)`.
- [x] `src/interpretability` — SHAP / path-length attribution for the Isolation
  Forest (`iforest_explain.py`: `shap_summary_iforest`, `path_length_analysis`)
  and latent-space / per-feature reconstruction analysis for the VAE
  (`vae_explain.py`: `latent_space_plot`, `reconstruction_error_by_feature`);
  see `docs/interpretability_and_reporting.md`. `shap_summary_iforest` tries
  three tiers in order (native `shap.TreeExplainer` -> model-agnostic
  `shap.Explainer` over `score_samples` -> permutation importance) and logs
  which fired. Every function returns a plain `dict`/path and saves its figure
  under `artifacts/reports/figures/` (the hard figures rule).
- [x] `src/reporting` — HTML/MD report builder (`report.py`: `build_report`);
  see `docs/interpretability_and_reporting.md`. From a plain `context` dict
  (metrics, figure paths, OOT Excel path(s), model configs) it renders three
  self-contained deliverables under `artifacts/reports/`: a **Markdown** (`$$`
  ELBO/KL math, relative figure links), an **offline dashboard-style HTML**
  (inline CSS, every chart an interactive Plotly figure, no CDN/network, no
  embedded raster images, math as preformatted text), and a technical
  **`model_documentation.md`** (hyperparameters, threshold-calibration record,
  preprocessing settings, artifact catalog). No PDF is generated -- a PDF
  renderer existed here previously and was removed by explicit decision; see
  `docs/decisiones_de_modelado.md`. Missing `context` pieces are tolerated and omitted.
  **`context['oot_excel']` accepts a single path OR a `{model_name: path}`
  dict** — all entries render (e.g. both the iForest and VAE OOT exports).
  The HTML report (redesigned 2026-07-25 per user feedback that the original
  was "too basic") is a real dashboard, not a plain doc dump: a sticky
  header with jump-nav and a light/dark toggle (palette + status colors per
  the `dataviz` skill: light/dark CSS custom properties, fixed categorical
  accents per model — iForest = slot 1, VAE = slot 2, never cycled), dataset
  KPI stat tiles (rows/entities/periods/anomaly rate/OOT split), an "OOT
  deliverable" callout card linking every model's Excel export, per-model
  comparison cards with headline metric tiles carrying a **good/warning/
  serious status chip as visible text, never color alone** (thresholds in
  `_METRIC_META`/`_badge_for`), and a figure gallery auto-grouped into
  collapsible sections by `artifacts/reports/figures/<module>/` folder. The Markdown
  mirrors the same "Highlights" (headline metric + badge) framing.
- [x] `main.py` — full pipeline orchestrator. Phases: data (conditional
  load-or-generate) -> ground-truth labels -> preprocessing + statistical
  justification -> OOT split -> iForest (tune or fit) -> VAE (tune or fit) ->
  per-model evaluation -> per-model OOT Excel export -> per-model
  interpretability -> report. CLI flags include `--quick` (fast smoke),
  `--full` (spec-scale 100k x 10), `--tune/--no-tune`, transform/encoding
  choices. Verified end-to-end multiple times, incl. a 30,000-row
  (3,000 x 10) run: 2.09% imbalanced anomaly rate, iForest OOT ROC-AUC 0.71,
  VAE OOT ROC-AUC 0.74, both OOT Excels + all three report formats produced.
- [removed] `tests/` — the project carried a pytest suite (peaked at 319 tests
  across 11 files: data, preprocessing, both models, evaluation,
  interpretability/reporting, tuning budget, linear scaling, Optuna storage)
  used throughout development to verify every fix in this log. Removed by
  explicit decision (2026-08-22): not needed to run the project, so not
  shipped. `pytest` and `pytest.ini` were removed alongside it.
- [x] `docs/documentation.html` — all project docs (README, CONTEXT, the
  `docs/*.md` model/evaluation/interpretability docs, and the GeeksforGeeks
  reference notes) consolidated into one offline, sidebar-navigable HTML
  page for easier reading. Regenerate with `python docs/build_docs.py`
  after editing any source doc (`docs/build_docs.py` is a hand-written
  markdown->HTML converter, no new dependency) — **never hand-edit
  `docs/documentation.html` directly**, it's derived output.

When a module lands, check its box and add a one-line note here if it
introduces a new convention future modules should follow.

## Data contract

Future modules code against this contract, produced by `src/data`:

- **Panel shape**: a *balanced* panel keyed by `(entity_id, period)` —
  `n_individuals * n_periods` rows, every entity observed exactly once in
  every period. `period` values are consecutive month-starts (`freq="MS"`).
- **Main file (`artifacts/data/data.csv`)**: 22 columns, NO label columns. Exactly:
  `entity_id`, `period`, `age`, `tenure_months`, `region`, `segment`,
  `employment_status`, `marital_status`, `product_type`,
  `transaction_channel`, `is_digital_active`, `income`, `account_balance`,
  `monthly_transactions_amount`, `monthly_transactions_count`,
  `avg_transaction_amount`, `withdrawal_amount`, `credit_score`,
  `overdraft_count`, `num_products`, `days_since_last_login`,
  `customer_satisfaction_score`.
- **Ground truth is a SEPARATE file** (never a column in `data.csv`), so
  detection stays unsupervised by default. Columns: `entity_id`, `period`,
  `is_anomaly` (bool), `anomaly_type` (one of `global`, `local`,
  `contextual`, `collective`, `none`). It is written as parquet when a
  parquet engine (`pyarrow`) is installed, else it falls back to a sibling
  `.csv`. Consumers must use the path reported by the generator/loader, NOT
  assume the `.parquet` extension.
- **Schema-inference contract**: `load_or_generate_panel()` returns
  `(df, PanelSchema)`. `PanelSchema` carries `time_col` (`"period"`),
  `entity_col` (`"entity_id"`), `target_col` (`None` for synthetic data —
  labels live in the separate file; non-None only if a real dataset ships
  inline labels), and `ground_truth_path` (the real path of the separate
  ground-truth file, or `None` if none was located). `time_col`/`entity_col`
  can be `None` for arbitrary inputs; callers must handle that.
- **Anomaly semantics** (load-bearing for evaluation; full definitions in
  `src/data/synthetic.py`): `global` = extreme in the overall distribution;
  `local` = normal globally but inconsistent with the entity's own history
  (invariant: local anomalies must NOT be globally extreme); `contextual` =
  anomalous only given month/channel context; `collective` = individually
  unremarkable rows forming a synchronised near-identical cluster. The four
  types are mutually exclusive (at most one label per row).

## Preprocessing contract

Produced by `src/preprocessing` (`fit_transform_panel`), consumed by the
future modeling/evaluation modules:

- **Input**: `(df, schema)` exactly as returned by `load_or_generate_panel`
  — the raw panel DataFrame (keys still present) plus its `PanelSchema`.
- **Output**: `fit_transform_panel(df, schema, fit_mask=None, **config)` returns
  `(X, keys, feature_names)`. `X` is the model-ready feature matrix (sparse
  or dense); `keys` is the `(entity_id, period)` DataFrame held aside,
  row-for-row aligned with `X`, for the downstream join back to the separate
  ground-truth file — **entity/time are keys, not features**; `feature_names`
  is a list matching `X`'s column count.
- **`fit_mask` — the OOT leak fix, and the trap it avoids**: the pipeline has
  two stages and only one can leak. `PanelFeatureEngineer` estimates *nothing*
  (lags/diffs look strictly backwards, `own_z` uses only prior periods, ratios
  are within-row, the month encoding is a function of the timestamp) and runs
  over the **whole panel**. The `ColumnTransformer` estimates everything that
  can leak (imputation medians, scaler moments, Yeo-Johnson exponents, one-hot
  categories, frequencies, the `"auto"` per-column choice) and is fitted on
  `df[fit_mask]` only. `main.py` passes the in-time mask.
  **Do not "simplify" this to `fit(in_time)` then `transform(oot)`.** Handed
  only the last period, `shift(1)` finds no history inside that subset, so all
  18 lag/diff/own-z features become NaN → filled with `0.0` → identically zero
  on exactly the rows being evaluated. Guarded by
  `tests/test_preprocessing.py::TestFitMask`, which asserts both that the fix
  works and that the naive version still reproduces the failure.
- **Ratio features** (`_DEFAULT_RATIO_FEATURES`): `txn_amount_to_income`,
  `withdrawal_to_balance`, `balance_to_income`, `avg_txn_to_income`, guarded
  division (`den > 0`, else `0.0`). An IF splits on `uniform(min_f, max_f)` of
  one feature, so its boundaries are axis-parallel boxes; a ratio is constant
  along a ray through the origin, which boxes can only approximate with a
  staircase (one split per stair). Materialising the ratio makes that structure
  axis-aligned — what an Extended IF buys via oblique cuts, without the
  dependency.
- **Name-selectable transforms**: `numeric_transform` (one of
  `NUMERIC_TRANSFORMS`) and `categorical_encoding` (one of
  `CATEGORICAL_ENCODINGS`), plus the imputation / missing-indicator /
  panel-feature toggles, are all plain string/bool args so an Optuna study
  can tune each as a categorical hyperparameter. The pipeline is
  fit/transform-able and joblib-picklable for reuse at inference.
- **Defaults with a reason**: missing-indicator features are ON (upstream
  missingness is MNAR/informative — an anomaly cue). Within-entity panel
  features (lag/diff/own-history z-score/seasonality) exist to serve the
  `local` and `contextual` anomaly definitions and default ON in
  `fit_transform_panel`/`PanelFeatureEngineer` directly — but `main.py`'s CLI
  defaults them **OFF** (`panel_features: bool = False` on `PipelineConfig`,
  `--panel-features`/`--no-panel-features` to override; see "Panel features
  default OFF in main.py" below). Both toggles remain available for ablation.
- **Diagnostics**: `compute_transform_diagnostics` / `recommend_transform`
  justify the numeric transform per feature via scale-stable effect sizes
  (normality p-values are meaningless at ~1M rows). Figures from
  `plot_transform_diagnostics` land in `artifacts/reports/figures/` per
  the hard figures rule above.

## Leakage-free pipeline (7 phases)

Full derivation, per-phase rationale and the anti-leakage checklist live in
**`docs/leakage_free_pipeline.md`** — read that before touching splits,
preprocessing fits, tuning objectives or the threshold. Summary of the contracts
it locks in:

| Phase | Where | Contract |
| --- | --- | --- |
| 1 Features | `PanelFeatureEngineer` | `lag/diff/ratio` at horizons `(1, 3, 6)`, resolved against the **fit window**; ratios fill to `1.0`, not `0.0` |
| 2 Split | `chronological_split` | train / val / test / OOT strictly by period, no randomness; OOT (`n_oot_periods`, default 1) is reserved strictly after test and is never the same block |
| 3 Preprocessing | `fit_transform_panel(fit_mask=)` | stage 1 (causal) over the full panel, stage 2 (estimators) on train only |
| 4 Tuning | `tune_*(valid_mask=)` | static temporal holdout; label-free proxies `tail_separation` / `recon_p50`; VAE gets KL annealing + early stopping |
| 5 Final fit | `tune_*` refit | winning config refit on train+val |
| 6 Threshold | `calibrate_threshold` | POT/GPD (or percentile) fitted on **validation**, applied to test |
| 7 Deliverable | `export_oot_top_anomalies` | distinct individuals at/above P90 of the OOT score (default), graded p90/p95/p99; `--top-n` switches to a fixed headcount |

**Two hazards that bit us and are now guarded.** A short training window makes a
column near-constant *in the fit block*, and any scaler fitted there amplifies
unseen values without bound: `month_cos` reached **4.9e18** (blowing the VAE's
MSE to 1.8e35) before cyclical features were routed around the scaler, and
`lag6`/`diff6` did the same before horizons were validated against the fit
window. `_warn_on_extreme_magnitudes` now names any feature above `1e6` so this
class of bug cannot be silent again.

**Panel depth matters.** The default is now **15 periods** (`--quick` uses 12)
so a 10/2/3 chronological split leaves a training block deep enough for the
`h=6` contrast. With fewer periods `PanelFeatureEngineer` silently drops the
deep horizons — correct, but it means the feature is not being exercised.

## IF → VAE stacking (measured 2026-08-16)

`--stack-iforest-into-vae` (default **on**) appends the Isolation Forest's score
to the VAE's input matrix, and the VAE then ships the only Excel queue.
Implementation in `src/models/stacking.py`; full three-arm measurement in
`docs/leakage_free_pipeline.md` (Appendix).

**The measurement does not support the architecture on this data:**

| Arm | PR-AUC | ROC-AUC | Recovers the forest's top-50 |
| --- | ---: | ---: | ---: |
| VAE on raw X (parallel) | **0.1416** | **0.8045** | 16/50 |
| VAE on X + IF score (stacked) | 0.1251 | 0.7576 | 16/50 |

The appended column is **0.736% of the VAE's reconstruction MSE** (uniform share
= 0.769%), i.e. the VAE treats it as one ordinary feature among 130. Stacking
therefore *does not* transfer the forest's ranking: the stacked VAE's queue
covers the forest's picks exactly as poorly as the un-stacked one. Out-of-fold
scores changed nothing, for the same reason.

**Implication for the deliverable.** The premise "the forest's score is already
included, so one VAE queue suffices" is not borne out — the two rankings still
share only ~33% of their members. If a single queue is wanted, the evidence
points to **score-level rank combination**, not feature-level stacking. Flip
`--no-stack-iforest-into-vae` to restore the parallel arrangement (and both
Excels) at any time.

## Panel features default OFF in main.py (2026-08-17)

`PipelineConfig.panel_features` defaults to `False`, and `main.py`'s call to
`fit_transform_panel` passes `add_panel_features=config.panel_features`
accordingly — flip it back on with `--panel-features`.

**Why:** this pipeline's real-data usage computes within-entity
lag/diff/ratio/own-z + seasonality features in a separate upstream flow
outside this project, so generating them again here would duplicate/conflict
with that. The underlying `fit_transform_panel`/`PanelFeatureEngineer`
function still defaults `add_panel_features=True` when called directly (used
that way by several tests and by the synthetic-data workflow in the README) —
only `main.py`'s CLI-level default changed.

**Consequence.** With panel features off, the `local`/`contextual` anomaly
definitions lose their intended instrument (own-history lag/diff/z-score) —
see "OOT recall by anomaly type" below, measured with panel features on. A
run with `--no-panel-features` (the new default) has not yet been
re-measured against that table; treat those recall numbers as stale for the
new default until someone reruns the comparison.

## Observability, assumption gate, and tuning early-stopping (2026-08-19)

Three additive layers, built in this order because each is a prerequisite
for the next (a flow visualization needs real events to draw from; the
assumption gate's failures need a structured trace; tuning early-stopping is
told apart from the VAE's *existing* per-epoch stopping by the same event
schema).

**A. `src/utils/observability.py`.** A second, JSON Lines event channel
(`artifacts/logs/run_events.jsonl`) alongside the existing text logger
(`src/utils/logging_config.py`), which is unchanged for any caller that never
calls `observability.start_run()` — the 232 tests do not, and see zero
behavior difference. `log_phase` was extended (local import inside the
function, no new module-level dependency) to emit `phase_started`/
`phase_completed`/`phase_failed` for every existing call site across the
whole codebase, not just `main.py`'s 12 phases. `main.py` calls `start_run`/
`end_run` around the whole pipeline (failure path closes the run with
`status="failed"` from `main()`'s except-block, without needing to re-indent
`run_pipeline`'s existing linear body). Verified 2026-08-19 with a real
`--quick --no-tune` run (81 JSONL events, `run_ended` status="success",
`peak_python_memory_mb` via stdlib `tracemalloc` — no `psutil` added) and a
real forced failure (`--numeric-transform bogus_value_xyz`), which correctly
nested `phase_failed` and closed with `status="failed"` + the exception text.

**B. `src/utils/assumptions.py`.** Typed exception hierarchy
(`SchemaAssumptionError`, `DataQualityAssumptionError`, `LeakageAssumptionError`,
`TemporalSplitAssumptionError`, `IsolationForestAssumptionError`,
`VAEAssumptionError`, `ArtifactGenerationError`) — every raise also records a
failed `category="assumption"` health check via `observability.check(...)` in
the same call, so a blocking stop still leaves a structured trace. Wired into
`main.py` as a new **Phase 3b** (blocking: duplicate `(entity_id, period)`
keys, unparseable periods, infinite values — before Phase 4 preprocessing,
which is undefined over either) plus non-blocking diagnostics (null rates,
constant features, full-row duplicates) and a **person-overlap measurement**
in Phase 3a (diagnostic only, never raises — see the theoretical contrast
below). Phase 5's pre-existing bare `RuntimeError` for split/mask disagreement
is now `LeakageAssumptionError`. Phase 6/7 validate the feature matrix is
finite immediately before each model's `.fit()`, and Phase 6 sanity-checks
`contamination`. Verified with the real `--quick` run (Phase 3b passes in
0.14s) and by calling all four blocking checks directly with deliberately
broken inputs (duplicate keys, `inf`, `contamination=0.9`, a `NaN` matrix) —
each raised the correct typed exception naming the exact failing values.

**Measured person overlap (2026-08-19, `--quick`, this synthetic panel):**
train/test=100%, val/test=100%, 0 test entities never seen in train/val. This
is the **expected** reading for this project's balanced synthetic panel
(every entity observed in every period, by the Data contract above) — not
evidence of a defect by itself. The theoretical contrast: under a pure
rescoring/forecasting objective on a closed, stationary population, 100%
overlap is a designed property; it becomes a real leakage-adjacent risk claim
only once churn/attrition/new-entity-inflow are part of the evaluated
population, which this synthetic panel does not model. Real-data runs should
re-measure this (`assumptions.measure_person_overlap`, logged every run) and
revisit the claim against what the real portfolio actually does over time.

**C. `src/models/_tuning_stop.py` — `TrialPatienceStopper`.** Trial-level
early stopping for both `tune_iforest` and `tune_vae`'s Optuna studies,
distinct from the VAE's existing *per-epoch* early stopping inside one fit
(`VAEDetector.early_stopping_patience`, unrelated and untouched). An Optuna
trial is an independent draw from the search space, not one more step of the
same optimization, so there is no epoch-style "loss stopped decreasing" to
test; the rule instead is economic — stop once `patience` consecutive trials
(default 10) show no `min_delta`-relative improvement (default 0.5%) in
`study.best_value`, never before `min_trials` (default 10) trials complete.
Verified against the installed `optuna==4.9.0` API (`Study.stop()`) before
writing, and empirically: a 40-trial request on a saturating objective
stopped at trial 15 (best found at trial 9, skipped 25 trials), with the
`{model}.tuning_early_stopped` health check recording the exact counts.
Defaults are opt-out (`early_stopping_patience=None` disables it) and
`min_trials=10` guarantees zero behavior change for every existing test,
which all use `n_trials` in [2, 4].

**D/E. `src/evaluation/oot_report.py::export_p95_checkpoint`.** A new **Phase
6c**, between Phase 6 (Isolation Forest fit) and Phase 6b (IF -> VAE
stacking)/Phase 7 (VAE) in `main.py`, not wrapped in a lenient try/except on
purpose: `export_p95_checkpoint` raises `ArtifactGenerationError` (defined in
`assumptions.py`) on any validation failure, and letting that propagate is
what makes "the VAE does not start without a validated IF export" true rather
than aspirational. This is a distinct artifact from the existing Phase 9
OOT top-N/top-decile deliverable (`export_oot_top_anomalies`) — different
purpose (an immediate post-IF checkpoint of the whole panel vs. the final,
threshold-calibrated, per-model OOT review queue), different selection rule,
same underlying `openpyxl` export mechanism.

**P95 rule, decided explicitly (not left implicit):**
- Score = `detector.score_samples(X)` (project convention: higher = more
  anomalous).
- Threshold = the 95th percentile of scores **restricted to `in_mask`**
  (train+val, the rows the detector was allowed to see) — never OOT/test,
  mirroring how `calibrate_threshold` is fitted on validation only elsewhere
  in this pipeline.
- That fixed threshold is then **applied to every row of the panel**,
  in-time and OOT alike (scoring is not fitting, so this does not leak).
- Ties at the threshold are **included** (`score >= threshold`), so the
  exported set is `>= 5%`, not exactly 5%.
- Columns: every original column of `df` (never overwritten/subsetted) plus
  `isolation_forest_score`, `p95_threshold`, `run_id`, `split`.
  `isolation_forest_decision_function`/`_prediction` were considered and
  deliberately left out — both would need the preprocessed matrix `X`
  threaded through a function whose contract is otherwise "a df and its
  row-aligned scores."
- Validation before returning: file exists, non-empty, a full re-read via
  `pd.read_excel` reproduces the in-memory table's exact shape, plus a
  SHA-256 checksum, row/column counts, threshold, and timestamp all recorded
  as an `artifact`-category health check.

Verified 2026-08-19 both ways: a real `--quick` run selected 328-343/6000
rows (varies slightly run to run — expected, contamination/tuning shift the
forest, not a bug) with a passing health check and Phase 6b starting only
after Phase 6c logged complete; and a direct call with mismatched
`scores`/`df` lengths raised `ArtifactGenerationError` naming the exact
mismatched counts.

**F. Parameter reference tables** added to `docs/models_isolation_forest.md`
(section "2b", verified against installed `scikit-learn==1.7.2` via
`inspect.signature` — not recalled from memory, since sklearn has changed
`contamination`'s and `max_samples`'s defaults across versions before) and
`docs/models_vae.md` (section "2b", verified against installed
`torch==2.9.1+cpu`, this project's own architecture so no external default to
check but every value re-read from `VAEDetector.__init__`, `vae.py:315`).
Each row: default, meaning, valid alternatives (with the actual `tune_iforest`
/`tune_vae` Optuna search-space bounds where one is searched), and the
trade-off driving the choice. One correction made while writing the VAE table
against the real code rather than assumption: `activation` **is** validated
against a named set (`_ACTIVATIONS = {"relu", "leaky_relu", "elu", "tanh",
"gelu"}`, `ValueError` on anything else) exactly like `optimizer` — an
earlier draft of this note claimed otherwise before the code was checked.

**Trial-budget honesty note on `TrialPatienceStopper`.** A full `--quick`
pipeline run with `--iforest-trials 20 --vae-trials 12` completed 2026-08-19
without early-stopping firing for either model (0 of 13 health checks
failed, run healthy) — with `min_trials=10` and `patience=10` both at their
defaults, a stop is only reachable once at least `min_trials + patience` =
20 trials have run with the plateau starting right at trial `min_trials`, so
a 20-trial budget has essentially no margin to actually trigger it. The
40-trial isolated test above is the real evidence the mechanism *works*;
whether it fires in a given production run is a separate, expected function
of how large `n_trials` is relative to `patience`.

## Parameter centralization in `main.py` (2026-08-19)

Audited every place `run_pipeline` calls into a model/tuning/export function
and classified each parameter as already threaded through `PipelineConfig`
vs. silently defaulted at the callee. Design rule adopted: **each module
keeps its own sensible defaults** (`IsolationForestDetector`, `VAEDetector`,
`tune_iforest`, `tune_vae` all remain fully usable standalone, per the
README's "drive one of them directly" workflow, and every existing test that
constructs them directly is unaffected) — but `PipelineConfig` now has an
explicit field for every one of them, defaulting to the exact same value the
callee would use on its own, so **changing a full pipeline run's behavior
never requires editing anything outside `main.py`**.

**New `PipelineConfig` fields**, all wired into their call sites: `p95_percentile`
(was hardcoded `95.0` in Phase 6c), `iforest_holdout_frac`,
`iforest_tuning_early_stopping` / `vae_tuning_early_stopping` (dicts of
`patience`/`min_delta`/`min_trials` — the trial-level stopping added earlier
today was wired into `tune_iforest`/`tune_vae`'s function signatures but not
threaded from `main.py` until now), and `vae_params` (a new dict covering
every `VAEDetector` architecture/training arg used by the untuned fallback
fit: `latent_dim`, `hidden_dim`, `n_layers`, `dropout`, `beta`, `lr`,
`optimizer`, `batch_size`, `weight_decay`, `activation`, `kl_anneal_epochs`,
`early_stopping_patience`).

**Real bug the audit found and fixed, not just a tidiness pass:**
`iforest_params["contamination"]` (`0.02`) was only ever passed to the
*untuned* fallback `IsolationForestDetector(...)`. The *tuned* path's call to
`tune_iforest(...)` never passed `contamination=`, so it silently used that
function's own internal default (`_DEFAULT_CONTAMINATION = 0.10`,
`src/models/iforest.py:73`) instead — meaning **the effective contamination
changed depending on whether `--tune`/`--no-tune` was passed, with zero log
warning either way.** `tune_iforest`'s call now explicitly passes
`contamination=config.iforest_params.get("contamination", 0.02)`, so both
paths agree by construction. Verified 2026-08-19: a tuned run's log now reads
`contamination=0.02 (fixed)` (previously would have read `0.10`), and the
untuned fallback's log confirms the same `0.02`.

**New CLI flags**: `--contamination` (overrides `iforest_params["contamination"]`
directly, since this is the one users are most likely to want to sweep
per-run) and `--p95-percentile`. The tuning early-stopping dicts and
`vae_params` are deliberately **not** individually exposed as ~15 more CLI
flags — they are still fully centralized (one field each, in
`PipelineConfig`, editable directly in `main.py`), but a CLI flag per
VAE-architecture key was judged more surface area than value for parameters
that are normally swept *by the tuner*, not hand-set per run.

## Flow visualization (2026-08-19)

`src/reporting/flow_visualization.py::build_flow_visualization` renders
`artifacts/reports/flow_visualization.html` — an n8n/Databricks-style diagram
of one run's phases, built **only** from `run_events.jsonl` (Option A). No
state is invented: node status/duration/ordering come directly from
`phase_started`/`phase_completed`/`phase_failed` events; the only
non-literal part is the replay's per-step pacing (a fixed 700ms, like a
video scrubber), not the sequence or numbers themselves.

**Grouping rule, self-updating by design.** A "node" is any phase name
matching `^Phase \d+` (the convention every `main.py` top-level phase already
follows); everything else (`iforest.fit`, `preprocessing.fit_transform_panel`,
...) is a nested sub-event, attributed to whichever top-level phase was open
via file order in the JSONL (strictly append-only, single-process, so line
order is true emission order — no timestamp-precision issues even though
`observability._now()` only has second resolution). A trailing `[modelname]`
in a phase name becomes a `model` tag, so the per-model branch in Phases
8/8b/9/10 renders as distinct nodes without this file knowing `main.py`'s
phase names beyond that one regex. This module has zero hardcoded phase list
— verified 2026-08-19 that it correctly rendered **18 nodes** from a real run
including the true conditional branching ("Phase 9: top-N Excel deliverable"
appearing only for `model=vae`, never `model=iforest`, matching
`config.deliverable_models` under stacking).

Wired into `main.py` on **both** the success path (Phase 12b, after
`end_run()` so the diagram's own summary panel reflects final status) and the
failure path (`main()`'s except-block, so a crash still produces a diagram
showing exactly which phase stopped) — both best-effort, wrapped in
try/except, since a visualization bug must never mask (or be mistaken for) a
real pipeline failure. Verified both: a clean run produced 18 completed
nodes; a forced failure (`--numeric-transform bogus_xyz`) produced 4 nodes
ending in `status: failed`.

**Design pass.** Built as a UI/dashboard treatment (scanned, not read):
status encoded as color *and* text label *and* a left-border stripe on each
node (never color alone), semantic ok/bad colors kept separate from the
amber accent, tabular-nums on every numeric column, `prefers-reduced-motion`
respected, keyboard-operable nodes (`tabindex` + Enter/Space). Deliberately
**no Google Fonts / CDN** — system font stacks only (sans for labels, mono
for data/run-ids/durations) — this file inherits the project's existing,
explicitly-stated convention for its other HTML artifact (`report.py`:
"self-contained... no external CDNs/network, opens offline anywhere"), which
takes precedence over generic webfont advice for a file the pipeline itself
writes to disk on every run. Palette is a deliberately-chosen cool
slate-navy + amber (not the near-black-plus-blue-accent look most dev-tool
dashboards default to), correctly implemented as three CSS states (bare
`:root` light default, `prefers-color-scheme: dark` guarded by
`:not([data-theme="light"])`, and an explicit `[data-theme="dark"]` override)
so it also renders correctly if published as a claude.ai Artifact, which it
was for visual verification (this repo has no browser/screenshot tool of its
own, so publishing was the only way to actually confirm the interactive
replay renders/behaves as coded, not only that it parses).

**Superseded 2026-08-19 (same day, later): live local view added.** The
static replay above only appears *after* a run ends. `start_live_view`
(`src/reporting/flow_visualization.py`) starts a background `127.0.0.1`-only
HTTP server (stdlib `http.server`, daemon thread, no new dependency) the
moment `run_pipeline` starts, and `main.py` opens it in the default browser
immediately via `webbrowser.open`. Its one page polls `/state` every second;
`/state` rebuilds the same node/health-check structure fresh from
`run_events.jsonl` on every request, so the diagram grows in real time as
phases actually complete. Never a claude.ai Artifact publish, never reachable
off the local machine (binds `127.0.0.1`, not `0.0.0.0`). Controlled by
`PipelineConfig.live_view` (default `True`) / `--live-view`/`--no-live-view`.
Verified 2026-08-19 by running the pipeline as a background process and
polling `/state` twice, 15s apart, from outside that process: went from 4
nodes (`Phase 4` shown `running`) to 7 nodes (`Phase 4` now `completed`,
`duration_s=29.491`, `Phase 6` now `running`) -- genuinely live, not a
snapshot. The static post-run file (`build_flow_visualization`) still runs
too, for later review; the two are complementary, not redundant.

## Strategy default: unsupervised unless requested (2026-08-19)

`PipelineConfig.supervised` (default `False`) / `--supervised`/`--no-supervised`.
Previously `supervised = n_pos > 0` in Phase 3 -- the run's strategy silently
depended on whether a ground-truth file happened to exist, not on an
explicit choice, so the exact same code could evaluate supervised or
unsupervised depending only on what data happened to be sitting in
`artifacts/data/`. Ground truth is still always loaded (diagnostics need it
regardless), but it now only feeds the tuning objective and supervised
metrics when `--supervised` is passed; `n_pos > 0` remains a hard
requirement even then (an explicit request against a label-free dataset logs
a warning and falls back to unsupervised rather than raising or silently
computing degenerate PR-AUC). Verified 2026-08-19: a real run against ground
truth with 111 real positive labels (1.85% rate) still logged `->
UNSUPERVISED evaluation (strategy=default)`, confirming the presence of
usable labels no longer flips the strategy on its own.

## SHAP progress heartbeats (2026-08-19)

`shap_summary_iforest` (`src/interpretability/iforest_explain.py`) previously
went silent between `Starting interpretability.shap_iforest` and the next log
line, with a numpy `FutureWarning` in between that looks like a stray error
but is actually incidental evidence of *progress* (it only fires once
execution reaches the `shap.summary_plot(...)` call, i.e. after SHAP-value
computation already finished) -- easy to misread as a hang. Added explicit
start/finish log lines around both the `TreeExplainer` computation and the
beeswarm-plot rendering, and wrapped `_permutation_importance`'s per-feature
loop (path 3, the label-free fallback) in `tqdm`. Verified 2026-08-19 against
a real run: `Computing SHAP values via shap.TreeExplainer...` /
`shap.TreeExplainer finished.` / `Rendering SHAP beeswarm plot...` /
`SHAP beeswarm plot saved.`, each pair ~10s apart -- no more silent gap.

## Windows atomic-write retry (2026-08-19)

Hit in production: `PermissionError: [WinError 5] Access denied` on
`os.replace(tmp, path)` while checkpointing VAE tuning trial 29's
`best_model.pth` (`src/models/vae.py::_save_state`). Not a logic bug --
Windows (unlike POSIX) can transiently deny a rename/replace onto a
just-written file while antivirus/the search indexer briefly holds it open;
it happened on 1 of ~30 trials, not reproducibly. `src/utils/atomic_io.py::atomic_replace`
wraps every `os.replace(tmp, ...)` in this codebase (VAE checkpoints, VAE and
Isolation Forest best-params YAML -- all three call sites updated) with up to
5 retries, exponential backoff (0.1s/0.2s/0.4s/0.8s), **only** for
`PermissionError` -- any other exception (e.g. a genuinely missing source
file) still propagates on the first attempt, so this cannot mask an actual
bug the way a blanket retry would. Verified with a 4-case test: succeeds
immediately with no contention, recovers after simulated transient failures,
re-raises after exhausting retries against a simulated permanent lock, and
never retries a non-`PermissionError` exception.

## Compact yyyyMM period parsing (2026-08-19)

`pd.to_datetime(["202401"])` raises (`month must be in 1..12`) rather than
reading `2024-01-01` -- a bare 6-digit string is ambiguous without a format
hint, so pandas' dateutil fallback reads it as a 4-digit year plus a 2-digit
"month" of `01`...`41` extracted some other way and rejects the impossible
values. Real banking panels ship the period this way often enough that it
needed handling, not just a documented workaround (see the `horizons`/
`UnboundLocalError` conversation earlier this project for the original
report of this exact failure on a different codebase copy).

`src/data/loader.py::detect_period_format` inspects a column's values (after
normalizing numeric dtypes through nullable `Int64` first, so `202401` and
`202401.0` both match) against two explicit compact patterns --
`^\d{6}$` -> `"%Y%m"`, `^\d{8}$` -> `"%Y%m%d"` -- and `parse_period_column`
passes the detected format straight to `pd.to_datetime`, so parsing is
deterministic instead of left to dateutil's inference. Everything else
(`2024-01-01`, already-`datetime64`) still goes through plain
`pd.to_datetime(series)` unchanged. A genuine parse failure is logged and the
column is returned **unmodified**, not coerced with `errors="coerce"` --
silently turning every period into `NaT` would let `chronological_split` and
the panel feature engineer run over missing timestamps without ever raising;
`assumptions.validate_panel`'s `temporal.parseable` check is what turns an
unparsed period column into an actual stop. Verified against 6 cases (yyyyMM
string, yyyyMM int, yyyyMMdd, ISO strings, already-datetime, and genuine
garbage) -- all six parse or fail exactly as intended.

## Cancellation is now a detected, terminal state (2026-08-19)

Two independent gaps, found by actually sending real signals to a running
process rather than reasoning about the code:

**1. `except Exception` in `main()` never saw Ctrl+C.** `KeyboardInterrupt`
inherits from `BaseException`, not `Exception` -- confirmed the bug by
reading the exception hierarchy, not assumed. The run's event stream was
left mid-phase with no terminating `run_ended`, so the live view (added
earlier the same day) kept showing the last phase as "running" forever,
exactly the complaint that prompted this fix. `main()` now has a dedicated
`except KeyboardInterrupt:` above the existing `except Exception:`, sharing
a new `_close_run_as(status, error, live_view)` helper that both paths call.

**2. Ctrl+Break (Windows `SIGBREAK`) was not, and cannot by default, become
a `KeyboardInterrupt` at all.** Discovered empirically, not assumed: a bare
`time.sleep(30)` script with **no application code** hard-kills on
`CTRL_BREAK_EVENT` with `STATUS_CONTROL_C_EXIT`, before any Python `except`
clause runs, every time. `signal.getsignal(signal.SIGBREAK)` reads `0`
(`SIG_DFL`) by default -- unlike `SIGINT` (real Ctrl+C), which CPython wires
to `default_int_handler` out of the box. `main._install_sigbreak_handler`
now registers a handler that raises `KeyboardInterrupt` on `SIGBREAK` too
(Windows-only; a no-op everywhere else via `hasattr(signal, "SIGBREAK")`,
since plain Ctrl+C already works correctly on every platform without this).

**Fundamental limit, not fixed and not fixable without a much larger
redesign:** if the interrupt arrives while execution is inside a long-running
C-extension call (matplotlib rendering, some torch/scipy internals) rather
than in Python bytecode, CPython cannot act on a pending signal until that
call returns control -- confirmed by reproducing a hang-until-forced-kill
sending `CTRL_BREAK_EVENT` ~20s in, mid-`plot_transform_diagnostics`. A
`--quick` run interrupted at t=2s (pure pandas/Python) closed cleanly with
`run_ended`/`status: "cancelled"` and exit code 130 every time tested; one
landing inside a plotting call did not, in the time tested. Running each
phase in a killable subprocess would close this fully but is a materially
different architecture, not attempted here.

**Live view now shows "interrupted" even when the process dies before
writing anything.** `flow_visualization._LIVE_HTML`'s poll loop tolerates one
missed `/state` request (a transient hiccup) but on two consecutive misses
concludes the server -- and therefore the pipeline process that owns it --
is gone, and freezes the UI: any node still shown "running" is relabeled
"interrupted" client-side, without needing a final event from the Python
side at all. This is the layer that actually covers the C-extension-block
case above: even when `_close_run_as` never gets to run, a live view that
was open at the time still reflects the interruption instead of a stale
"running" snapshot. `build_flow_visualization` (the static, post-run file)
gained the equivalent server-side relabeling: any node whose `run_ended`
status is `"cancelled"`/`"failed"` but whose own phase status is still
`"running"` is written out as `"interrupted"`, for review after the fact,
in the (common) case the Python-side cleanup did run.

## Report redesign: Plotly, explanations, statistical rationale (2026-08-20)

`src/reporting/report_content.py` (new) owns the report's *meaning* --
indicator glossary, statistical-reliability rationale, parameter glossary,
ML-vs-econometrics positioning, and the interactive charts -- while
`report.py` keeps owning document assembly. The split exists because the
explanatory text is long and is edited for entirely different reasons than
the HTML/MD scaffolding.

**Interactive charts (Plotly 6.8.0, already a declared dependency).** Five
figures: ROC+PR (both models, OOT), one score-distribution histogram **per
model**, headline metric comparison, and recall by injected anomaly type.
Two rules shaped them:

* *Never a dual axis.* The two detectors' scores live on different scales
  (isolation depth vs reconstruction error), so score distributions are one
  chart **per model**, never overlaid. ROC and PR *are* overlaid across
  models because both axes are rates in [0,1]; they sit in separate subplots
  from each other because they answer different questions.
* *Honest baselines drawn on the chart.* The PR panel draws the panel's own
  anomaly rate as its dashed baseline (not 0.5), annotated with the value,
  because PR-AUC on a ~2% positive rate is otherwise read against the wrong
  reference.

Offline-safety is preserved: `plotly.js` is inlined once (~4.8MB; subsequent
figures cost ~7KB each), so the HTML is ~6.8MB and loads with **zero remote
requests** -- verified by scanning every `<script>/<link>/<img>/<iframe>` in
the output (23 tags, 0 pointing at a remote URL; the 16 `https://` strings
that *do* appear are map-tile attribution literals inside the plotly bundle,
which this report never renders).

**Colour.** Series use categorical slots 1-2 of the project's validated
reference palette, iForest always blue / VAE always orange, never cycled.
The palette validator was run against **this report's own surfaces**, not
the skill defaults: light `#ffffff` and dark `#171e2d` both pass all six
checks (worst adjacent pair ΔE 24.7 protan / 33.6 normal-vision light; 26.8 /
31.8 dark). A single fixed colour set was tried first and **FAILS** -- slot-2
orange `#eb6834` sits at OKLCH L 0.671, outside the dark band's 0.48-0.67 --
which is why per-theme hexes exist and why `THEME_RESTYLE_JS` recolours every
trace on load and on each theme toggle (Plotly cannot read CSS custom
properties, so figures cannot inherit the page's tokens the way everything
else does). Backgrounds are transparent so the panel shows through in both
themes. Identity never rests on colour alone: every chart legends its series.

**Anti-overlap.** Every figure reserves `margin.t=56` with the title pinned
at `y=0.97`, and bar charts set `cliponaxis=false` so outside value labels
are not clipped -- verified present in all five emitted figures. Chart
containers carry `min-width:0` because a flex/grid child defaults to
min-content width, which is what lets a wide SVG push a page into horizontal
scroll.

**Figures actually removed, not just reordered.** Five static PNGs are now
dropped from the report per run (logged as such):
score-distribution and reconstruction-error PNGs and both ROC/PR PNGs are
strictly less informative than their interactive replacements (which also
draw the calibrated alert threshold); PCA embeddings are dropped where a UMAP
embedding of the same matrix is present. A genuine duplicate was found and
fixed: **"VAE latent space" was being emitted twice by two different
modules** (`models.vae.plot_latent_space` and
`interpretability.vae_explain.latent_space_plot`), both reaching the report.
The ten raw-feature histograms are kept but demoted to a collapsed group
placed last -- they answer "is the input sane", which the reliability section
already asserts.

**Figure grouping was silently broken and is fixed.** `_group_figures` keyed
off the `reports/figures/<module>/` parent folder, but that layout was
deliberately flattened long ago (CONTEXT.md "HARD RULE -- Figures"), so every
figure's parent became the single word "figures" and *all* groups collapsed
into one bucket titled "Figures" nested under the "## Figures" heading --
visible in the Markdown output as a duplicated heading. It now groups by
filename prefix, which is the convention the flattening was justified by.

**New sections, ordered for a reader rather than for the code:** results
(charts) -> per-model detail -> how to read each indicator -> why the results
are statistically trustworthy -> parameters and what each value means ->
ML-vs-econometrics -> diagnostic figures. The reliability section documents
eight real gates (chronological split, train-only preprocessing fits, causal
panel features, key integrity, finite matrix, validation-calibrated
threshold, person-overlap measurement, rank stability), each with what it
verifies, why it matters, and what a failure would mean. The parameter
tables mark each value <span>this run</span> (selected by the tuner) or
<span>default</span>, so the report says what the run actually did rather
than only what the defaults are.

**On the ML-vs-econometrics question:** the report states the verdict as
machine learning (unsupervised anomaly detection), and answers "which
parameter drives the classification" explicitly -- it is **the calibrated
threshold**, not any model coefficient, since neither detector exposes an
interpretable coefficient at all. It also names the two parameters most
easily mistaken for it: the Isolation Forest's `contamination` (moves only
`predict()`'s internal boundary, not the ranking this pipeline uses) and the
top-N queue size (an operational capacity decision).

## Ruteo de features por dtype: iForest numérico, VAE completo (2026-08-20)

**Qué cambió.** Los dos detectores ya no reciben la misma matriz. Las columnas
derivadas de variables categóricas se **excluyen del Isolation Forest** y se
**mantienen para el VAE**. La identificación es puramente por **tipo de dato**:
`build_preprocessing_pipeline` rutea con
`make_column_selector(dtype_include=["object", "category"])` y el
`ColumnTransformer` marca esas salidas con el prefijo `cat__`, así que
`categorical_feature_mask` / `split_matrix_for_model`
(`src/preprocessing/pipeline.py`) sólo leen ese prefijo. **No hay lista
hardcodeada de columnas**: una columna de texto nueva en los datos de origen
se rutea sola, sin tocar código. Las llaves (`entity_id`, `period`) nunca
entran — `PanelFeatureEngineer` las descarta antes, porque son llaves, no
features.

**Por qué la asimetría (no es arbitraria):**

* El **Isolation Forest** parte con `uniform(min, max)` sobre *una* feature a
  la vez, así que cada corte es una afirmación de orden ("por debajo / por
  encima de este valor"). Una columna one-hot es 0/1 sin interior
  significativo: todo corte degenera en "tiene este nivel / no lo tiene", y
  con categóricas de alta cardinalidad el encoding aporta muchas columnas
  casi constantes que diluyen el muestreo de `max_features` sin agregar
  estructura aislable.
* El **VAE** reconstruye su vector de entrada completo, y un bloque one-hot es
  reconstruible exactamente como la arquitectura espera. Además el contexto
  categórico es informativo para la definición de anomalía `contextual` — un
  monto ordinario en un canal y extremo en otro.

**Medido 2026-08-20** (`--quick`): 67 features totales, 44 derivadas de
categóricas → el Isolation Forest ve 23 features numéricas, el VAE ve las 67.
El detector de stacking usa la misma vista numérica que el forest principal
(un score apilado calculado sobre otro conjunto de features no sería la misma
cantidad). Cubierto por `TestPerModelFeatureRouting` (3 tests), incluyendo uno
que agrega una columna de texto nueva y una numérica con nombre "categórico"
para confirmar que el ruteo sigue el dtype y **no** el nombre.

## Interpretabilidad movida después de los Excel (2026-08-20)

La Fase 10 salió del bucle por modelo y ahora corre después de que **todos**
los deliverables Excel están en disco. Motivo: la interpretabilidad es la
etapa más lenta del pipeline (SHAP sobre el forest, la compilación numba
única de UMAP) y no produce ningún entregable propio, así que ejecutarla
dentro del bucle dejaba la cola Excel del VAE esperando detrás del cálculo
SHAP del forest. Verificado 2026-08-20: Fase 9 (Excel) termina 15:50:49,
Fase 10 arranca 15:50:50.

## El reporte respeta la estrategia no supervisada (2026-08-20)

**Corrección de un error propio introducido el día anterior.** Al agregar los
gráficos Plotly se calcularon métricas supervisadas (ROC/PR, comparación de
métricas, recall por tipo) *sin importar* el valor de `config.supervised`,
con el argumento de que evaluar contra ground truth offline es su propósito
documentado. Eso contradice el default explícito del proyecto: la estrategia
por defecto es **no supervisada**, y un reporte que muestra curvas ROC está
afirmando una evaluación que la corrida no realizó.

Ahora `chart_data` adjunta etiquetas **sólo** si `supervised` es verdadero, y
los tres gráficos con etiquetas están condicionados a eso. El glosario de
indicadores también se filtra: sólo lista los indicadores que la corrida
realmente produjo, porque explicar ROC-AUC en un reporte que (correctamente)
nunca lo calculó invita a buscar números que no existen. Verificado: 0
menciones de ROC-AUC / PR-AUC / Precision@10 / Recall@10 en el cuerpo del
reporte de una corrida por defecto.

**Gráfico nuevo que sí aporta sin etiquetas — acuerdo entre detectores.**
Scatter de percentil de score bajo cada detector, con rho de Spearman y
solapamiento del top-5% anotados. Es el diagnóstico central cuando no hay
ground truth: los dos modelos parten de principios distintos (geometría de
aislamiento vs error de reconstrucción) y **ahora además ven conjuntos de
features distintos**, así que un individuo que ambos rankean alto está
corroborado por evidencia independiente; los que quedan altos en un eje y
bajos en el otro marcan desacuerdo genuino y son interesantes precisamente
por eso.

**Leyendas fuera del área de título.** Estaban en `y=1.0` (orientación
horizontal arriba), que es la misma banda donde vive el título — de ahí la
superposición. Ahora van abajo (`y=-0.22`, ancladas al centro) con
`margin.b=86` reservando esa franja.

## Pantalla de carga con progreso real (2026-08-20)

La vista en vivo mostraba sólo el diagrama de nodos. Ahora encabeza con
porcentaje, barra de progreso, fase actual con spinner, y conteo
`hechas/vistas` + segundos de trabajo acumulados. El porcentaje se calcula
como completadas/vistas y se **etiqueta como tal** ("phases done"), no como
avance total: el pipeline descubre sus fases desde los eventos conforme
ocurren, así que no existe un total conocido de antemano y presentarlo como
"% del run" sería inventar una cifra.

## Baseline and open problems (measured 2026-08-01)

Two runs, both `python main.py --quick`, seed 42. **They are not directly
comparable** — the panel depth, the split geometry and the feature set all
changed — so the table records what each configuration was, not a clean A/B.

| | iForest | VAE | configuration |
|---|---|---|---|
| OOT ROC-AUC | 0.656 | 0.825 | 800×6, 2-way in/OOT split, 1 test month, lag1 only |
| OOT PR-AUC | 0.094 | 0.414 | " |
| OOT ROC-AUC | **0.799** | 0.816 | 500×12, 3-way chronological, **3** test months, horizons (1,3,6) |
| OOT PR-AUC | **0.137** | 0.140 | " |

The **iForest improves substantially** (ROC-AUC 0.656 → 0.799) once the
multi-horizon contrast features exist — consistent with the Phase 1 argument
that a single-period lag under-reports a shock.

The **VAE's PR-AUC drops** (0.414 → 0.140) and this is *not* yet explained.
Confounded by all three changes at once; the honest next step is a controlled
one-factor-at-a-time sweep before drawing any conclusion. Candidates worth
testing in order: the test block growing from 1 to 3 months (it now contains
"plateau" repeats of the same event, which changes the positive mix), the
narrowed search space (`beta` ≤ 2.0, `dropout` ≥ 0.1), and the 129-column
feature matrix being harder to reconstruct. Do not tune this away without
measuring which factor it is.

An older `artifacts/tuning/best_params_iforest.yaml` reported
`best_value: 0.0833`. That number is **not comparable to anything**: it was an
*in-sample* PR-AUC from a study that had pooled 60 trials across different
datasets and preprocessing configs.

### OOT recall by anomaly type — the point of the instrumentation

`anomaly_type` had always been in the ground truth and *nothing* read it
(`labels.py` only loaded `is_anomaly`), so an aggregate PR-AUC could not say
which geometry was failing. It now can:

| type | iForest recall@10 % | mean score pctile |
|---|---|---|
| global | 0.600 | 0.855 |
| contextual | 0.600 | 0.648 |
| **local** | **0.000** | **0.457** |

**`local` is the open problem.** The forest ranks a local anomaly *below the
median row*. By construction these values are drawn from inside the population
`[p2, p95]` band and are anomalous only against the entity's own history, so
no marginal geometry separates them; `_own_z` is the intended instrument and is
not sufficient. Next step is here, not in more hyperparameter search.

### Measured conflict: the two models want opposite numeric transforms

Controlled sweep (fixed data, fixed hyperparameters, only `numeric_transform`
varies) — see `docs/models_isolation_forest.md` for the full table:

| transform | max\|x\| | IF PR-AUC | VAE PR-AUC |
|---|---|---|---|
| `robust` / `standard` (affine, shape-preserving) | 5.3×10⁵ | **0.272** | **NaN — total failure** |
| `yeo-johnson` (default) | 47 | 0.117 | **0.414** |
| `log1p` | — | 0.089 | — |
| `auto` (per-column, min `abs_skewness`) | — | 0.057 | — |

The IF wants the heavy tail intact (extremes isolate in few splits; compressing
the right tail is compressing exactly where `global` anomalies live). The VAE
cannot survive it — MSE gradients on a scaled value of 5×10⁵ overflow to NaN.

**Decisions taken:** the shared default stays `yeo-johnson` (one side degrades
gracefully, the other fails outright). `auto` is implemented and tested but is
**not** the default — the sweep shows minimising skewness is the wrong criterion
for this task. Genuinely resolving this needs a per-model shape transform (two
matrices), which is an architecture change, not a default change.

Note this is the *correct* version of a rejected proposal to "bifurcate the
scaling per model": scaling is provably a no-op for the IF
(`TestAffineRescalingIsANoOp`) and the pipeline already delivers scaled data to
the VAE, so that bifurcation was empty. The real conflict is over **shape**.

## Scale characteristics

Measured on the synthetic generator (`generate_synthetic_panel`):

- **Throughput**: ~25 µs/row. 50k rows ≈ 1.3 s; the 1M-row default
  (`n_individuals=100_000 * n_periods=10`) ≈ 22–25 s.
- **Memory / disk (1M-row default)**: ~1.4 GB peak RSS during generation;
  `artifacts/data/data.csv` ≈ 192 MB on disk. Reloading yields a ~0.56 GB in-memory
  DataFrame, dominated by object-dtype string columns.
- **Standing recommendations**: cast categorical columns to `category` on
  load to cut memory sharply, and consider persisting the panel as parquet
  (rather than CSV) once size matters.

## Conventions

- **Logging**: always obtain the logger via `setup_logging()` in
  `src/utils/logging_config.py` (idempotent, writes to `artifacts/logs/execution.log`
  and console). Wrap any long-running phase (data gen, preprocessing,
  training, tuning) with the `log_phase(name)` context manager from the
  same module so start/end/duration are logged consistently.
- **Figures**: see the HARD RULE above — everything goes under
  `artifacts/reports/figures/`.
- **Hyperparameters**: tuning outputs (e.g. Optuna best params) are
  persisted as YAML under `artifacts/tuning/` (e.g. `artifacts/tuning/best_params_iforest.yaml`,
  `artifacts/tuning/best_params_vae.yaml`).
- **Checkpoints/weights**: trained model artifacts (`.pth`, `.pkl`) go under
  `artifacts/models/`.
- **Environment setup**: run `python setup_validator.py` before doing
  anything else in a new environment; it checks Python version and
  dependencies and attempts to auto-install anything missing. `pyarrow`
  (parquet engine for ground truth) is in `requirements.txt` and validated too.
- **Testing**: removed from the project (2026-08-22) — not needed to run it.
  See the `tests/` entry in the checklist above for what the suite covered
  while it existed.
