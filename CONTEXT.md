# CONTEXT.md — Project Memory

This file is persistent memory for future sessions/agents working on this
project. Read it before making changes. It holds the **current** architecture,
contracts and defaults only — the history of how each one was reached (dated,
with measurements and verification notes) lives in **`CHANGELOG.md`**, so this
file stays short enough to read in full before touching code.

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
├── CONTEXT.md              # this file — current architecture/contracts
├── CHANGELOG.md            # dated history: fixes, measurements, decisions
├── .gitignore              # artifacts/ and caches stay out of version control
├── src/
│   ├── data/               # panel loader + synthetic generator
│   ├── preprocessing/      # pipeline (transforms, panel features, ratios) + diagnostics + linear_scaling
│   ├── models/              # Isolation Forest + VAE + stacking + trial early stopping
│   ├── evaluation/         # OOT split, GT join, metrics, thresholds, OOT Excel export
│   ├── interpretability/   # SHAP, path length, latent space, per-feature recon
│   ├── reporting/          # HTML/MD report builder + flow visualization (no PDF)
│   └── utils/              # paths, logging, observability, assumptions gate, atomic_io
├── docs/                   # *.md sources + generated documentation.html
└── artifacts/              # EVERYTHING the pipeline writes (gitignored)
    ├── data/               # data.csv + ground_truth.parquet (hidden labels)
    ├── logs/               # execution.log + run_events.jsonl
    ├── models/             # iforest.joblib, vae_best.pt, vae/, vae_tuning/
    ├── tuning/             # optuna_*.db, best_params_*.yaml (Optuna outputs)
    └── reports/            # anomaly_report.{html,md}, model_documentation.md, oot_p90_*.xlsx, feature_attribution.xlsx, flow_visualization.html
        └── figures/        # ALL figures, flat (see rule below)
```

`artifacts/models/vae_tuning/` grows one subdirectory per Optuna trial for
crash-resume. That is expected and transient — delete it once a study has
finished; nothing downstream reads it.

**`tests/` was removed (2026-08-22)** — not needed to run the project, so not
shipped; `pytest`/`pytest.ini` went with it. Some `docs/*.md` files still name
specific `Test*`/`test_*.py` files as the historical source of a claim; those
names no longer resolve to anything on disk.

## Paths — single source of truth

**Never hardcode an artifact path.** `src/utils/paths.py` holds every one of
them and is the only module that knows the `artifacts/` layout; each consumer
takes its default from there (`_DEFAULT_MODEL_OUT = paths.IFOREST_MODEL`, etc.).
Moving the tree again is a one-file change.

Two properties those values must keep: **relative, never absolute**, and
built with `os.path.join`, so separators are the standard library's problem.

## HARD RULE — Figures

**All generated plots/figures/charts, from ANY module, MUST be saved under
`artifacts/reports/figures/`** (i.e. `paths.FIGURES_DIR`). Project-wide
convention, no exceptions: not in `artifacts/data/`, not next to scripts, not in
`artifacts/models/`.

The directory is **flat**. Every filename already begins with its producer
(`iforest_*`, `vae_*`, `embedding_*`, `roc_pr_*`), so a subdirectory-per-module
layout was dropped — keep filenames unique when adding a figure.

## Tech Stack

- Python 3.12+
- scikit-learn (Isolation Forest)
- torch (VAE)
- numpy, pandas, scipy, statsmodels (data/panel handling, stats)
- optuna (hyperparameter tuning)
- shap, umap-learn (interpretability)
- matplotlib, plotly, seaborn (visualization)
- tqdm, joblib, rich, psutil (progress, parallelism/persistence, console dashboard)
- pyyaml (config files)

## Data contract

Produced by `src/data`, consumed by every downstream module:

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
  `contextual`, `collective`, `none`). Written as parquet when a parquet
  engine (`pyarrow`) is installed, else falls back to a sibling `.csv`.
  Consumers must use the path reported by the generator/loader, NOT assume
  the `.parquet` extension.
- **Schema-inference contract**: `load_or_generate_panel()` returns
  `(df, PanelSchema)`. `PanelSchema` carries `time_col` (`"period"`),
  `entity_col` (`"entity_id"`), `target_col` (`None` for synthetic data —
  labels live in the separate file), and `ground_truth_path` (or `None` if
  none was located). `time_col`/`entity_col` can be `None` for arbitrary
  inputs; callers must handle that.
- **Period parsing**: `src/data/loader.py::detect_period_format` /
  `parse_period_column` handle compact formats pandas cannot infer on its
  own — `^\d{6}$` -> `"%Y%m"`, `^\d{8}$` -> `"%Y%m%d"` (after normalizing
  numeric dtypes through nullable `Int64` first, so `202401` and `202401.0`
  both match). Anything else goes through plain `pd.to_datetime`. A genuine
  parse failure is logged and the column returned **unmodified**, never
  coerced with `errors="coerce"` — `assumptions.validate_panel`'s
  `temporal.parseable` check is what turns an unparsed period column into an
  actual stop, instead of silent `NaT`s flowing into the split/feature code.
- **Anomaly semantics** (load-bearing for evaluation; full definitions in
  `src/data/synthetic.py`): `global` = extreme in the overall distribution;
  `local` = normal globally but inconsistent with the entity's own history
  (invariant: local anomalies must NOT be globally extreme); `contextual` =
  anomalous only given month/channel context; `collective` = individually
  unremarkable rows forming a synchronised near-identical cluster. The four
  types are mutually exclusive (at most one label per row).
- **Synthetic-data provenance**: `generate_synthetic_panel` writes a marker
  file next to the panel, named after the panel file itself
  (`data.csv.synthetic.json`, not one marker per folder), and
  `PanelSchema.is_synthetic` propagates it. `main.py` redirects a
  non-official run's model/params/study under `artifacts/**/_dev/` so a
  synthetic-data run can never overwrite real-data artifacts. An unreadable
  or mismatched marker resolves to **real data** (the conservative,
  noisy-if-wrong direction) — see `CHANGELOG.md` 2026-08-22/23 for the
  regression this guards against.

## Preprocessing contract

Produced by `src/preprocessing` (`fit_transform_panel`), consumed by every
modeling/evaluation module:

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
  on exactly the rows being evaluated.
- **Feature routing by dtype (`iForest` numeric-only, `VAE` full)**. The two
  detectors do **not** receive the same columns. `split_matrix_for_model(X,
  feature_names, "iforest" | "vae")` gives the Isolation Forest only the
  non-categorical-derived columns, and the VAE the full matrix. Routing is
  purely by the **source column's dtype** —
  `build_preprocessing_pipeline` selects `dtype_include=["object", "category"]`
  via `make_column_selector`, the `ColumnTransformer` tags those outputs with
  a `cat__` prefix, and `categorical_feature_mask`/`split_matrix_for_model`
  (`src/preprocessing/pipeline.py`) key off that prefix — never a hardcoded
  column-name list, so a new text column routes itself. Keys (`entity_id`,
  `period`) never enter either matrix. **Why the asymmetry**: an Isolation
  Forest split is `uniform(min, max)` over one feature, and a one-hot column
  is 0/1 with no meaningful interior — every cut degenerates to "has this
  level / doesn't", and high-cardinality categoricals dilute `max_features`
  sampling without adding isolable structure. The VAE reconstructs its whole
  input vector, and the categorical context is informative for the
  `contextual` anomaly definition (an ordinary amount on one channel, extreme
  on another). The stacking detector (below) uses the same numeric-only view
  as the standalone forest.
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
- **Numeric imputation defaults to `"zero"`** (`SimpleImputer(strategy=
  "constant", fill_value=0.0)`), alongside `"median"`/`"mean"`/
  `"most_frequent"` (`--no-zero-impute` restores `"median"`). Zero estimates
  nothing from the data, so it cannot leak between train/test and does not
  drift with the fit window the way a recomputed median does. The
  trade-off: a zero is a real value, not a "missing" symbol, so on a column
  where 0 already means something (empty balance, zero transactions) an
  imputed zero is indistinguishable from a genuine one — `add_missing_
  indicators=True` (default) is what keeps that recoverable, via a 0/1
  flag per column that had a NaN. Disabling indicators while keeping
  zero-imputation is the combination to avoid.
- **Defaults with a reason**: missing-indicator features are ON (upstream
  missingness is MNAR/informative — an anomaly cue). Within-entity panel
  features (lag/diff/own-history z-score/seasonality) exist to serve the
  `local` and `contextual` anomaly definitions and default ON in
  `fit_transform_panel`/`PanelFeatureEngineer` directly — but `main.py`'s CLI
  defaults them **OFF** (`panel_features: bool = False` on `PipelineConfig`,
  `--panel-features`/`--no-panel-features` to override). **Why:** this
  pipeline's real-data usage computes within-entity lag/diff/ratio/own-z +
  seasonality features in a separate upstream flow outside this project, so
  generating them again here would duplicate/conflict with that.
  **Consequence:** with panel features off, the `local`/`contextual` anomaly
  definitions lose their intended instrument (own-history lag/diff/z-score) —
  see "Known open problems" below.
- **Diagnostics**: `compute_transform_diagnostics` / `recommend_transform`
  justify the numeric transform per feature via scale-stable effect sizes
  (normality p-values are meaningless at ~1M rows). Figures from
  `plot_transform_diagnostics` land in `artifacts/reports/figures/` per
  the hard figures rule above.
- **Stateless linear scaling** (`src/preprocessing/linear_scaling.py`) is a
  separate, independent path from the sklearn pipeline above: pure
  numpy/pandas affine rescale (`y = a*x + b`), no `.fit()`, no estimator
  object — for EDA or a lighter-weight serving path. Robust
  `(x - median)/IQR` by default. See `docs/escalamiento_lineal.md`.

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
| 4 Tuning | `tune_*(valid_mask=)` | static temporal holdout; label-free proxies `rank_agreement` (IF) / `recon_p50` (VAE); VAE gets KL annealing + early stopping |
| 5 Final fit | `tune_*` refit | winning config refit on train+val |
| 6 Threshold | `calibrate_threshold` | POT/GPD (or percentile) fitted on **validation**, applied to test |
| 7 Deliverable | `export_oot_top_anomalies` | distinct individuals at/above P90 of the OOT score (default), graded p90/p95/p99; `--top-n` switches to a fixed headcount |

**Two hazards that bit us and are now guarded.** A short training window makes a
column near-constant *in the fit block*, and any scaler fitted there amplifies
unseen values without bound: cyclical features (`month_sin`/`month_cos`) now
bypass the scaler entirely via a `passthrough` branch, and contrast horizons
(`lag6`/`diff6`) are validated against the fit window so a short window keeps
only the horizons it can actually support. `_warn_on_extreme_magnitudes` names
any feature above `1e6` after transformation, so this class of bug cannot be
silent again.

**Panel depth matters.** The default is now **15 periods** (`--quick` uses 12)
so a 10/2/3 chronological split leaves a training block deep enough for the
`h=6` contrast. With fewer periods `PanelFeatureEngineer` silently drops the
deep horizons — correct, but it means the feature is not being exercised.

**Numeric transform default is a cross-model compromise, not either model's
optimum.** The Isolation Forest and VAE want opposite treatment of
distribution *shape*: `robust`/`standard` (affine, shape-preserving) score
better for the Isolation Forest but blow the VAE's MSE to `NaN` on the
untouched heavy tail (~5e5 in scaled units); `yeo-johnson` is the shared
default because it degrades gracefully for both. `auto` (per-column, min
`abs_skewness`) is implemented but not the default — minimising skewness is
the wrong criterion for this task. See `CHANGELOG.md` 2026-08-01 for the
measured numbers, and `docs/models_isolation_forest.md` §"Measured" for the
per-anomaly-type breakdown.

## IF → VAE stacking

`--stack-iforest-into-vae` (default **on**) appends the Isolation Forest's
score (fitted on train only) to the VAE's input matrix
(`src/models/stacking.py`); the VAE then ships the only Excel queue. A
three-arm measurement (`CHANGELOG.md` 2026-08-16; full detail in
`docs/leakage_free_pipeline.md` Appendix) found stacking does **not**
transfer the forest's ranking — the stacked VAE's top-50 overlaps the
forest's top-50 exactly as much as the plain parallel VAE does (16/50), because
the appended column is ~0.7% of the VAE's reconstruction loss, i.e. one
ordinary feature among many. If a single combined queue is wanted, the
evidence favours **score-level rank combination** over feature-level
stacking. Flip `--no-stack-iforest-into-vae` to restore the parallel
arrangement (both models export their own Excel).

## Model routing, defaults and known-current behavior

- **Strategy default is unsupervised.** `PipelineConfig.supervised` (default
  `False`) / `--supervised`/`--no-supervised`. Ground truth is still always
  loaded (diagnostics need it regardless), but it only feeds the tuning
  objective and supervised metrics when `--supervised` is passed explicitly —
  the mere presence of a labels file no longer flips the strategy on its own,
  and the report only shows supervised charts/glossary entries (ROC/PR,
  metric comparison, recall-by-type) when the run actually computed them.
- **OOT deliverable — the headline business artifact.** Each detector writes
  `artifacts/reports/oot_p90_<model>.xlsx` by default: every individual at or
  above the **90th percentile** of the OOT score, graded `p90`/`p95`/`p99`
  (the highest band it reaches, cut-offs computed over the full de-duplicated
  OOT population, never over the exported subset), sorted by score descending
  (`export_oot_top_anomalies`, `src/evaluation/oot_report.py`). Layout is
  **ID – PERIOD – SCORE – BAND – VARIABLES**, plus an `alert` column when a
  calibrated threshold is supplied. One row per individual: with more than one
  OOT month, each entity is represented by its highest-scoring month, and
  that month is the one recorded in the period column. `--top-n N` switches
  to a fixed headcount instead (the older "top 50" behavior); `--oot-min-
  percentile` changes the percentile cut. Threshold: `--threshold-method pot`
  (default) fits a Generalized Pareto to the validation score tail and
  inverts it for a target false-alarm rate (`--threshold-target-far`);
  `--threshold-method percentile` uses `--threshold-percentile`. Always
  calibrated on validation, applied to test only.
- **P95 inter-layer checkpoint (distinct from the deliverable above).**
  Immediately after the Isolation Forest fits and before the VAE starts,
  `export_p95_checkpoint` (`src/evaluation/oot_report.py`, `main.py` Phase
  6c) writes `artifacts/reports/p95_checkpoint_iforest.xlsx`: every row at or
  above the 95th percentile of scores restricted to `in_mask` (train+val,
  never OOT/test), with every original column preserved, applied to the
  whole panel. The artifact is validated (exists, non-empty, a full re-read
  reproduces the in-memory shape, checksum) and a failure raises
  `ArtifactGenerationError` — the VAE phase does not start without a
  validated export. This is a mid-run gate, not the final OOT review queue.
- **Interpretability runs after all Excel exports**, not inside the
  per-model loop — it is the slowest stage (SHAP over the forest, one-time
  UMAP compilation) and produces no deliverable of its own, so running it
  earlier would leave the VAE's Excel queue waiting behind the forest's SHAP
  computation.
- **`shap_summary_iforest`'s two fallback paths are wall-clock/call budgeted,
  not unbounded.** Both `shap.TreeExplainer` failing over to the
  model-agnostic `shap.Explainer`, and that in turn failing over to manual
  permutation importance, cost roughly `O(n_features)` work per row/repeat —
  fine on this project's 20-70 feature synthetic panel, but at ~150-200
  features (a realistic real-data feature count) the model-agnostic path
  measured **~2.7 hours to explain 2000 rows**, which is what "the pipeline
  hangs/crashes during interpretability" actually was. The model-agnostic
  path now times a couple of calibration rows and explains only as many as
  fit a ~60s budget (`_MODEL_AGNOSTIC_TIME_BUDGET_S`,
  `src/interpretability/iforest_explain.py`); the permutation-importance
  fallback subsamples rows and reduces `n_repeats` to keep total
  `score_samples()` calls under a fixed budget
  (`_PERM_IMPORTANCE_CALL_BUDGET`). Both degrade gracefully to a smaller,
  still-representative sample rather than a smaller time — never the other
  way around. See `CHANGELOG.md` 2026-08-26 for the measurements.
- **Full feature-attribution workbook.** Each attribution chart (SHAP
  beeswarm, reconstruction-error bars) is cropped to its top 20 variables for
  readability; `export_attribution_workbook`
  (`src/interpretability/attribution_export.py`) writes the uncropped values
  for **every** variable to `artifacts/reports/feature_attribution.xlsx`, one
  sheet per model (`mean_abs_shap` for the forest, `mean_reconstruction_error`
  for the VAE — not comparable across sheets, so each sheet's row 1 states
  its own methodology).

## Observability, assumption gate, and tuning early-stopping

- **`src/utils/observability.py`** — a second, JSON Lines event channel
  (`artifacts/logs/run_events.jsonl`) alongside the text logger
  (`src/utils/logging_config.py`); unused code paths (anything that never
  calls `observability.start_run()`) see zero behavior difference.
  `log_phase` emits `phase_started`/`phase_completed`/`phase_failed` for
  every phase across the codebase, not just `main.py`'s top-level ones.
  `main.py` calls `start_run`/`end_run` around the whole pipeline, including
  the failure and cancellation paths (see below), so a crash still closes
  the run with a structured status instead of leaving the stream mid-phase.
- **`src/utils/assumptions.py`** — typed exception hierarchy
  (`SchemaAssumptionError`, `DataQualityAssumptionError`,
  `LeakageAssumptionError`, `TemporalSplitAssumptionError`,
  `IsolationForestAssumptionError`, `VAEAssumptionError`,
  `ArtifactGenerationError`); every raise also records a failed
  `category="assumption"` health check via `observability.check(...)`, so a
  blocking stop still leaves a structured trace. Wired into `main.py` as
  **Phase 3b** (blocking: duplicate `(entity_id, period)` keys, unparseable
  periods, infinite values) plus non-blocking diagnostics (null rates,
  constant features, full-row duplicates) and a person-overlap measurement
  (diagnostic only, never raises — 100% train/test entity overlap is the
  *expected* reading for this project's closed, balanced synthetic panel; a
  real-data run with churn/attrition should re-measure and revisit that
  assumption). Phase 6/7 validate the feature matrix is finite immediately
  before each model's `.fit()`, and Phase 6 sanity-checks `contamination`.
- **`src/models/_tuning_stop.py::TrialPatienceStopper`** — trial-level early
  stopping for both `tune_iforest` and `tune_vae`'s Optuna studies, distinct
  from the VAE's *per-epoch* early stopping inside one fit
  (`VAEDetector.early_stopping_patience`). Stops the study once `patience`
  consecutive trials (default 10) show no `min_delta`-relative improvement
  (default 0.5%), never before `min_trials` (default 10) trials complete —
  opt-out via `early_stopping_patience=None`. With `min_trials=10` and
  `patience=10` both at defaults, a stop is only reachable once at least 20
  trials have run, so small trial budgets (e.g. `--quick`'s 5) have no
  margin to actually trigger it; that is expected, not a bug.
- **Windows atomic-write retry** (`src/utils/atomic_io.py::atomic_replace`)
  wraps every `os.replace(tmp, ...)` in the codebase (VAE checkpoints, both
  models' best-params YAML) with up to 5 retries, exponential backoff, **only
  for `PermissionError`** — any other exception still propagates immediately.
  Windows (unlike POSIX) can transiently deny a rename onto a just-written
  file while antivirus/the search indexer briefly holds it open; this is rare
  (hit once in ~30 VAE tuning trials in production) but not reproducible on
  demand, so the retry exists rather than a one-off patch.
- **Cancellation is a detected, terminal state, with one known limit.**
  `main()` has a dedicated `except KeyboardInterrupt:` (separate from
  `except Exception:`, since `KeyboardInterrupt` inherits from
  `BaseException`) sharing a `_close_run_as(status, error, live_view)` helper
  with the failure path. On Windows, `main._install_sigbreak_handler` also
  turns Ctrl+Break (`SIGBREAK`) into a `KeyboardInterrupt` — unlike Ctrl+C,
  Python does not wire that up by default, and an unhandled `SIGBREAK`
  hard-kills the process before any `except` clause runs. **Fundamental
  limit**: if the interrupt arrives while execution is inside a long-running
  C-extension call (matplotlib rendering, some torch/scipy internals),
  CPython cannot act on it until that call returns control — this needs a
  killable-subprocess-per-phase architecture to close fully, not attempted
  here. The live view's poll loop covers the gap partially: after two
  consecutive missed `/state` requests it assumes the pipeline process is
  gone and relabels any node still "running" as "interrupted" client-side,
  without needing a final event from the Python side.

## Reporting

- `build_report(context, ...)` (`src/reporting/report.py`, content in
  `report_content.py`) renders a run into `artifacts/reports/`: a
  **Markdown**, an **offline dashboard-style HTML** (inline CSS, every chart
  an interactive Plotly figure — `plotly.js` inlined once, zero remote
  requests — light/dark toggle, fixed categorical accents per model, iForest
  = slot 1, VAE = slot 2, never cycled, status shown as a visible text chip
  never color alone), and a technical **`model_documentation.md`**
  (hyperparameters, threshold-calibration record, preprocessing settings,
  artifact catalog). No PDF is generated — see `docs/decisiones_de_modelado.md`.
  `context['oot_excel']` accepts a single path OR a `{model_name: path}` dict.
- Five interactive figures: ROC+PR (both models, OOT; PR draws the panel's
  own anomaly rate as its baseline, not 0.5), one score-distribution
  histogram **per model** (never a dual axis — the two detectors' scores
  live on different, non-comparable scales), a headline metric comparison,
  recall by injected anomaly type, and a detector-agreement scatter (score
  percentile under each detector, annotated with Spearman rho and top-5%
  overlap) — the last is the central diagnostic when there is no ground
  truth, since the two detectors use different principles and (per the
  dtype-routing contract above) different feature sets.
- Charts and the glossary are conditioned on `config.supervised`: a run that
  did not compute a supervised metric does not show it, and does not list it
  in the indicator glossary either.
- `src/reporting/flow_visualization.py` renders an n8n-style diagram of one
  run's phases from `run_events.jsonl` alone (`build_flow_visualization`,
  post-run) — a "node" is any phase name matching `^Phase \d+`, everything
  else nests as a sub-event under whichever top-level phase was open, so the
  module has no hardcoded phase list. `start_live_view` additionally serves
  the same structure live over a `127.0.0.1`-only HTTP server
  (stdlib `http.server`, daemon thread) from the moment `run_pipeline`
  starts; `main.py` opens it in the browser automatically. Controlled by
  `PipelineConfig.live_view` (default `True`) / `--live-view`/`--no-live-view`.
- During a run, `main.py` shows a live console dashboard (`rich`, disables
  itself when stdout is not a terminal or `rich` is missing; `--no-console-ui`
  forces it off): a fixed 15-row phase checklist, an "Assumptions (IF/VAE)"
  panel fed by the same `observability.check(...)` calls, and a team-health
  line (RAM/CPU, always with the number visible, never color alone).

## Known open problems

- **`local`-type anomalies are unrecovered.** The Isolation Forest ranks a
  `local` anomaly at roughly the population median (recall@10% ≈ 0 across
  every numeric transform tried), because by construction a `local` anomaly
  sits inside the population's normal band and is anomalous only against the
  entity's own history — `_own_z` is the intended instrument and is
  evidently not sufficient on its own. See `docs/models_isolation_forest.md`
  §"Measured" and `CHANGELOG.md` 2026-08-01 for the numbers. This needs
  feature/architecture work, not more hyperparameter search.
- **VAE health was not independently validated until recently.** See
  `docs/diagnostico_del_proyecto.md` for the fullest current account — it
  found (2026-08-22/23) that the VAE's loss scaling made its effective
  `beta` scale with the number of features (`src/models/vae.py::vae_loss`),
  which collapsed the posterior on every real run to date; both the loss
  scaling and the (now-`beta`-invariant) tuning objective have since been
  corrected in `src/models/vae.py`, and posterior-collapse detection
  (`VAEDetector.latent_diagnostics`, `collapse_verdict`, gated in `main.py`
  Phase 7) now runs on every fit. **Any `best_params_vae.yaml` or VAE result
  produced before 2026-08-22 was tuned/measured under the old, incorrect
  scaling and should be treated as stale**, including the stacking
  measurement above and the numeric-transform table's VAE column — both
  predate the fix and have not been re-run against it.
- **`--quick`'s VAE epoch budget may be too small to avoid collapse even with
  the scaling fix**, and its Optuna trial budget is small enough that
  `TrialPatienceStopper` essentially never fires (see above) — treat a
  `--quick` VAE result as a smoke test, not a measurement, until this is
  revisited (`docs/diagnostico_del_proyecto.md` A-10).
- **No cross-seed stability measurement exists yet** for either detector — a
  single fixed seed (`PipelineConfig.seed = 42`) runs today;
  `unsupervised_metrics`'s `rank_stability` is a bootstrap-jitter proxy for
  score sensitivity to noise, not a re-fit-under-a-different-seed measurement.
  See `docs/validacion_no_supervisada.md` §6 for the proposed design.
- **The Isolation Forest permanently runs a sub-optimal numeric transform**
  for its own objective (`yeo-johnson`, not `robust`) because the VAE cannot
  survive `robust` — see "Leakage-free pipeline" above. Worth re-measuring
  once the VAE fix above has had a chance to change how it behaves under a
  heavy-tailed input.

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
- **Where the pipeline's wall-clock actually goes** (a reference `--quick`
  run): evaluation's rank-stability metric (which refits the model several
  times) dominates at ~100 s, ahead of preprocessing (~25 s) and
  interpretability (~23 s) — model fitting itself (~9 s IF, ~6 s VAE) is not
  the bottleneck. See `docs/decisiones_de_modelado.md` §4.4 for the full
  per-phase table.

## Conventions

- **Logging**: always obtain the logger via `setup_logging()` in
  `src/utils/logging_config.py` (idempotent, writes to `artifacts/logs/execution.log`
  and console). Wrap any long-running phase (data gen, preprocessing,
  training, tuning) with the `log_phase(name)` context manager from the
  same module so start/end/duration are logged consistently, and so it emits
  the `phase_started`/`phase_completed`/`phase_failed` observability events.
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
- **Testing**: there is no test suite in this project (removed 2026-08-22,
  see `CHANGELOG.md`) — verify changes by running the pipeline directly
  (`python main.py --quick`) and reading its log/health checks.
