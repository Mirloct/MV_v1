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
├── tools/                  # external/adjacent tooling -- NOT part of the
│   │                       # pipeline import graph; see "IF-VAE Diagnostic
│   │                       # Suite integration" below
│   ├── export_diagnostic_suite_inputs.py
│   └── if_vae_diagnostic_suite/   # vendored external package (own venv-less
│                                  # pip install -e ., own tests/, own AGENTS.md)
└── artifacts/              # EVERYTHING the pipeline writes (gitignored)
    ├── data/               # data.csv + ground_truth.parquet (hidden labels)
    ├── logs/               # execution.log + run_events.jsonl
    ├── models/             # iforest.joblib, vae_best.pt, vae/, vae_tuning/
    ├── tuning/             # optuna_*.db, best_params_*.yaml (Optuna outputs)
    └── reports/            # anomaly_report.{html,md}, model_documentation.md, oot_p90_*.xlsx, feature_attribution.xlsx, flow_visualization.html, analyst_dashboard.html
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
- **Exact-zero row filter**: `drop_exact_zero_rows` (Phase 2, before any split/fit) removes rows whose numeric/boolean input columns are >= `exact_zero_row_cutoff` (default 0.90, `--exact-zero-row-cutoff`) exact 0. `NaN` is not 0; keys, target and categoricals are outside the denominator. Rows/dropped/kept are logged to `execution.log`. Distinct from the Phase 9d post-hoc study, which removes nothing.
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
| 2 Split | `chronological_split` | train / val / test / OOT strictly by period, no randomness; OOT (`n_oot_periods`, default 3) is reserved strictly after test and is never the same block |
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

**Per-run validation export (2026-09-07).** The 2026-08-16 measurement above
answers "does stacking transfer the forest's ranking" once, in the
aggregate, on one historical run. It does not tell an analyst whether
*this specific run's* stacked VAE queue dropped an individual the forest
alone would have flagged. "Phase 6d: IF OOT export (validation)"
(`main.py`, right after the forest is fit, before its score is folded into
the VAE's matrix) answers that directly: when stacking is on, it exports
the forest's OWN OOT queue (`export_oot_top_anomalies`, same ID-PERIOD-
SCORE-BAND-VARIABLES layout, same P90/P95/P99 grading, own threshold
calibrated on validation) to `artifacts/reports/oot_p90_iforest.xlsx`
*alongside* the VAE's stacked `oot_p90_vae.xlsx` — the P95 checkpoint export
right before it is unaffected and still runs. An analyst compares the two
files' entity sets directly; on the `--quick` synthetic fixture this found
21 of 50 individuals in the forest's own top-P90 queue that do **not**
appear in the stacked VAE's top-P90 queue — a real, per-run divergence, not
necessarily a defect (the two detectors are not supposed to agree
perfectly; that is exactly the question "IF-VAE Diagnostic Suite
integration" below cross-validates from a different angle). Guarded by
`if config.stack_iforest_into_vae:` — in parallel mode the forest already
gets this exact export via Phase 9's per-model loop, so Phase 6d would only
recompute and overwrite an identical file; it is skipped there, not run
twice. Best-effort (`severity="warning"`): a failure here never blocks the
P95 checkpoint, stacking, or the VAE deliverable.

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
- **The report's headline "anomalías marcadas para revisión" figure is NOT
  the `alert`/calibrated-threshold count above.** It is read directly off
  the just-written OOT export: `int(_table[BAND_COL].isin(("p95",
  "p99")).sum())` (`main.py` Phase 9) -- count of individuals in that exact
  file at or above P95, always non-zero when the export has any rows,
  since it never depends on a POT calibration that can fail to find a
  usable tail on real, unlabeled data (the calibrated `alert` column can
  legitimately be 0; this cannot, short of an empty export). Deliberately
  not the ground-truth positive count either (`n_pos`/`anomaly_rate` from
  Phase 3) -- that answers "how many are truly anomalous," unknowable in
  production and the reason this figure previously showed 0 on a real run
  with no ground-truth file. Falls back to the ground-truth count only if
  no deliverable model ran at all. Under `--no-stack-iforest-into-vae`
  (two deliverables), the last one processed in the per-model loop wins
  (VAE, always last in `models = {"iforest": ..., "vae": ...}`).
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
- **All three of `shap_summary_iforest`'s paths (`shap.TreeExplainer`, the
  model-agnostic `shap.Explainer`, and manual permutation importance) are
  wall-clock/call budgeted, not unbounded.** Every path's cost scales with
  tree complexity and/or feature count — fine on this project's small
  synthetic panel and small training blocks, but confirmed (not just
  theorized) to reach minutes-to-hours in two independent, realistic
  scenarios: ~150-200 features (`src/interpretability/iforest_explain.py`,
  paths 2/3), and — the actual root cause behind a real production hang —
  **`max_samples` tuned to a float fraction (Optuna's search space allows
  0.3-1.0) combined with a multi-month training block**. `max_samples` as a
  fraction is relative to the *training set size*, not a fixed row count, so
  a 200,000-row train block (e.g. 10 months × 20k entities) with
  `max_samples=0.5` builds trees with ~100,000-row leaves and ~17 levels of
  depth instead of the ~256-row/~8-level trees `max_samples="auto"` always
  produces regardless of dataset size — and `shap.TreeExplainer`'s cost scales
  with tree depth/leaf count, not just feature count. Measured: this
  configuration alone projects to **~5.7-17+ minutes** to explain 2000 rows
  unbounded, matching the reported symptom almost exactly. All three paths
  now calibrate on a handful of rows against the real model/data, extrapolate
  a per-row cost, and only explain as many rows as fit a fixed time/call
  budget (`_TREE_EXPLAINER_TIME_BUDGET_S`, `_MODEL_AGNOSTIC_TIME_BUDGET_S`,
  `_PERM_IMPORTANCE_CALL_BUDGET`, all in `iforest_explain.py`) — degrading to
  a smaller, still-representative sample rather than a smaller time budget.
- **`shap.TreeExplainer` and the model-agnostic `shap.Explainer` run in an
  isolated child process with a real, enforced kill ceiling** —
  `_TREE_EXPLAINER_HARD_KILL_S` / `_MODEL_AGNOSTIC_HARD_KILL_S`
  (`_run_with_hard_kill`, `iforest_explain.py`). The soft time budget above
  only protects against slowness *after* the calibration call returns; a
  real production run hung 3+ hours with no forward progress, past every
  soft budget, proving that assumption wrong. No in-process timer (a thread
  with a timeout, a signal handler) can stop a blocked call into shap's
  C/Cython internals from Python — the only way is to run it in a separate
  OS process and forcibly terminate that process (`SIGTERM`, then
  `Process.kill()` after a grace period) if it does not finish in time. Both
  paths are wrapped this way; a kill is treated exactly like an exception —
  the next path is tried. The child still reports its calibration
  measurement back to the parent before attempting the (budget-bounded) full
  explain, so a kill does not erase the fine-grained checkpoints below, only
  the final "done" one never arrives. Verified: a worker that would block for
  300s is confirmed killed at the configured ceiling (not left running), with
  no child process left behind afterward; the exact reconstructed
  200,000-row/`max_samples=0.5` scenario above still completes in ~113s
  through the isolated path.
- **Every part of interpretability that runs leaves a checkpoint trace —
  not just the SHAP paths.** All three interpretability modules
  (`iforest_explain.py`, `vae_explain.py`, `attribution_export.py`) define a
  module-local `_checkpoint(name, **observed)` recording an always-passing
  `observability.check(...)` under its own namespace —
  `interpretability.iforest.<name>` (covers `shap_summary_iforest`,
  `path_length_analysis`, **and `explain_rows_iforest`**, since all three
  live in `iforest_explain.py`), `interpretability.vae_explain.<name>`
  (covers `latent_space_plot`, `reconstruction_error_by_feature`, **and
  `explain_rows_vae`**), and `interpretability.attribution_export.<name>`
  (Phase 10b, the per-model Excel-sheet writer that runs right after Phase
  10 and right before the report — `started` → one `sheet_written` per model
  → `completed`). Every meaningful sub-step gets one: calibration
  started/measured, full explain started/done, a `*_hard_killed` name when
  the ceiling above actually fires, beeswarm render started/done, UMAP
  started/done, permutation-importance progress every ~25% of features,
  path-length analysis started/completed, VAE encode/UMAP/batch-loop steps,
  attribution-workbook sheet writes, and (2026-08-28) the per-row
  `explain_rows_iforest`/`explain_rows_vae` functions' own
  calibration/explain/progress steps — these two ran with **zero**
  checkpoints until then despite `explain_rows_iforest` reusing the exact
  same hang-prone `shap.TreeExplainer` subprocess path as
  `shap_summary_iforest`, a real coverage gap now closed with the identical
  checkpoint names/shape (`explain_rows_calibration_started` ->
  `explain_rows_calibrated` -> `explain_rows_explain_started` ->
  `explain_rows_done`/`explain_rows_hard_killed`/`explain_rows_failed` ->
  `explain_rows_completed`). Every checkpoint is appended to
  `artifacts/logs/run_events.jsonl`; whichever name is *last* in the log is
  exactly the sub-step that was in flight when the process stopped
  advancing — e.g. a `tree_explainer_calibration_started` with no matching
  `tree_explainer_calibrated` after it means the calibration call itself
  hung (the one sub-step the budget above cannot preempt).
- **Live view: interpretability's checkpoints get their own line, not just
  a spot in "Supuestos" (2026-08-28).** Previously every
  `observability.check(...)` project-wide — genuine IF/VAE assumption gates
  *and* interpretability's routine progress pings alike — fed one shared,
  8-slot "Supuestos (IF / VAE)" deque in the console dashboard
  (`src/utils/console_ui.py`); a burst of a few dozen interpretability
  checkpoints during Phase 10 could flush every real assumption result out
  of view, under a panel title that only names IF/VAE. `ConsoleUI._on_check`
  now routes anything named `interpretability.*` to a separate 3-slot deque
  (`_interp_checks`) instead, so "Supuestos" is assumption-gates-only again.
  That deque feeds a **new sub-step line directly under the current-phase
  readout** — `↳ interpretabilidad  <last up-to-3 checkpoint names, oldest
  first> <time on the latest> (<n> checkpoints)` — the "one level more of
  detail" a phase-only progress bar cannot give: "Phase 10 is running" says
  nothing about whether it is progressing or stuck, but a sub-step frozen in
  place with a growing timer next to it is an unambiguous stall signal, live,
  without reading `execution.log`. Stays visible (frozen on its last value)
  after Phase 10 finishes, the same way the phase checklist keeps its green
  boxes lit.
- **Full feature-attribution workbook.** Each attribution chart (SHAP
  beeswarm, reconstruction-error bars) is cropped to its top 20 variables for
  readability; `export_attribution_workbook`
  (`src/interpretability/attribution_export.py`) writes the uncropped values
  for **every** variable to `artifacts/reports/feature_attribution.xlsx`, one
  sheet per model (`mean_abs_shap` for the forest, `mean_reconstruction_error`
  for the VAE — not comparable across sheets, so each sheet's row 1 states
  its own methodology).
- **VAE feature attribution: categorical granularity.** One-hot encoding
  turns one string column into one column *per category*; since the VAE's
  score and its per-feature reconstruction-error attribution are both sums
  **over columns**, a high-cardinality categorical (many one-hot slices) can
  out-weigh a single numeric column in both the ranking and, if granular
  enough, the score itself — not because it is more informative, but because
  there are more columns representing it. Two additive, non-breaking pieces
  address this:
  - `reconstruction_error_by_feature(..., categorical_columns=[...])`
    (`src/interpretability/vae_explain.py`) — when given the *original*
    (pre-transform) categorical column names, logs and checkpoints
    (`interpretability.vae_explain.categorical_contribution`) what share of
    total reconstruction error vs. column count the categorical-derived
    block carries (over-represented / roughly proportional /
    under-represented — a measured fact, not an assumption), and the bar
    chart groups one-hot slices back under their source variable
    (`src/preprocessing/pipeline.py::aggregate_attribution_by_source`,
    `group_name_by_source`) so the ranking is a fair comparison. The
    **returned dict stays the full, ungrouped detail** regardless — grouping
    only touches the chart/diagnostic, never silently the data.
    `main.py` always passes this (`df.select_dtypes(include=["object",
    "category"]).columns`).
  - `export_attribution_workbook(..., categorical_columns=[...])` writes an
    additional `vae_by_source` sheet (grouped) alongside the existing
    uncropped `vae` sheet (still fully granular) — both are always written
    together when the run has categorical columns, so a reviewer can compare
    either view.
  - **If the diagnostic shows genuine over-representation and the ranking/
    grouping above is not enough** (i.e. the *score itself*, not just the
    report, is judged too categorical-driven): `--rare-min-frequency`
    (`PipelineConfig.rare_min_frequency`, default `0.001`, threaded to
    `fit_transform_panel`) collapses low-frequency categories into one
    bucket *before* one-hot encoding, shrinking column count per categorical
    while keeping identity for common categories — raise it. The more
    drastic lever, `--categorical-encoding frequency` (or `ordinal`),
    collapses each categorical to **one** numeric column regardless of
    cardinality, eliminating the effect entirely — at the cost of the VAE
    losing the specific category identity that the "contextual" anomaly
    definition relies on (see "Feature routing by dtype" above); this
    changes what the VAE is trained on and needs a re-tune, so treat it as a
    deliberate trade-off, not a default fix.

- **Per-row explanation: "why is this individual flagged" as a column in
  the OOT deliverable, not just an aggregate ranking.** Every attribution
  function above (`shap_summary_iforest`, `reconstruction_error_by_feature`)
  explains a *representative subsample* to build one population-level
  ranking — it never answers "what drove this specific person's score."
  `explain_rows_iforest` (`src/interpretability/iforest_explain.py`) and
  `explain_rows_vae` (`vae_explain.py`) are the per-row complement: given
  exactly the rows to explain (not subsampled — an alert queue, not a
  population sample), they return one comma-joined string of the top-`k`
  feature names **per row**, in the same order. `main.py` (Phase 9) computes
  this for exactly the OOT rows (`oot_period`-derived mask, the same
  population `export_oot_top_anomalies` selects from — not the whole panel)
  right after scoring and attaches it to `scored_df` as **`top_5_variables`**
  before the Excel export, so it rides along as an ordinary column (it lands
  last, after the raw feature columns, since `export_oot_top_anomalies`
  preserves `scored_df`'s column order).
  - **Isolation Forest**: reuses `shap.TreeExplainer` via the *same*
    isolated-child-process, hard-kill-guarded path as `shap_summary_
    iforest`'s path 1 (`_run_with_hard_kill`) — the same tree-depth-driven
    slowness applies here too, just against a much smaller, fixed row set.
    Ranked by `|SHAP value|` per row. Never blocks the deliverable: on any
    failure or a hard-kill, affected rows get `None` rather than raising,
    and a warning names how many rows were left unexplained.
  - **VAE**: always exactly computable, no fallback needed — per-row
    squared reconstruction error `(x_ij - x̂_ij)^2`, ranked per row. When
    `categorical_columns` is given (`main.py` always passes it), one-hot
    columns are summed back under their source variable **per row** first
    (same `group_name_by_source` mechanism as the aggregate diagnostic
    above), so the column never leaks a raw one-hot slice name like
    `cat__region_North` — it reports `region`.
  - Verified: injecting a deliberately extreme value into a known column
    makes that column top the explanation for that exact row (both
    detectors); a full pipeline run's real OOT export carries a fully
    populated (0 nulls), per-row-differentiated `top_5_variables` column.

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
  recall by injected anomaly type, and a detector-agreement density heatmap
  — the central diagnostic when there is no ground truth, since the two
  detectors use different principles and (per the dtype-routing contract
  above) different feature sets.
  - **Population, made exact (2026-08-28):** built from
    `chart_data["models"][name]["true_oot_entity_scores"]`
    (`main.py` Phase 8) — `{entity_id: max score across the genuine
    `oot_mask` window}` — the same one-row-per-individual dedup
    `export_oot_top_anomalies` applies for the alert queue. This
    deliberately does **not** reuse `oot_scores` (misleadingly named:
    that's `scores[eval_mask]`, the **test** block — see the `eval_mask`
    vs `oot_mask` split above — and not deduplicated by entity), so the
    chart and the Excel deliverable are now provably the same population
    (verified: with `n_oot_periods=2`, both report the identical entity
    count).
  - **Rendering:** a `go.Histogram2d` (Plotly's Cartesian equivalent of a
    hexbin — true hexbin binning is mapbox-only) of rank percentile under
    each detector, Viridis-coloured by individual count, with a y=x
    reference line and the top-5%-x-top-5% quadrant (score ≥ its model's
    95th percentile on both axes) outlined by a shape. Spearman rho and
    the count of individuals inside that quadrant are kept as an
    annotation. A second panel (`go.Heatmap`, `make_subplots`) shows the
    same population as a 4x4 quartile confusion matrix (count + % per
    cell) — the table view for a fast read.
- Charts and the glossary are conditioned on `config.supervised`: a run that
  did not compute a supervised metric does not show it, and does not list it
  in the indicator glossary either.
- The report may close with **“Qué no se ejecutó o falló”**. `run_pipeline`
  attaches an `IncidentCollector` at ERROR level before the first phase,
  passes its records into `build_report`, and detaches it after report
  generation so repeated in-process runs cannot leak incidents into one
  another. Routine warnings are intentionally excluded from HTML/Markdown,
  and `main.py` applies `warnings.filterwarnings("ignore")` to Python/library
  warnings. The table is a quick-glance mirror of genuine failures;
  `execution.log` remains the authoritative runtime log.
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
  forces it off): a fixed 15-row phase checklist, an "Supuestos (IF/VAE)"
  panel fed by genuine assumption/gate `observability.check(...)` calls, a
  dedicated interpretability sub-step line under the current-phase readout
  (see "Live view: interpretability's checkpoints..." above), and a
  team-health line (RAM/CPU, always with the number visible, never color
  alone).
- **New phases need no registration to appear in either flow view** — both
  read `run_events.jsonl` directly and treat any `^Phase \d+[a-z]?` name as a
  node, so wrapping a block in `with log_phase("Phase 6d: ..."):` is
  sufficient by itself (confirmed for "Phase 6d: IF OOT export (validation)"
  below). The console dashboard's checklist is the one place that DOES need
  a manual entry — `_PHASE_PLAN` (`src/utils/console_ui.py`) is a fixed,
  ordered list used only for the upfront pending-checklist display and the
  progress-percentage weighting; an unregistered phase still runs and still
  logs, it just appears dynamically instead of as a pre-drawn pending row.
- **Elapsed-time display: minutes, not seconds; `Xh Ym` past 60 minutes**
  (2026-09-07) — for the RUN-TOTAL counters only, never per-phase. The
  console dashboard's header (`ConsoleUI._fmt_elapsed_minutes`) and the live
  browser view's cumulative "`N/M phases done · ... of work`" line
  (`flowElapsedMinutes` in `flow_visualization.py`'s `_LIVE_HTML` template)
  both switched from `H:MM:SS`/raw-seconds to this coarser format. Every
  PER-PHASE duration (the running-phase spinner, the interpretability
  sub-step timer, each node's own `fmtDur` in both the live and the static
  post-run diagram) deliberately kept the original fine-grained
  seconds/milliseconds formatting — most phases finish in single-digit
  seconds, and a "0m" readout for the currently-running phase would hide
  whether it is progressing or hung. Two formatters exist
  (`_fmt_elapsed`/`_fmt_elapsed_minutes`, `fmtDur`/`fmtElapsedMinutes`)
  specifically so the coarser one is never accidentally reused where
  sub-minute resolution matters.

## Downstream analyst dashboard

`src/reporting/analyst_dashboard.py::build_analyst_dashboard` runs in Phase
9b and writes exactly one `artifacts/reports/analyst_dashboard.html` in
stacked or parallel mode. It is best-effort: an HTML failure never blocks
the two Excel OOT deliverables.

**Selection is now the union of both P95 rankings, not whichever model was
the primary export.** Phase 8 already computes one maximum OOT score per
entity for both detectors. Phase 9b converts each distribution to percentile
ranks and partitions the P95 union into three disjoint tabs:

- **Solo IF:** IF ≥ P95 and IF+VAE < P95.
- **Solo IF+VAE:** IF+VAE ≥ P95 and IF < P95.
- **Intersección:** both scores ≥ P95.

Each row shows both percentiles. Top-variable explanations and monthly
recurrence are kept per detector: `months_present_by_entity` runs once for
IF and once for IF+VAE against each detector's own P95 cut-off. Search only
filters the active tab; the KPIs update with that same partition.

**Full-history download and case workflow.** `main.py` passes the complete raw
panel into the dashboard; only entities in the P95 union are embedded, but all
their available periods and original columns are retained. The UTF-8 profile
download is therefore a complete entity history, not only OOT. Each case has
exactly three operational states (`Sin revisión`, `En revisión`, `Cerrado`),
persisted in browser storage with an ISO change timestamp. “Casos revisados”
shows only the two non-default states and exports their ID, configured identity
field (default `puesto`), status, date and time. `puesto` is shown immediately
below the ID in the profile; `--analyst-identity-column` changes the source
column without changing model inputs.

Detector explanations are sourced from the complete, de-duplicated explained
OOT frame, not the filtered P90 export. This matters for a VAE-only P95 entity
whose IF percentile is below P90: its IF variables still exist and must render.

The dashboard remains self-contained and client-side: no API, server, or
external business taxonomy is introduced. Entity IDs and variable labels are
escaped before HTML insertion, and detail chips are populated with
`textContent`.

## Post-training sensitivity and data-quality validation

Phase 9d (`src/evaluation/sensitivity.py`) is on by default and never refits a
model. It reuses the fitted preprocessor, Isolation Forest, VAE and calibrated
thresholds. Raw-variable ablation uses a neutral value learned only from the
training window; explicit null and numeric/boolean zero replacement are
separate scenarios. Pairwise combinations are generated from the six
highest-leverage variables by default, followed by random 10/25/50/75/90%
information-loss scenarios and a cumulative least-impact-first pruning path.

Stability is defined objectively as rank correlation ≥0.98, alert-flip rate
≤5%, and post-label PR-AUC/F1 deterioration ≤2 percentage points. Labels are
optional and, when present, enter only this post-training evaluation. Zeros,
false booleans, empty strings and nulls all count as missing information for
the record-quality analysis. Records at or above 90% are compared with the
rest using score CDFs, a KS test, alert rates and performance before/after
exclusion, producing one of three recommendations: retain in train/test,
retain only for test/monitoring, or exclude from future evaluations.

Artifacts under `artifacts/reports/`: `sensitivity_analysis.html`,
`sensitivity_analysis.xlsx`, scenario/variable/matrix/high-zero CSVs and
`sensitivity_summary.json`. The main HTML/Markdown report links them and shows
the minimum stable variable count plus the ≥90% record recommendation.

## IF-VAE Diagnostic Suite integration

**What it is.** A vendored standalone package (`tools/
if_vae_diagnostic_suite/`, v1.0.0, own `pyproject.toml`/CLI/tests/AGENTS.md)
that diagnoses *why* Isolation Forest and IF+VAE disagree. Phase 9c validates
that `ifvae_diag` is importable and, when needed, runs an editable install
from this repository's own vendored path. It never resolves the package name
from an index. After `pip install -e`, it adds the vendored `src/` directory
to `sys.path` so the newly installed package is available in the **same
process** (invalidating import caches alone does not reprocess `.pth` files).
`--no-auto-install-suite` disables the side effect and reports the missing
suite explicitly; no manual pip step is required in the default path.

**How to run it against this project:**
```
py tools/export_diagnostic_suite_inputs.py --out tools/if_vae_diagnostic_suite/build/modelo_run
cd tools/if_vae_diagnostic_suite
PYTHONPATH=src py -m ifvae_diag run --reference build/modelo_run/reference.csv \
  --scored build/modelo_run/scored.csv --config build/modelo_run/modelo_config.yaml \
  --out build/modelo_run/report
```
`export_diagnostic_suite_inputs.py` mirrors `main.py`'s own Phases 2→3a→4→6→
6b→7 (same functions, same seed, stacking on) to fit a real, fresh IF+VAE
pair and export the suite's exact contract: `reference.csv` = train block
(no labels — not required there), `scored.csv` = the true OOT block
(`n_oot_periods=3`), one row per **(entity, period)** — deliberately *not*
deduplicated to one row per entity the way the OOT Excel is, since the
suite has no concept of "each entity's best month" and more rows give its
drift/family diagnostics more to work with. Ground truth
(`load_ground_truth_labels`/`_types`) is attached to `scored.csv` for this
integration only, from this project's own synthetic labels — see
"Label-free mode" below for why a real production run cannot do this.

**Two real, root-caused fixes made to the vendored copy** (both covered by
new tests, full `make chore-lint` green — `31 passed`, `compile=True
tests=True mutations=True`):

- **`scripts/mutation_probe.py` was silently broken on Windows.** It built
  the mutated-copy `PYTHONPATH` with a hardcoded POSIX `:` separator
  (`f"{source_root}:{ROOT}"`). On Windows, `os.pathsep` is `;`, and a
  drive-letter path (`C:\Users\...`) already contains a colon, so the join
  produced one unparseable string; Python silently fell back to importing
  the real, pip-installed (unmutated) package for every mutant, and
  `_mutant_is_killed` always returned `False` — a false "the gate is
  broken" reading with nothing to do with test quality. Fixed with
  `os.pathsep.join(...)`; all 4 mutants (`percentile_tie_direction`,
  `if_quadrant_threshold_exclusive`, `vae_quadrant_threshold_exclusive`,
  `lift_formula`) are now genuinely killed. Regression-tested
  (`tests/test_mutation_probe.py`) by actually running one mutant end to
  end and asserting it is caught — not a tautology.
- **No label-free mode existed.** The data contract hard-required a binary
  `label_col` in `scored.csv` (`contracts.py::_require_columns`), and every
  supervised computation (`metrics.py`, autopsies, coverage,
  orientation-risk warnings) read `config.label_col` unconditionally. This
  project's **official, real-data runs are unsupervised and carry no
  target at all** — a real production run could not produce a valid
  `scored.csv` for this tool as shipped. Added `label_col: str | None`
  (`config.py`); `contracts.py`/`pipeline.py` skip every label-dependent
  computation (`metrics.csv`, `autopsies.csv`, `coverage.json`,
  `metrics_by_group.csv`, the two orientation-risk warnings) when it is
  `None`, while percentiles, quadrants, latent diagnostics, and drift —
  the genuinely label-free half of the suite — still run and still write
  `report.md`/`disagreement.png`. Covered by
  `tests/test_pipeline_unsupervised.py` (asserts both the label-free
  degrade *and* that supplying a real label column still enforces the
  existing binary-label validation — the fix must not weaken the
  supervised path). Verified against this project's own real export in
  both modes: identical quadrant counts (77/115/132/1176) with and without
  labels, proving quadrant assignment is genuinely label-independent.

**Report enhancement: per-layer performance chart.** `report.md` originally
carried only `disagreement.png` (a percentile-agreement scatter — shows
where IF and VAE *agree*, not which one *performs*). Added
`metrics_bar_plot()` (`reporting.py`) — a precision@k bar chart grouped by
alert budget (k=10/25/50), one bar per score candidate
(`if_percentile`/`vae_percentile`/`ensemble_max`/`ensemble_mean`, fixed
colors so a candidate is visually stable across runs) — as the visual
counterpart to `metrics.csv`, so monitoring each detection layer's
operational performance doesn't require reading a table. TDD: wrote
`tests/test_reporting_metrics_plot.py` first (Red —
`ImportError: cannot import name 'metrics_bar_plot'`), implemented, both
tests green. Wired into `pipeline.py::_write_outputs`, which now also
passes a `has_metrics_plot` flag into `write_markdown_report` so the
markdown embeds `![...](metrics.png)` when the chart exists and a plain
sentence explaining its absence when it does not — this chart is
inherently label-dependent (precision/recall need known positives), so it
is correctly skipped, not broken, on a real unsupervised run
(`report_unsupervised/`: no `metrics.png` written, `report.md` reads "no
label column (label-free mode)"). Verified against this project's real
export in both modes: labeled run's `report/` has `metrics.png` embedded
after "Operational metrics"; label-free run's `report_unsupervised/`
correctly has neither the file nor a broken image reference.

**What still does NOT apply to an official (unsupervised, real-data) run,
by design of the underlying method** — the label-free mode above makes
these *not crash*, not makes them meaningful:
- `metrics.csv`/`coverage.json`/`metrics_by_group.csv`
  (Precision@K/Recall@K/Lift@K/AP, unique/shared coverage, segment/family
  cohorts) — need known positives to mean anything.
- `autopsies.csv` (known-positive feature autopsies) — selects rows by
  `label_col == 1`; nothing to select without one.
- The two `*_percentile_orientation` warnings — need both classes present.
- `if_stability.json` — unrelated to labels, but unavailable whenever
  `if_score_col` is set (diagnosing the *production* forest), regardless
  of label mode; see "No cross-seed stability measurement" below.

**What DOES apply and was validated against this project's real, freshly-
fitted IF+VAE (2026-09-03, synthetic labels used only to prove the numbers
line up, not as a production measurement):**
- `scored_diagnostics.csv` (percentiles, `if_percentile`/`vae_percentile`,
  `ensemble_max`/`ensemble_mean`, disagreement quadrant per row).
- `drift.csv` (KS/Wasserstein/out-of-range population shift, reference vs.
  scored).
- Latent diagnostics (`summary.json::latent_diagnostics`) — active units,
  collapsed fraction, per-unit KL.
- `report.md` + `disagreement.png`.

**Findings from that run** (500 individuals × 13 months, `--quick`-scale,
20 VAE epochs, stacking on — a smoke-scale run, not a production
measurement; treat magnitudes as directional):
- **VAE dominates IF on `global` anomalies** (AP 0.82 vs. 0.08, recall@10 ≈
  89% vs. 11%) but **both are near-random on `local` and `contextual`**
  (AP ≈ 0.01–0.02, recall@10 = 0% for almost every score/ensemble
  combination) — see "Known open problems" below, now with an independent,
  differently-coded confirmation.
- **IF contributed zero unique hits to the top-10/25 queue that VAE did
  not already find** (`coverage.json`: `a_only_positive_hits: 0` at every
  budget tried) — on this run, a simple mean ensemble was *worse* than VAE
  alone (AP 0.24 vs. 0.28), exactly the risk the suite's own README warns
  about ("do not use `IF AND VAE` as the default... compare against
  IF-only, VAE-only, max, mean").
- **No posterior collapse** (`collapsed_fraction: 0.0`, 8/8 active units)
  — an independent, external confirmation that the 2026-08-22/23
  loss-scaling fix (see "Known open problems") is holding.
- **The drift table's top entries are dominated by calendar/lag-feature
  artifacts, not genuine concerning drift**: `cyc__period_month_sin/cos`
  and every `*_lag3`/`*_diff3`/`*_ratio3` panel feature show KS ≈ 0.37–1.0
  simply because `reference` (train months) and `scored` (strictly later
  OOT months) cover different calendar months by construction — any
  chronological split will "drift" on month-of-year. Read this table with
  calendar-derived and panel-lag features filtered out, or expect them to
  dominate meaninglessly.
- **The `collective` anomaly family had zero known positives in this
  particular 3-month OOT window** (`metrics_by_group.csv` only has
  `local`/`contextual`/`global` rows) — with only ~33 total positives
  spread over one 3-month slice, a family-level breakdown can miss an
  entire family by chance. A longer or repeated OOT window would be needed
  before reading "family X has 0 recall" as evidence rather than absence.

**In-process integration + report chapter (2026-09-04, superseded below).**
Runs the suite itself, in-process, as an optional phase (`--run-diagnostic-
suite`, then default off), building `reference`/`scored` directly from this
run's own already-fitted detectors and indexing the result into
`anomaly_report.{html,md}` as a 15-section, non-interpretive chapter. The
architecture (contract → gathering → renderers, three modules) and the
label-free/traceable-field/no-silent-fallback design all still stand; what
changed on 2026-09-05 is the section count, two previously-permanent
`UNAVAILABLE` diagnostics, and the addition of a second, explicitly
interpretive chapter — see immediately below.

**Restructured into a leaner ficha + a separate interpretation chapter
(2026-09-05).** Two requests arrived back to back and both stand: keep the
factual ficha free of interpretation (2026-09-04's own explicit spec), *and*
add a per-analysis interpretation toolkit, indicator validation, a dynamic
decision-flow diagram, and an analyst recommendation (2026-09-05's request).
The resolution is two contracts, not one weakened contract:

- `--run-diagnostic-suite` is **ON by default**. `ensure_suite_installed`
  removes the former manual-install prerequisite; `--no-auto-install-suite`
  validates without installing and `--no-run-diagnostic-suite` skips the
  whole phase. Stability refits still carry the runtime cost described below.
- **Six sections removed from the factual ficha**, by explicit editorial
  request, not because they could not run: disponibilidad de diagnósticos,
  unidad de análisis (folded into a note on `§1 Alcance` and the agreement
  table instead of its own section), reconstrucción/autopsias de alertas,
  calidad/desplazamiento de datos (raw table), bloques condicionados a
  verdad base, riesgos/limitaciones/procedencia. The surviving nine are
  renumbered 1–9: alcance, configuración efectiva, concordancia y
  desacuerdo, candidatos VAE, sensibilidad, latente, estabilidad,
  temporal/segmentación, experimentos. `CONTRACT_VERSION` bumped to
  `"2.0.0"` for the shape change. Drift and autopsy numbers were not thrown
  away — they still feed the interpretation chapter as distilled signals
  (see below), just not as raw per-row tables in the ficha.
- **Two previously-permanent `UNAVAILABLE` diagnostics now actually run:**
  - *Estabilidad IF* was always `UNAVAILABLE` because the "production" IF
    score is precomputed (`if_score_col` set), so the suite's own
    `top_k_stability` never had a fresh multi-seed fit to measure. Fixed by
    refitting `IsolationForestDetector` (this project's own class, same
    hyperparameters read off the fitted production detector's own public
    attributes — `n_estimators`, `contamination`, etc., not off a possibly-
    partial `best_params` dict) `diagnostic_stability_refits` times
    (default 3) with different seeds and running the SAME
    `ifvae_diag.stability.top_k_stability` the suite already uses for IF —
    not a new metric, the identical one, applied to a detector the suite
    itself does not refit in this integration.
  - *Estabilidad VAE* did not exist at all (the suite has no VAE-refit
    path). Implemented the same way: refit `VAEDetector` (this project's
    class) `diagnostic_stability_refits` times, same architecture read off
    the fitted instance's own attributes (`latent_dim`, `hidden_dim`, …),
    `top_k_stability` over the refits' scores on the OOT population.
    **Trade-off, stated rather than absorbed:** VAE refits are full
    training runs, not just scoring — this is the single most expensive
    part of Phase 9c (roughly `diagnostic_stability_refits` extra VAE
    fits). `--diagnostic-stability-refits 0` disables it (`UNAVAILABLE`
    with a stated reason) if that cost is not acceptable. Seeds are derived
    from `base_seed` (`config.seed + 1000·i`), not hardcoded, so two
    official runs never collide and the choice doesn't need defending as a
    "random" magic number.
  - **Methodological note, grounded before implementing (not asserted from
    memory):** searched for prior evidence on cross-seed stability of
    autoencoder-family scores before picking a pass/fail threshold, and
    found none that would justify one — "Evaluating the Stability of Deep
    Learning Latent Feature Spaces" (arXiv:2402.11404, 2024) reports
    Jaccard dissimilarity commonly *exceeding* 0.6 (mode ≈0.86) between
    independently-trained autoencoder embeddings, i.e. VAE-family models
    are, in the published record, considerably less stable across seeds
    than tree ensembles by default. **Deliberately did NOT invent a
    universal "stable ≥ X" cutoff** for this reason — an expert asked to
    defend an arbitrary Jaccard threshold for a VAE would reject it as
    unsupported by the evidence. Only the genuinely degenerate case
    (mean Jaccard < 0.05, essentially zero overlap regardless of
    architecture) is flagged; everything else is reported numerically with
    the citation, comparatively (IF vs. VAE), never as a verdict.
- **Segmentation executes with a configurable source column.** The default is
  `--diagnostic-segment-column segment`; point it at any raw categorical
  column (for example `--diagnostic-segment-column region`). An empty string
  disables the breakdown. A configured name that is absent logs a warning and
  makes §8 `NOT_APPLICABLE`, never a silent no-op.
- **Sensitivity grid and entity view are now ON by default** —
  `diagnostic_sensitivity_grid` defaults to `(0.90, 0.95, 0.99)` (this
  project's own P90/P95/P99 operating points, not an arbitrary choice) and
  `diagnostic_entity_view` defaults to `True` — both previously sat at
  `NOT_REQUESTED`/off purely because they were opt-in, not because they
  couldn't run.
- **`§9 Experimentos diagnósticos` executes what the current run can support.**
  Ensembles máximo/promedio and reconstruction variants reuse scores already
  computed, so they always execute. IF contamination sweeps real refits over
  `0.01 0.02 0.05` by default and can be replaced with
  `--diagnostic-experiment-contamination-grid`. Capacity and beta each require
  a full VAE refit per point, so they are opt-in through
  `--diagnostic-experiment-capacity-grid` and
  `--diagnostic-experiment-beta-grid`. The five families that require model/
  preprocessing code changes or multiple retained temporal windows remain
  `NOT_REQUESTED`, each with its own reason rather than a generic placeholder.

**New: a second, explicitly-labelled interpretation chapter.** Reconciles
the standing "no interpretation in the ficha" rule with the new request for
a toolkit, a decision flow, and a recommendation — by building them as a
SEPARATE contract from the same numbers, not by softening the first one.

- `src/evaluation/ifvae_interpretation.py` — `build_interpretation_contract(
  ...)` returns `{toolkit, indicator_validation, decision_flow,
  methodology_notes}`. Every claim carries a `basis` (what it was computed
  from) and a `severity` (`info`/`attention`/`caution`, never a pass/fail
  verdict on the run). `METHODOLOGY_NOTES` documents every threshold used —
  including the ones deliberately NOT turned into a threshold (see
  stability above) — so a reader can check no cutoff is asserted without a
  citation or an existing project precedent behind it.
- **Validación de indicadores**: sample-size adequacy for Spearman/Jaccard
  (flags n < 30), refit-count adequacy for stability, sensitivity-grid
  range validity, and — the concrete statistical upgrade this pass made —
  **Benjamini-Hochberg FDR correction** (Benjamini & Hochberg, 1995) across
  every feature's KS p-value for the drift signal, replacing the earlier,
  explicitly-flagged gap ("no se declaró una regla de severidad") with a
  principled one: testing dozens of features simultaneously at an
  uncorrected α=0.05 produces several false "drifted" features by
  construction.
- **Flujo de decisión**: 5 nodes evaluated against THIS run's real numbers
  (¿hay observaciones en BOTH? ¿espacio latente activo (≥⅓, Burda et al.
  2016 — the SAME threshold this project's own
  `src.models.vae.collapse_verdict` already uses, reused not reinvented)?
  ¿drift FDR-significativo en variables de negocio (calendario/panel
  excluidos por el motivo estructural ya documentado)? ¿sensibilidad al
  umbral ≥3× entre el mínimo y el máximo de la malla?), each producing a
  fragment; the **recommendation is assembled from whichever fragments this
  run's branches actually produced**, sorted attention-first — never a
  fixed string per scenario.
- `src/reporting/interpretation_section.py` — HTML/Markdown renderers,
  mirroring `diagnostic_section.py`'s split exactly, with one deliberate
  difference asserted by its own test: this renderer IS allowed
  interpretive language (that is its entire purpose), but like the factual
  renderer it still computes nothing itself — every fragment, check, and
  node arrives pre-built.

**Unit of analysis is stated everywhere** (moved from its own section into
a note on `§1`/the agreement table): counts are entity–period observations,
never "individuos"/"clientes". The chapter's `BOTH` count differs from the
in-house "Concordancia entre detectores" chart (`report_content.py`), which
deduplicates to one row per entity and ranks against the OOT population
itself, while this chapter keeps one row per (entity, month) and ranks
against the training distribution — the interpretation chapter's own
methodology notes make this explicit where it matters.

**Tests.** `tests/test_diagnostic_section.py` plus the focused dashboard
contract tests drive the real contract path via
`diagnose_frames`/`build_interpretation_contract` — HTML/Markdown parity for
BOTH contracts, traceability, no silent fallbacks, response to
threshold/candidate/aggregation changes, architecture modes, label states,
quadrant arithmetic, absent/empty artifacts, degenerate cases (zero alerts,
ties, missing `logvar`, duplicate ids, infinities, missing values, temporal
overlap, segmentation), decision-flow branch selection under constructed
scenarios (zero-BOTH, collapsed-latent, business-vs-calendar drift), and —
the part that used to be untestable — **real seeded stability refits**,
fitting actual (tiny) `IsolationForestDetector`/`VAEDetector` instances and
asserting a valid Jaccard in `[0, 1]`. Renderer-purity tests assert the
factual renderer still has no verdict language/run-specific constants and
that neither renderer computes over the run's numbers.

**Verified end to end**, `python main.py --quick --no-tune` (diagnostic
suite on by default now), both `--stack-iforest-into-vae` and
`--no-stack-iforest-into-vae`: zero failed health checks in either mode
(the total count itself is not a fixed invariant — it varies a little
run to run with how many distinct checks a given `--quick` synthetic
sample happens to exercise; 0 failures is the thing that must always
hold), real stability Jaccard values computed (IF ≈0.80, VAE =1.00 on this
`--quick`-scale run —
a real number, not asserted as "good" or "bad" anywhere in the ficha), real
segment table populated, sensitivity grid swept by default, decision flow
and recommendation rendered with this run's own numbers in both formats,
zero unclosed HTML tags. `tools/render_diagnostic_example.py` still renders
both chapters from the same synthetic fixture for review without a pipeline
run.

## Known open problems

- **`local`-type anomalies are unrecovered — and, independently confirmed
  2026-09-03, so is `contextual`.** The Isolation Forest ranks a `local`
  anomaly at roughly the population median (recall@10% ≈ 0 across every
  numeric transform tried), because by construction a `local` anomaly sits
  inside the population's normal band and is anomalous only against the
  entity's own history — `_own_z` is the intended instrument and is
  evidently not sufficient on its own. See `docs/models_isolation_forest.md`
  §"Measured" and `CHANGELOG.md` 2026-08-01 for the numbers. The external
  IF-VAE Diagnostic Suite (see "IF-VAE Diagnostic Suite integration" below)
  independently reproduces this with a differently-coded percentile/AP
  methodology, on real project data: `local` AP ≈ 0.012, `contextual` AP ≈
  0.023 (both near the ~0.007 base rate — indistinguishable from random) for
  **every** score candidate (IF, VAE, both ensembles), while `global` reaches
  AP ≈ 0.82 on the VAE score alone. This needs feature/architecture work, not
  more hyperparameter search.
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
  See `docs/validacion_no_supervisada.md` §6 for the proposed design. Partial
  external option for the Isolation Forest specifically: the IF-VAE
  Diagnostic Suite's own seed-refit top-K Jaccard/selection-probability
  diagnostic (`if_stability.json`) does exactly this — but only when
  `if_score_col` is left unset (a fresh in-suite refit across
  `random_seeds`), which then diagnoses a *different* forest than the
  production one. No equivalent exists for the VAE either way.
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
