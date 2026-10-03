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
├── main.py                 # orchestrator entry point (Phases 2-11, argparse CLI)
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
│   ├── evaluation/         # OOT split, GT join, metrics, thresholds, OOT Excel export,
│   │                       # IF-VAE diagnostic bridge (ifvae_*), post-training sensitivity
│   ├── interpretability/   # SHAP, path length, latent space, per-feature recon
│   ├── reporting/          # HTML/MD report builder, analyst dashboard, flow visualization (no PDF)
│   └── utils/              # paths, logging, observability, console dashboard, assumptions gate, atomic_io
├── tests/                  # pytest suite: diagnostic chapter, analyst dashboard, sensitivity,
│                           # zero-row filter, report incidents, live/phase progress, event
│                           # labels + gate 4.5 + challengers, single config file, IF/VAE tuners,
│                           # §9 experiment families, mixed (embedding) VAE
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

**Tests.** The original suite was removed on 2026-08-22; a new, smaller one has
been rebuilt under `tests/` alongside each feature added since (see
"Conventions → Testing"). Some `docs/*.md` files still name `Test*`/`test_*.py`
files from the *original* suite as the historical source of a claim; those
particular names do not resolve to anything on disk.

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

**Startup dependency check (2026-10-01, installs automatically by default
since 2026-10-02).** `run_pipeline` (`main.py`) calls
`src.utils.dependency_check.check_dependencies` right after `_ensure_dirs()`,
before any phase does real work: parses `requirements.txt` (plain
`name>=X.Y` lines only), compares against what is actually installed via
`importlib.metadata.version`, and on a missing/outdated package installs it
directly via `pip install --upgrade` (`auto_install=True` is the default --
explicit user request: never raise and stop the run over something this
check can fix itself). After `pip` reports success, it re-checks
`importlib.metadata.version` rather than trusting the exit code alone (a
conflicting pin elsewhere in the environment could make the resolver land on
a version that still does not satisfy the floor). `--no-auto-install-deps`
switches to check-only -- log the problem and the exact `pip install`
command, then stop (`SystemExit`) -- for the rare case where mutating the
active Python environment is not wanted at all. `--skip-dependency-check`
(default OFF) disables the check entirely for a pre-vetted environment.
Runs this early specifically because most third-party imports in this
codebase are deferred to inside the functions that need them, not at module
top, so a fix applied here can still take effect for the rest of THIS run.
Tests: `tests/test_dependency_check.py` (14 tests) plus one CLI-wiring test
in `test_config_file.py`.

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
  labels live in the separate file), `ground_truth_path` (or `None` if
  none was located), and `identification_columns` (see below; `()` unless
  `main()` sets it right after loading). Column-name hints (2026-09-27):
  `_TIME_NAME_HINTS`/`_ENTITY_NAME_HINTS` in `src/data/loader.py` also match
  `codmes`/`codclavepartycli` (this project's own real-data naming), on top
  of the generic `period`/`date`/`time` and `entity`/`individual`/`id`. This
  matters for `codmes`: an all-digit period column (`202401`-typed as a
  plain integer, not `object`/datetime) is invisible to the structural
  fallbacks below, so without the name hint it would not be inferred as
  `time_col` at all. Tests: `tests/test_schema_inference.py`. `time_col`/`entity_col` can be
  `None` for arbitrary inputs; callers must handle that.
- **Identification columns (2026-09-27, generalized to a list 2026-10-01)**:
  `data.identification_columns` of `configs/pipeline.yaml` (or
  `--identification-columns`) names panel columns that exist purely to
  identify/describe a record for a human reader (job title, name, area, an
  internal reference number, a segment/grouping column used elsewhere in the
  report, ...) and carry no modelling signal. `main._resolve_identification_columns`
  folds in every field of `dashboard.identity_columns` automatically (none of
  them needs to be listed twice) and sets the result on
  `schema.identification_columns` once, right after `load_or_generate_panel`,
  before anything else reads `schema`. Every phase that decides what the
  model sees calls `src.data.loader.key_columns(schema)` — the *one* place
  that set is assembled — instead of keeping its own column list: feature
  building (`PanelFeatureEngineer`/`build_preprocessing_pipeline`), the VAE's
  categorical sources (`categorical_sources`), the exact-zero-row filter, the
  numeric-transform diagnostic (`infer_numeric_features`) and the
  post-training sensitivity study. A name absent from the loaded panel only
  warns (`config.identification_columns_present`, category `data`) — unlike
  the segment column, nothing downstream breaks, and the same config file is
  meant to run against panels that do not all carry the same optional
  columns. This is a *modelling* exclusion, not a display one in general: the
  OOT Excel "VARIABLES" columns and the raw data profile (and the analyst
  dashboard's full per-entity CSV download, which embeds every raw source
  column verbatim) keep showing all of these columns regardless, since
  identification is exactly what they are for. The ONE place this is
  display-selective is the dashboard's per-case identity summary (the rows
  shown directly under the entity ID when a case is opened): only fields also
  listed in `dashboard.identity_columns` render there — see "Full-history
  download and case workflow" below for that superset relationship.
  `identification_columns` is therefore a superset of `identity_columns`, not
  the same set: a name that belongs ONLY in the former is excluded from
  modelling but never becomes a per-case display row. The vendored IF-VAE
  Diagnostic Suite needs no separate configuration for
  this — its `DiagnosticConfig.features` list is built from the
  already-preprocessed feature names, so an excluded column is already absent
  by the time the suite sees it. Tests: `tests/test_identification_columns.py`
  (the shared `key_columns` helper, the two call sites with no test file of
  their own, and `_existing_identity_columns` -- the dashboard-list trimming
  helper, see below), plus one test in each of `test_config_file.py`
  (resolution/precedence), `test_zero_row_filter.py`, `test_mixed_vae.py`
  (`categorical_sources`) and `test_sensitivity.py`.
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
| 4 Tuning | `tune_*(valid_mask=)` | static temporal holdout. IF: `n_estimators=300` fixed, `max_samples` an **absolute integer** (1 024–32 768) + `max_features`, Sobol sweep with anchors, 1-SE + noise-margin selection against the ψ=256 default, objective = AP vs reviewed labels when gate 4.5 authorises them, else `tail_separation`; contamination not searched. VAE: `-ELBO@β=1`/`recon_p50`, KL annealing + early stopping, fingerprinted study |
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

**Panel depth matters.** The default is **16 periods** (`--quick` uses 13,
`--full` 16) so the chronological split — train 8 / validation 2 / test 3 / OOT
3 by default (`--n-val-periods`, `--n-test-periods`, `--n-oot-periods`) — leaves
a training block deep enough for the `h=6` contrast. With fewer periods
`PanelFeatureEngineer` silently drops the deep horizons — correct, but it means
the feature is not being exercised.

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
  **`max_samples` tuned to a float fraction (Optuna's search space allowed
  0.3-1.0 until 2026-09-24; `max_samples` is now an absolute integer in the
  tuner, so this trigger no longer comes from tuning) combined with a
  multi-month training block**. `max_samples` as a
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
  Every record carries `ts` (whole seconds) **and** `t` (epoch seconds, ms
  precision — what the live view uses to time a running function; files
  written before `t` existed fall back to `ts`). Two more emitters feed the
  same stream for code that is timed elsewhere: `logging_config.
  report_phase_event` (a phase transition reported by a vendored package — the
  diagnostic suite's steps) and `observability.progress_event` (one
  progress-bar update: `desc`, `n`, `total`, `unit`, `state` =
  `start|update|end`, `current`, `elapsed_s`). Phase, check and progress
  observers are best-effort callback lists; a raising observer is dropped.
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
- **Isolation Forest split-count analysis (2026-09-28)**:
  `src.interpretability.split_count_analysis` — own analysis, not a published
  metric. For each row of the OOT alert queue (the calibrated-threshold
  flagged rows, not every row) and each tree, walks the real decision path
  (`tree.decision_path`, translating the tree's own feature subset back to
  the original index via `estimators_features_` when `max_features < 1`) and
  measures, per feature, the mean TOTAL path length of the isolations it
  participated in. Low = the variable is usually involved in fast, clean
  isolations (clear signal); high = it only shows up in long, many-split
  isolations (weak/noisy signal). Two report charts: top-10 fewest cuts, top
  10 most cuts (requested so "the model only carries clear signals"). Wired
  in `main.py` Phase 10 alongside SHAP/path-length, via
  `main._flagged_rows_for_split_count`: normally the calibrated-threshold OOT
  alert set, but a strict business threshold (this project's default POT
  calibration targets a 0.1% false-alarm rate) can legitimately flag zero
  rows in a small OOT window — confirmed live on a `--quick` run — so a
  below-`min_rows` (default 10) flagged set falls back to the top OOT rows by
  score (default 50) instead of leaving the chart empty; logged either way.
  Tests: `tests/test_iforest_split_counts.py` (13 tests total: the analysis
  itself + the fallback helper, mutation-checked).
- **Removal recommendation from the split-count analysis (2026-09-29)**:
  explicit user request, built on the block above. `main.py` now also stores,
  in the same `chart_data.static.iforest_splits` payload, `n_features_total`
  and `unused_features` — model features present in `X_model`/`names_model`
  that never appear in `split_result["per_feature"]` at all, i.e. never
  helped isolate a single flagged OOT row this run (a stronger signal than
  merely needing many cuts when a feature *does* participate). Only computed
  when the analysis actually ran (`top_clear`/`top_noisy` non-empty) — an
  empty `per_feature` from zero analysed rows would otherwise look like every
  feature is unused, which is a data-availability artefact, not a signal
  about the features. New report section (MD `_iforest_splits_section_md` /
  HTML `_iforest_splits_section_html` in `src/reporting/report.py`, right
  after the existing split-count charts): a table naming the 0%-usage
  features as "Candidata a eliminar" and the `top_noisy` features as the
  weaker "Candidata a revisar (señal débil)" — the two tiers are worded
  differently on purpose (mutation-checked: swapping the wording makes
  `test_noisiest_used_variables_are_flagged_to_review_not_remove` and
  `test_no_unused_variables_states_that_explicitly` fail). Always framed as a
  suggestion pending cross-checks (SHAP importance, multi-seed stability),
  never an automatic action, and explicit when no 0%-usage feature exists (no
  blanket removal claim from this signal alone). Tests:
  `IForestSplitsRecommendationSectionTests` in `tests/test_report_incidents.py`
  (6 tests).
- **Per-seed stability, visual (2026-09-28)**: `_seeded_refit_stability`
  (`src/evaluation/ifvae_diagnostic.py`) now exposes the full seed×seed
  pairwise-Jaccard matrix and each seed's mean against the rest (the
  aggregate mean/min already shown were always computed over this same
  matrix; it was just discarded after averaging). A fact table per seed-pair
  in the diagnostic contract §7, and a Plotly heatmap per detector
  (`chart_static["stability_seeds"]`) in the report, so a reader can see
  *which* seed disagrees, not only the aggregate number. The interpretation
  chapter also gained `METHODOLOGY_NOTES["jaccard-causes-and-reference-band"]`
  (why the Jaccard mean is not 1.0 — stochastic subsampling by design, a
  finite top-K amplifying boundary ties, no ground truth to average noise
  against — plus a literature-grounded reading band: <0.4 investigate,
  0.4–0.6 expected for a small stochastic top-K ensemble, ≥0.6 comfortable;
  Kuncheva 2007, arXiv:2402.11404 already cited by this project). This is
  additive, not a replacement — the existing "no universal cutoff" stance
  (`stability-no-universal-cutoff`) and the degenerate-Jaccard hard flag
  (`DEGENERATE_JACCARD`) are unchanged.
- **Row-filter section, explicit (2026-09-28)**: the exact-zero-row filter's
  report section now leads with one prose sentence stating the count AND the
  percentage removed, not only a table cell.
- **Stability refits default raised 3 -> 5 (2026-10-01)**: explicit user
  request. `PipelineConfig.diagnostic_stability_refits` (`main.py`), its
  `--diagnostic-stability-refits` CLI help text, the commented example in
  `configs/pipeline.yaml`, and `run_ifvae_diagnostic_suite`'s own parameter
  default (`src/evaluation/ifvae_diagnostic.py`) all moved from 3 to 5. Still
  configurable per run (`diagnostic.stability_refits` in the config file, or
  the CLI flag); 0 still disables the section entirely (UNAVAILABLE, stated
  reason), and 2 is still the hard minimum `top_k_stability` needs. Five
  seeds gives 10 pairwise comparisons for the per-seed Jaccard table/heatmap
  above instead of 3 — a more robust visual read — at the cost of two more
  full VAE refits per run (this is the single most expensive part of the
  diagnostic chapter; see the trade-off note on the field itself). No test
  asserted the old default value, so none needed updating; tests that
  exercise `stability_refits` do so with an explicit value of their own
  regardless of the project default.
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
  forces it off): a fixed phase checklist (one row per `_PHASE_PLAN` entry,
  20 today), an "Supuestos (IF/VAE)" panel fed by genuine assumption/gate
  `observability.check(...)` calls, a dedicated interpretability sub-step line
  under the current-phase readout (see "Live view: interpretability's
  checkpoints..." above), a **"↳ función" line** (the nested functions
  currently running, innermost highlighted, each with its own elapsed time),
  **one tqdm-style bar per running loop-shaped test**, and a team-health line
  (RAM/CPU, always with the number visible, never color alone). Keys: `v`
  detail, `o` open the web view, `p` pause. "n/total fases" counts only
  top-level `Phase N` entries — nested functions reported through the same
  observer are detail, not progress through the plan.
- **Progress in every phase (2026-09-23).** `src/utils/progress.py` (`Bar`,
  `track`, `show_best_trial`) replaces every bare `tqdm` in `src/`: iterate it or
  drive it with `update()` / `set_postfix()`, and it publishes `start|update|end`
  `progress` events (throttled to one update per 0.5 s) besides drawing tqdm
  only when `console_ui.is_live()` is false. Never call `tqdm` directly again --
  a bare bar tears the dashboard and never reaches the flow pages. Bars now
  cover: Phase 2 collective-anomaly groups; Phase 4 `transform_diagnostics[
  features]` / `transform_plots[features]`; Phases 6/7 `optuna[<study>]` (best
  value as postfix) and `vae[epochs]` (loss as postfix); Phase 8
  `rank_stability[bootstraps]`; Phases 9/10 `explain_rows_vae[batches]`,
  `vae_recon_by_feature[batches]`, `permutation_importance`; Phase 9d
  `sensitivity[single-variable|variable-pairs|information-loss|pruning-path]`;
  Phase 11 `build_report[formats]`. Slow calls without a loop are named
  `log_phase` steps instead: `evaluation.silhouette_score`,
  `evaluation.calinski_harabasz_score`, `evaluation.rank_stability`,
  `evaluation.supervised_metrics`, `evaluation.reduce_2d[<method>]` (UMAP),
  `interpretability.explain_rows_*`, `sensitivity.high_zero_analysis` /
  `write_tables` / `write_workbook` / `write_html_report`, and
  `reporting.build_markdown` / `build_html` / `build_model_documentation`.
  In the flow pages **each phase's card carries the bars that ran inside it**
  (`node["progress"]`, attributed by file order like nested functions): the
  live page draws a mini bar per bar on the card, the static post-run HTML
  lists them (n/total, time, last item) in the node's detail panel.
- **Live progress of the diagnostic suite (2026-09-23).** Three consumers of
  one event source, see "IF-VAE Diagnostic Suite integration → Live progress":
  the console dashboard lines above, the live browser view (`/state` now
  returns `live.running_functions`, `live.progress`, `live.recent_functions`
  and `server_now`; the page's "Function running / Progress of running tests /
  Last finished functions" panels tick every 250 ms from the last poll, so a
  timer keeps growing while nothing moves — the stall signal), and real tqdm
  bars on stderr **only when no live dashboard owns the terminal**
  (`console_ui.is_live()`). `flow_visualization._build_nodes` stays a pure
  function of the event file; the clock-dependent elapsed fields are added
  only by `_annotate_live` for `/state`. Once the run has ended, leftover
  running functions/bars are cleared instead of shown as still going.
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
shows only the two non-default states and exports their ID, configured
identity field(s), status, date and time (see below for how many and which
ones).

**Several identity fields, not just one (generalized 2026-10-01).** The
identity field shown immediately below the ID in the profile was generalized
from one hardcoded column (default `puesto`) to a LIST,
`dashboard.identity_columns` in
`configs/pipeline.yaml` (`--analyst-identity-columns` on the CLI, `nargs="*"`)
— one row per configured field, in order, explicit user request ("pueda ver
varios campos adicionales"). Each field is folded into
`data.identification_columns` automatically (see "Identification columns"
above), so none of them ever reaches feature building. For each field
independently, if its value varies across periods for the same entity, the
profile shows the value from the **latest** period that has a non-empty one
(`analyst_dashboard._identities_for`, walks periods newest-first per field
and returns the first non-blank value for THAT field) — never the first, a
blank period never masks an earlier real value, and one field being blank in
the latest period never affects another field's own latest value. Rendered
as `profiles[key]["identities"]`, a list of `{"label", "value"}` pairs,
index-aligned with a matching list of static `<div class="midentity">` rows
built from `identity_columns` at render time (`id="mIdentity_<i>"`); the CSV
export (`exportReviewedCases`) gained one column per configured field
instead of a single "identity" column. Test:
`test_case_workflow_identity_and_reviewed_export_are_present`, plus
`MultipleIdentityColumnsTests` (two fields rendering independently, and an
empty list rendering no identity row at all) in `tests/test_analyst_dashboard.py`.

**A configured field absent from the real panel never breaks the dashboard
(2026-10-01).** Explicit user request: `main.py` trims
`config.analyst_identity_columns` to the fields that survived
`_resolve_identification_columns` (i.e. actually exist in this panel) right
after resolving it (`main._existing_identity_columns`, order preserved)
*before* calling `build_analyst_dashboard` — a configured display field
absent from the real panel is silently dropped, never rendered as a
permanent "No disponible" row. `data.identification_columns` (the broader
list) already got this same tolerant treatment inside
`_resolve_identification_columns` itself (a missing name there only warns,
category `data`, via `config.identification_columns_present`). Tests:
`ExistingIdentityColumnsTests` in `tests/test_identification_columns.py`.

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

## Reviewed event labels, gate 4.5 and challengers (Phase 8c)

`data.csv` has **no target column**. A separate table gives `(entity_id, codmes)`
rows a reviewed target, and Phase 8c (`src/evaluation/event_supervision.py`)
uses it **after** IF and VAE are fitted and scored, so labels can never touch
training. The design comes from `reports/Supervisión temporal por eventos.md`
(a referential gate, not a model): supervision is conditioned on evidence
counted in **mature, independent positive episodes**, not on a row rate.

**Where the table lives.** Any CSV/parquet with at least `entity_id`, `codmes`
(`YYYYMM`) and `target` (`0` confirmed normal, `1` confirmed anomaly, empty = no
decision). Default folder `data/reviewed_labels/` (input, outside `artifacts/`,
gitignored, never created by the pipeline; `current/reviewed_labels.csv` wins,
else the newest table; `quarantine/`, `fixtures/`, `schema/` are skipped);
`--labels-path FILE` / `--labels-dir DIR` override it. Column names are
auto-detected (`entity_id|id_entidad|...`, `codmes|period|periodo|mes`,
`target|label|is_anomaly|y`) or mapped with `--labels-entity-column`,
`--labels-period-column`, `--labels-target-column`, `--labels-status-column`.
Optional columns: `label_status` (only `confirmed`/`adjudicated` count; override
with `--labels-usable-statuses`; `pending`, `uncertain`, `conflict`,
`superseded`, `withdrawn` never become negatives), `episode_id`,
`maturity_date`, `label_available_at` (rows available after the cut-off are
excluded).

**Never invented negatives.** A panel row the file does not mention is *unknown*,
not 0; so are null/invalid targets, excluded statuses and conflicting duplicates
(same entity+month, different target -> dropped and counted).
`--labels-unlisted-as-negative` is an explicit opt-in for an exhaustive base and
the gate **vetoes** it unless `--labels-audit-attested`. Positive months are
collapsed into **episodes** (`episode_id` if the file has it, else consecutive
positive months of one entity; `--label-washout-months` is how many non-positive
months close one). A row is usable only if **mature**: positives when their
episode's end + horizon (`--label-horizon-months`, default 1) +
`--label-confirm-delay-months` + `--label-maturity-buffer-months` has elapsed by
the cut-off (`--labels-as-of`, default last panel month); negatives when their own
horizon has elapsed.

**Gate 4.5 (`label_gate.py`).** Level by mature positive episodes, plus evidence
requirements; any veto forces `rojo`:

| Level | Episodes | Also required | Authorised |
| --- | --- | --- | --- |
| `rojo` | `<30` (or `<10` positive entities) | -- | IF/VAE only |
| `ambar_1` | 30-99 | >=10 entities, >=2 temporal origins, no month with >50% of episodes | + penalised logistic, discrete hazard, head over frozen scores |
| `ambar_2` | 100-199 | >=20 entities, >=3 origins, >=50 OOS positive episodes | + restricted boosting (authorised, **not implemented**) |
| `verde_condicionado` | >=200 | >=30 entities, >=3 origins, >=50 OOS, `--labels-formal-calc-ok` | + balanced ensembles (not implemented) |

The cut-offs are the document's internal governance heuristics
(`GateThresholds`), not published thresholds. Vetoes: `no_usable_labels`,
`no_confirmed_negatives`, `invalid_target_values`, `label_conflicts`,
`unreviewed_converted_to_negative`, `episodes_not_collapsed` (positives without
`episode_id` in a file that has the column). Unknowable fields (audited
non-alerts, regimes) are reported `None`, never made up. A file with 40 mature
episodes but a single temporal origin is still `rojo`; `unmet_for_next_level`
says exactly what is missing.

**Evaluation (`event_evaluation.py`).** Frozen IF and VAE scores against the
labels on the test and OOT windows, for two targets: `current_month` and
`onset_within_<H>m` (a NEW episode starts within `--hazard-horizon-months`;
rows inside an episode are not at risk, right-censored rows are dropped, not
called negative -- so with the default 3-month horizon the last 3 months of the
panel, i.e. the default OOT window, have no eligible rows for this target and
the report says "sin filas elegibles" instead of inventing a result). Metrics: AP (with its base rate), ROC-AUC (secondary),
precision/recall/lift/FP @K (`--review-capacity-k`, default 5% of eligible rows --
an assumption, flagged), episode recall@K, entity-cluster bootstrap 95% intervals
(`--event-bootstrap-reps`). A window with `< 20` positive episodes is
`conclusive: false` -- descriptive only, it cannot decide which model wins.

**Challengers (`src/models/event_challenger.py`).** L2 logistic regressions at
natural prevalence, no resampling: `logit_scores_head` (2 frozen scores),
`ridge_logistic` (<= `max_features` train-screened columns) and `discrete_hazard`
(person-period). Leakage is enforced in code: fit rows end `purge` months
(maturity lag; hazard: its horizon) before the first evaluation month and any
episode touching that embargo is dropped from fitting; `C` is chosen on the
validation months only. Reported: coefficients, events per parameter (warning
`< 10`), Brier/log-loss/calibration slope+intercept (only with >= 20 events),
and the IF/VAE baselines on the *same* rows. Nothing is promoted from this acta.
`--event-challengers off|auto|force`: `auto` runs them only if the gate allows
(red -> skipped with the reason); `force` runs them anyway, flagged exploratory.

**Fallbacks (the run never stops because of labels).** No file / empty or
missing folder -> `no_labels_file`; 0-byte file, header only, blank lines or an
all-empty target -> `empty_labels_file`; corrupt/binary/locked file, missing
key columns, unparseable months, no overlap with the panel -> `contract_error`;
red gate -> IF/VAE evaluated, no challenger; a family with too few training
episodes, one class, or a failed fit -> that family `skipped`/`failed` with its
reason. Any other exception inside the phase is caught in `main.py` and logged.
In every case the rest of the pipeline (Excel, dashboard, suite, sensitivity,
interpretability, report) runs unchanged, and the report chapter says whether
labels were used or ignored and why. `--no-run-event-supervision` skips the phase.

Artifacts under `artifacts/reports/`: `label_gate.json` (the acta: audit of the
file, metrics, level, vetoes, authorised families, thresholds),
`event_evaluation.csv` (one row per detector/challenger x target x window) and
`event_challengers.json`; the report gains "Labels de eventos y compuerta de
supervisión (4.5)". Tests: `tests/test_event_supervision.py`.

## IF-VAE Diagnostic Suite integration

The dated story of how this integration was built (the two fixes to the vendored
copy, the 2026-09-03 findings run, the 15-section → 9-section restructure)
lives in `CHANGELOG.md`; this section holds only what is true now.

**What it is.** A vendored standalone package (`tools/if_vae_diagnostic_suite/`,
v1.0.0, own `pyproject.toml`/CLI/tests/`AGENTS.md`) that diagnoses *why*
Isolation Forest and IF+VAE disagree. Phase 9c validates that `ifvae_diag` is
importable and, when needed, runs an editable install from this repository's
own vendored path — never resolving the name from an index — then adds the
vendored `src/` to `sys.path` so the package is usable in the **same process**
(invalidating import caches alone does not reprocess `.pth` files).
`--no-auto-install-suite` disables the side effect and reports the missing
suite explicitly. The suite is an optional dependency: outside tests, only
`src/evaluation/ifvae_diagnostic.py` imports it, and it never imports anything
from `src/`.

**Standalone run against this project:**
```
py tools/export_diagnostic_suite_inputs.py --out tools/if_vae_diagnostic_suite/build/modelo_run
cd tools/if_vae_diagnostic_suite
PYTHONPATH=src py -m ifvae_diag run --reference build/modelo_run/reference.csv \
  --scored build/modelo_run/scored.csv --config build/modelo_run/modelo_config.yaml \
  --out build/modelo_run/report
```
`export_diagnostic_suite_inputs.py` mirrors `main.py`'s Phases 2→3a→4→6→6b→7 to
fit a fresh IF+VAE pair and export the suite's contract: `reference.csv` = the
train block, `scored.csv` = the true OOT block, one row per **(entity, period)**
(not deduplicated to one row per entity like the OOT Excel). Ground truth is
attached there for this manual validation only — a real run has none.

**Phase 9c (in-process, ON by default).** `run_ifvae_diagnostic_suite` builds
`reference`/`scored` straight from this run's already-fitted detectors (no
refit for scoring, no CSV round-trip) and runs the suite **label-free**;
`--no-run-diagnostic-suite` skips the phase. Its output feeds two separate
report chapters — a reconciliation of "no interpretation in the ficha" with
"give me a toolkit and a recommendation" that must not be collapsed:

- **Factual ficha** (`ifvae_contract.build_diagnostic_contract`,
  `CONTRACT_VERSION "2.0.0"`; renderers in `src/reporting/diagnostic_section.py`):
  nine sections — alcance, configuración efectiva, concordancia y desacuerdo,
  candidatos VAE, sensibilidad, latente, estabilidad, temporal/segmentación,
  experimentos. Every visible value is traceable to a field with a status
  (`EXECUTED`/`NOT_REQUESTED`/`NOT_APPLICABLE`/`UNAVAILABLE`) and a reason; no
  silent fallbacks; the renderer computes nothing and uses no verdict language.
  Counts are entity–period observations, never "individuos".
- **Interpretation chapter** (`ifvae_interpretation.build_interpretation_contract`;
  renderers in `interpretation_section.py`): `{toolkit, indicator_validation,
  decision_flow, methodology_notes}`. Every claim carries a `basis` and a
  `severity` (`info`/`attention`/`caution`, never a pass/fail verdict). Drift is
  distilled with **Benjamini–Hochberg FDR** over every feature's KS p-value.
  The decision flow has 5 nodes evaluated on this run's numbers; the
  recommendation is assembled from whichever fragments the branches produced.
  `METHODOLOGY_NOTES` documents every threshold used and the ones deliberately
  *not* invented (see stability). This renderer may use interpretive language
  but, like the factual one, computes nothing itself.

**What runs, and the knobs (all `PipelineConfig` / `main.py` flags):**

- *Stability* — IF **and** VAE are genuinely refit `--diagnostic-stability-refits`
  times (default 3; `0` disables → `UNAVAILABLE` with a reason) with seeds
  `config.seed + 1000·i`, same architecture read off the fitted instance's own
  public attributes, and scored with the suite's own `top_k_stability`. VAE
  refits are full training runs — the single most expensive part of Phase 9c.
  **No universal "stable ≥ X" Jaccard cutoff is asserted**: published evidence
  (arXiv:2402.11404) shows autoencoder embeddings are routinely far less stable
  than tree ensembles, so only the degenerate case (mean Jaccard < 0.05) is
  flagged and everything else is reported numerically, comparatively.
- *Sensitivity grid* `(0.90, 0.95, 0.99)` and *entity view* — ON by default.
- *Segmentation* — `--diagnostic-segment-column` (default `segment`; `""`
  disables). A configured column absent from the panel logs a warning and makes
  §8 `NOT_APPLICABLE`.
- *§9 experiments* — **all active by default** (`src/evaluation/ifvae_experiments.py`,
  2026-09-25): ensembles and reconstruction variants reuse computed scores; the IF
  operating-point sensitivity compares the top-c % of the IF score with the production
  alert set (`0.02 0.01 0.005`, no refits: `contamination` does not change any score); and
  six families that used to be `NOT_REQUESTED` now run — VAE capacity/latent dimension,
  beta + KL schedule, loss by feature type, feature-family ablation, rolling-origin
  temporal backtests and window stability. Alert sets are built exactly like production
  (VAE: the diagnostic's `recon_topk` percentile vs the train block; IF: score percentile vs
  train), so the Jaccards are comparable (the previous VAE sweep compared a plain-MSE
  percentile with the production `recon_topk` one). All VAE retrains share ONE budget
  (`experiments.vae_fit_budget`, default 16 = 1 noise control + 5 capacity + 4 beta/KL + 1 one-hot control (embedding mode) + 3
  ablation + 2 backtest), epochs are capped (`epoch_cap` 30) and fits above `max_fit_rows` (300k) are
  subsampled; each cap is written into the row's detail, exhausted budget → `NOT_REQUESTED`
  with that reason. Each refit is isolated (temp checkpoint dir, `resume=False`) and a failing
  point is a `FAILED` row. Every variant row is read against a **control** (production configuration refitted with
  another seed and the same caps, compared through the same alert-set Jaccard); the
  stability section's raw-score top-k Jaccard is NOT used as a floor here (independent
  validation showed it measures a different set and was misleading).
  Backtest origins (last 6 periods, one step ahead, no gap) re-run `fit_transform_panel` with
  `fit_mask = period < origin` and fit on strictly earlier periods; the VAE backtest (2 last
  origins) uses the base matrix without the stacked IF column. Everything is descriptive (no
  pass/fail) and on this run's own data. Only "Preprocesamiento" stays `NOT_REQUESTED`.
  Switches/grids live in the `experiments:` block of `configs/pipeline.yaml`.

**Label-free mode (a change made to the vendored copy).** `label_col: str | None`;
`contracts.py`/`pipeline.py` skip every label-dependent output when it is `None`.
Percentiles, quadrants, latent diagnostics, drift, `report.md` and
`disagreement.png` still run. By design these do **not** apply to an official
(unsupervised) run: `metrics.csv`, `coverage.json`, `metrics_by_group.csv`,
`autopsies.csv`, `metrics.png`, and the two `*_percentile_orientation` warnings.
`if_stability.json` is unavailable whenever `if_score_col` is set (production
forest); the host measures IF stability itself instead (above). Other vendored
changes: `scripts/mutation_probe.py` uses `os.pathsep` (it was silently broken on
Windows), and `reporting.metrics_bar_plot` adds a precision@k chart when labels
exist. Read the drift table with calendar (`cyc__period_month_*`) and
panel-lag features filtered out — a chronological split "drifts" on
month-of-year by construction.

**Live progress (2026-09-23).** A full Phase 9c can spend minutes inside one
loop, so the suite reports what is running, for how long, and how far along.

- `ifvae_diag/progress.py` (suite-side, standalone) offers `step(name)` (one
  named function: start, then completed/failed with its own measured duration,
  nesting depth tracked), `track(iterable, desc=, unit=, label=)` (a tqdm bar
  over a loop; an item counts as done only when the caller finishes it; closed
  and reported even on early exit or exception) and `stages(desc, total=)` (a
  bar over a fixed sequence of named tests, each also a `step`). It is
  **instrumentation only** — no computed value changes. tqdm is optional (a
  missing import degrades to events only). Events are plain dicts sent to
  observers (`add_observer`); "update" events are throttled to
  `DEFAULT_MIN_INTERVAL_S = 0.5` so a JSONL log does not grow per iteration,
  while start/end always fire. One observer that raises is dropped, never
  allowed to break a test.
- What is instrumented: `run_diagnostic`'s 12 stages (each named after its
  function: `validate_frames` … `write_outputs`) with inner bars
  `isolation_forest[seeds]`, `compare_populations[features]` and
  `write_outputs[files]`; and, host-side in `src/evaluation/ifvae_diagnostic.py`,
  `_vae_forward[batches]`, the 11 `diagnose_frames` tests (`ifvae_diagnostic.
  _build_agreement` … `build_interpretation_contract`, bar `ifvae_diagnostic`),
  one step + bar tick per stability refit (`stability_refit[<Detector>]`,
  step `ifvae_diagnostic.refit[<Detector> seed=N]`) and per experiment-grid
  point (`experiment[<Detector>.<param>]`). Step names are `ifvae_diag.<stage>`
  for the suite's and `ifvae_diagnostic.<function>` for the bridge's.
- **The bridge** (`_suite_progress`, re-entrant, only the outermost block wires
  and unwires) forwards suite events to `logging_config.report_phase_event`
  (log line + `run_events.jsonl` + dashboard phase observers; a failed step is
  logged as a *warning*, since Phase 9c owns whether it is an incident) and
  `observability.progress_event`. **tqdm draws to the terminal only when
  `console_ui.is_live()` is false**: a repainting `rich` dashboard and tqdm's
  carriage-return redraws tear each other apart, so with the dashboard up the
  same bars are rendered *inside* it via `tqdm.format_meter` (the exact tqdm
  layout, no stderr writes). `--no-console-ui` (or a non-TTY) gives real tqdm
  bars. The rest of the pipeline reports through the pipeline-side twin,
  `src/utils/progress.py` (previous bullet).

**Tests.** `tests/test_diagnostic_section.py` drives the real contract path via
`diagnose_frames`/`build_interpretation_contract` (HTML/Markdown parity for both
contracts, traceability, no silent fallbacks, degenerate cases, decision-flow
branches, and **real seeded refits** with tiny `IsolationForestDetector`/
`VAEDetector`). `tests/test_live_progress.py` covers the progress path end to
end (bridge wiring, real refit steps, `/state` over a real local server, the
flow-state edge cases, dashboard rendering). The suite's own
`tests/test_progress.py` pins the primitives, and its quality gate must stay
green. `tools/render_diagnostic_example.py` renders both chapters from a
synthetic fixture without a pipeline run.

## IF tuner redesign, labels-aware objective and VAE reliability (2026-09-24)

Current contract (validation numbers in `CHANGELOG.md` 2026-09-24; **all measured on
synthetic data** — the direction of the ψ effect must be re-checked on real data):

- **IF tuner** (`src/models/iforest.py::tune_iforest`): `n_estimators=300` and
  `bootstrap=False` fixed; searches only `max_samples` (ψ, an **absolute integer**,
  log-uniform in `--iforest-max-samples-range`, default 1 024–32 768, capped to the fit
  rows) and `max_features` ∈ [0.5, 1.0]; Sobol (`QMCSampler`) with anchors at
  ψ ∈ {1 024, 4 096, 16 384, 32 768}. Same ψ in the trials, the refit and the stacking
  forest (`main.py` fits the latter on `train`, which is exactly the tuner's fit block).
  Untuned fallback: `PipelineConfig.iforest_params` = 300 trees, ψ=4096 (clipped to the
  rows), all features. `contamination` stays only as the deployed `predict()` operating
  point (`contamination_tuned: false` in the YAML).
- **Selection**: the paper default (ψ=256) is re-evaluated with `noise_seeds` (default 3)
  seeds → noise `sd`; deployed = cheapest replicated trial within 1 `sd` of the best
  replicated mean, kept only if it beats the default by > 1 `sd`, else the default. YAML
  `selection` block + `study.user_attrs["selection"]`. Cost:
  `(min(selection_top_k, n_trials) + 1) × noise_seeds` full refit+score cycles, sequential
  — deliberate (a real per-configuration noise estimate, not the tuning phase's
  single-seed value), and bounded independent of `n_trials` since **2026-09-28**: only the
  top `selection_top_k` (default 5) completed trials by single-seed value get replicated,
  not the whole budget (a trial ranked below that cutoff cannot be the true best once
  noise is accounted for). Still scales with the panel size on the scoring side
  (`max_samples` is capped absolute, the validation set it scores is not), so it can take
  real time on a large real panel (2026-09-27: 30 min observed pre-bound at `--full` scale,
  not a hang; a 2026-09-28 report of ~360k real rows/18 months is what prompted the bound).
  `iforest_selection_top_k`/`iforest_noise_seeds` (`PipelineConfig`, `--iforest-selection-
  top-k`/`--iforest-noise-seeds`, or `iforest.selection_top_k`/`iforest.noise_seeds` in
  `configs/pipeline.yaml`); `selection_top_k=0` skips the whole phase and deploys the
  tuner's raw single-seed winner (`deployed: "tuned_unreplicated"`), with none of the
  guarantees above — an explicit trade, never the default. Tracked live (`Bar`) and
  persistently (`log.info` per trial with cycles done/total, elapsed, ETA — the bar alone
  never reaches `execution.log`, tqdm writes straight to stderr). Tests:
  `TestSelectionTracking`, `TestSelectionTopKBound`, `TestSelectionSkipped` in
  `tests/test_tuning_iforest.py`.
- **Objective**: labelled → AP on the *known* validation rows (`NaN` = unknown); needs
  ≥ 10 positives and both classes, else it falls back and logs why. Label-free →
  `tail_separation`. `rank_agreement` remains selectable (`objective_metric=`), averaged
  over top-k fractions (0.5/1/2 %); it is a top-k **fraction**, never a decile.
- **Phase 5b** (`main.py`, before tuning): loads the reviewed labels and gate 4.5 once
  (`prepare_event_labels`); `labels_for_tuning` hands the tuner labels **only on the
  mature known validation rows** when the gate level is above `rojo` (no vetoes) and the
  validation months hold ≥ `--tune-min-positive-rows` (10) positives. Precedence:
  `--supervised` ground truth > reviewed labels > label-free. `--tune-with-labels off`
  disables it. Phase 8c reuses the same load. Recorded as `tuning.objective_source`.
- **Study identity / resume** (IF and VAE): the study name carries a fingerprint of the
  data, features, split, objective and search space; `n_trials` is a *total* budget
  (`n_trials − completed`); `RUNNING` trials from a crash are closed as failed.
- **VAE**: `save`/`load` persist `epochs`, `kl_anneal_epochs`, `early_stopping_patience`
  (was `epochs=0` on load); `fit` refuses `epochs < 1`; a checkpoint resumes only if the
  full training config **and** the data fingerprint match (legacy checkpoints start
  fresh); trial checkpoints live in `checkpoint_dir/<study_name>/trial_<n>`; the refit
  uses the winning trial's `epochs` and KL ramp (`_kl_anneal_for`); the diagnostic's VAE
  refits use a temp `checkpoint_dir` and `resume=False` (they used to resume the
  production checkpoint and return the same model regardless of seed).
- **Failure handling** (`main.py`): a failed IF/VAE tuning logs an ERROR, records
  `tuning.iforest_completed` / `tuning.vae_completed`, fits the default detector and
  **does not read** a best-params YAML left by an earlier run.
- **Phase 8c fixes**: (E-3) an unreadable/blank `label_available_at` now excludes the
  row (`availability_unreadable`; it used to pass); (E-2) contradictory duplicate rows
  stay unknown under `--labels-unlisted-as-negative` (they used to become negatives);
  (E-4) an episode counts as out-of-sample only if it **starts** in test/OOT; (E-8)
  `entity#suffix` episode ids escape `%`/`#` so they cannot collide.

## Single configuration file (`configs/pipeline.yaml`, 2026-09-25)

The diagnostic knobs a user edits most live in **one** file, `configs/pipeline.yaml` next
to `main.py` (or `--config PATH`), loaded by `src/utils/config_file.py`: the segment column
(`diagnostic.segment_column`), the analyst identity columns (`dashboard.identity_columns`),
the identification-only columns excluded from every modelling phase
(`data.identification_columns` — see "Identification columns" above), entity view,
stability refits, the sensitivity grid and the whole `experiments:` block.
Precedence: **explicit CLI flag > value set in code > file > built-in default** (the file only
overrides a field still at its built-in default and not given on the command line). Unknown
keys or invalid values are an error that lists the valid keys (a typo is never an ignored
setting). `execution.log` records the source of each value (`cli` / `code` / `file` /
`default`) and `config_sources` lands in the resolved config.

Why: the segment used to exist in three places (the `PipelineConfig` default, a CLI flag and
the vendored suite's default), editing the wrong one was a silent no-op and the report always
said "segmento". Now an **explicit** segment that is not a column of the panel stops the run
**at the start** (`_validate_segment_column`: closest match ignoring case + every available
column); only the built-in default degrades to `NOT_APPLICABLE` with a warning and an
observability incident. The report names the real column ("Por segmento (region)") and
`tools/export_diagnostic_suite_inputs.py` reads the same value. `--analyst-identity-columns`
no longer has a dead dataclass default (argparse default is `None`).

## Mixed-type VAE: embeddings for categorical variables (2026-09-25)

`vae.categorical_representation` (`configs/pipeline.yaml`, `--vae-categorical-representation`) chooses how the VAE sees categorical
variables. **Default `onehot`** (the original MLP over one-hot columns) until every acceptance criterion passes; `embedding` builds the
mixed-type VAE (`src/models/mixed_vae.py`, `architecture = mixed_v1`). Contract:

- **Two views.** IF view unchanged. VAE view (`src/preprocessing/mixed_view.py::MixedViewBuilder`): continuous / binary / missing-flag columns
  from the same causal preprocessing + ONE integer index column per categorical variable built from the raw panel column (tokens `0` = MISSING,
  `1` = UNKNOWN, `2..` = vocabulary sorted alphabetically; vocabulary learned on the train rows only, levels rarer than `rare_min_frequency` →
  UNKNOWN). No one-hot column enters the VAE (`assert_no_onehot`, check `vae.no_onehot_input`). Stacking appends the standardised IF score
  as one more numeric column (only that column is scaled).
- **Model.** Per-variable embeddings (`auto` width = `round(1.6·card^0.56)` clipped to `[2, 32]`), decoder heads: numeric (Huber/MSE), binary
  (BCE with logits), one softmax head per categorical (cross-entropy vs the true index; NLL floored at p = 1e-6). Family weights default 1.0.
  Loss and history are reported by family and by original categorical variable.
- **Score.** One contribution per ORIGINAL variable; `score_samples` = weighted mean. Diagnostic frames carry one column per original variable;
  `recon_topk` selects among variables. Contributions are centred at the train-reference median and scaled by MAD with a floor of
  `0.1 × mean` and `1e-6` (`MIXED_CONTRIBUTION_SCALE_FLOOR`, `MIXED_CONTRIBUTION_CENTER`; suite options `contribution_scale_floor`,
  `contribution_center`, both off by default for the one-hot path).
- **Identity.** `architecture_fingerprint` (variables + order, vocabularies/cardinalities, embedding dims, loss types/weights, token policy, code
  version) is part of checkpoint compatibility, the Optuna study name and the YAML. One-hot payloads/checkpoints are **rejected**, never loaded
  partially (`IncompatibleCheckpointError`; `VAEDetector.load(expect_architecture=, expect_fingerprint=)`).
- **Everywhere else.** Tuning (with and without labels, partial labels with `NaN`), the sensitivity study (perturbed frames re-encoded with the same
  vocabularies), attribution/explanations (original names; categorical shown as `variable=category (p=…)`) and the six §9 families (loss by type in
  original variables + MISSING/UNKNOWN rows + a one-hot control; ablation with a sub-layout; per-origin vocabularies in the backtest) work with both
  architectures.
- **Validation** (`tools/compare_vae_representations.py`, `docs/validation/2026-09-25_vae_representations/`): both arms are scored with the historical AND the
  centred normalisation (an independent review showed centring alone lowers the one-hot categorical share from 95 % to 70 %). With embeddings the categorical
  share of the top-k is 34–35 % vs 72–73 % under the same normalisation (25 % expected), `recon_topk` AP goes from chance to 0.22, seed stability 0.46 → 0.90
  (as-is). **Three criteria fail, so the default stays `onehot`**: tail separation of the production score (1.6 vs 2.5) and window stability judged literally
  against the complete one-hot arm (0.41 vs 0.78 Spearman; 0.05 vs 0.22 Jaccard — the one-hot value is inflated by time-invariant categorical attributes; like-for-like
  on numeric variables B is higher, 0.39 vs 0.25). Analyst explanations and the attribution chart rank NORMALISED contributions (train reference kept in the
  detector). Incompatible checkpoints are set aside, never overwritten. Synthetic data only.

## Known open problems

- **Diagnostic `recon_topk` is dominated by one-hot columns (found 2026-09-24, fix
  pending).** The diagnostic's VAE percentile is `vae_primary_score="recon_topk"`: the
  mean of the 5 largest `|residual| / MAD_reference` per row
  (`ifvae_diag.scoring.residual_contributions`). A one-hot column's residual is ~0 for
  most rows, so its MAD is ~10× smaller than a numeric column's and any active level is
  inflated (measured on the synthetic panel: 98 % of each row's top-5 contributors are
  one-hot, which are 66 % of the columns; Spearman(`recon_topk`, IF) = 0.06 versus 0.58
  for the raw MSE that the Excel deliverable uses; 8/76 of the VAE top-1 % rows have an
  IF percentile ≤ 20 % versus 0/76 with the raw MSE). It is a scale artifact, not model
  behaviour, and it inflates the VAE_ONLY quadrant and the "top features by error" table
  with `puesto`-like categorical levels. The IF is blind to categoricals by design
  (`split_matrix_for_model`), which is a second, legitimate source of VAE-high/IF-low.
  Remedy implemented behind `vae.categorical_representation: embedding` (see "Mixed-type VAE" above): one contribution per original variable, centred and
  floored normalisation. Still opt-in (default `onehot`) because the production score's tail separation is lower with embeddings; until the default flips, the
  "Pérdidas por tipo de feature" experiment keeps exposing the bias on every run.

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
- **The tuner itself could select a collapsed trial (fixed 2026-09-27).**
  Detection at the *final* fit (above) was not enough: ELBO, reconstruction
  loss and PR-AUC/ROC-AUC all stay finite and plausible on a decoder that has
  learned to ignore the latent code, so Optuna could rank a collapsed trial
  as "best" — observed twice in one session, with both `onehot` and
  `embedding`. `tune_vae`'s `objective` (`src/models/vae.py`) now runs
  `collapse_verdict` on every trial right after fitting and forces a
  collapsed one to the worst possible objective value, so it can only win a
  study where every trial collapsed (reported via `trial.user_attrs
  ["posterior_collapse"]` and the study's best value, not hidden). Tests:
  `TestAntiCollapseGuard` in `tests/test_tuning_vae.py`.
- **`--quick`'s VAE epoch budget may be too small to avoid collapse even with
  the scaling fix**, and its Optuna trial budget is small enough that
  `TrialPatienceStopper` essentially never fires (see above) — treat a
  `--quick` VAE result as a smoke test, not a measurement, until this is
  revisited (`docs/diagnostico_del_proyecto.md` A-10).
- **Cross-seed stability is measured only inside Phase 9c, and only for the
  alert set.** The pipeline itself trains with one fixed seed
  (`PipelineConfig.seed = 42`) and `unsupervised_metrics`'s `rank_stability` is
  a bootstrap-jitter proxy, not a refit-under-another-seed measurement (see
  `docs/validacion_no_supervisada.md` §6). What does exist: Phase 9c refits
  **both** IF and VAE `--diagnostic-stability-refits` times (default 3) and
  reports top-K Jaccard across those refits (see "IF-VAE Diagnostic Suite
  integration"). Three refits give a coarse reading, and no pass/fail cutoff is
  defined for it — there is no published one to lean on.
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
- **Testing**: `python -m pytest tests -q` (host project: diagnostic chapter,
  analyst dashboard, sensitivity, zero-row filter, report incidents, live/phase
  progress, event labels and gate 4.5, single config file, identification
  columns, IF/VAE tuners, §9 families, mixed VAE; ~5 min) plus the vendored
  suite's own gate, run **from its directory** —
  `cd tools/if_vae_diagnostic_suite && PYTHONPATH=src python scripts/quality_gate.py`
  (unit + adversarial tests, no tautologies, cyclomatic complexity ≤ 10,
  compile, mutation probe; the two suites cannot be collected in one pytest
  invocation because both have a top-level `scripts` package).
  **Rules for a test in `tests/`:**
  - It must pin a behaviour or a defect that was actually found (name the failure
    in the test name/docstring). Do not add tests that only check a file exists,
    grep the source for a name or a docstring word, or assert a magic size
    (`len(json) > 3000`) — those were removed in the 2026-09-25 audit.
  - A test that fits a real `VAEDetector` must pass `checkpoint_dir=<tmpdir>` and
    `resume=False`, otherwise it writes over `artifacts/models/vae/` (the
    2026-09-25 audit found and fixed the last one, in
    `RealStabilityTests._fit_tiny_detectors`). Tests still append to
    `artifacts/logs/execution.log`; that is only a log.
  - Do not re-run an expensive fixture per test: the §9 matrix takes 7–11 s, so
    the tests that only *read* the default matrix share one run through
    `default_rows()` (`test_ifvae_experiments.py`, `test_mixed_vae.py`); a test
    that changes a setting calls `run_families(**over)` itself.
  Beyond tests, verify a change by running the pipeline from a **throw-away
  working directory** (`cd <tmp> && py <repo>/main.py --quick`; the config file is
  resolved next to `main.py`, not the CWD) — running it in the project root
  rewrites `artifacts/`, so never do that as a side effect.
