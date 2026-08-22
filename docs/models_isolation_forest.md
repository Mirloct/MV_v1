# Isolation Forest — Concept, Implementation, and Tuning

This document covers the Isolation Forest anomaly detector shipped in
`src/models/iforest.py`: the underlying idea, this project's API around it,
how it consumes the preprocessing matrix, and how the Optuna tuning routine
recovers from crashes.

> Concept adapted from GeeksforGeeks, see
> `geeksforgeeks_notes.md` (section 2, "Isolation Forest").

---

## 1. Concept

Isolation Forest is an unsupervised, tree-based detector built on a different
intuition than most methods: instead of modeling what "normal" looks like and
measuring deviation from it, it directly *isolates* anomalies. Because
anomalies are few and different, they are easy to separate from the bulk of the
data with only a few random cuts.

The algorithm builds an ensemble of *isolation trees*. Each tree is grown by
recursively partitioning the data — pick a feature at random, then a random
split value within that feature's range — until points are isolated. The key
quantity is the **path length**: the number of splits needed to isolate a
point. Anomalies tend to be isolated near the root (short paths), while normal
points require many more splits (long paths). Averaging path lengths across the
ensemble yields an anomaly score; a shorter average path length means a higher
anomaly score and a greater likelihood of being an outlier. The method scales
well to high-dimensional data and is relatively robust to noise.

Key parameters:

- **`n_estimators`** — number of trees in the ensemble; more trees give more
  stable scores.
- **`max_samples`** — number of samples drawn to build each tree. Sub-sampling
  is a core part of the original algorithm; small samples actually help isolate
  anomalies.
- **`contamination`** — assumed proportion of anomalies in the data, used to set
  the score threshold that separates outliers from inliers.
- **`random_state`** — fixes the randomness for reproducible results.

Concept adapted from GeeksforGeeks:

- https://www.geeksforgeeks.org/machine-learning/what-is-isolation-forest/
- https://www.geeksforgeeks.org/machine-learning/anomaly-detection-using-isolation-forest/

Algorithmically, scikit-learn (which this project wraps) grows each tree to an
implicit height limit of `ceil(log2(max_samples))` and normalizes raw path
lengths by `c(n)`, the average path length of an unsuccessful binary-search-tree
lookup over `n` points, so the score `s = 2 ** (-E[h] / c(n))` is comparable
across sub-sample sizes (Liu, Ting & Zhou, 2008).

---

## 2. This project's implementation

`src/models/iforest.py` exposes `IsolationForestDetector`, a thin wrapper around
`sklearn.ensemble.IsolationForest` that standardizes the score sign, logs fit
time and matrix shape, accepts sparse or dense input, and persists via joblib.

### Score convention (important)

Throughout this project the anomaly score follows **higher = more anomalous**.
scikit-learn uses the opposite sign, so the wrapper flips it:

| Method | Returns | Sign convention |
| --- | --- | --- |
| `score_samples(X)` | anomaly score, one per row | **higher = more anomalous** (`-sklearn.score_samples`) |
| `decision_function(X)` | threshold-centered score | raw scikit-learn (negative = predicted outlier) |
| `predict(X)` | binary flag | `1` = anomaly, `0` = normal (remapped from sklearn's `-1`/`+1`) |

Use `score_samples` for ranking and for feeding evaluation/plots; use
`decision_function` only when you specifically want the contamination-shifted,
threshold-centered value.

### API

- `IsolationForestDetector(n_estimators=200, max_samples="auto", max_features=1.0, contamination="auto", bootstrap=False, random_state=42, n_jobs=-1)`
- `fit(X) -> self` — fits the underlying forest (logged via `log_phase`).
- `score_samples(X) -> ndarray` — higher = more anomalous.
- `decision_function(X) -> ndarray` — raw sklearn sign (negative = outlier).
- `predict(X) -> ndarray` — `1`/`0` anomaly flags.
- `save(path="artifacts/models/iforest.joblib") -> str` — joblib-serialize the detector.
- `IsolationForestDetector.load(path="artifacts/models/iforest.joblib")` — reload it.

### How it consumes the preprocessing matrix

The detector is deliberately decoupled from the data / out-of-time (OOT) logic.
It consumes an already-preprocessed feature matrix `X` — a dense `numpy.ndarray`
**or** a `scipy.sparse` CSR matrix, exactly as produced by
`src.preprocessing.pipeline.fit_transform_panel`. Isolation Forest is invariant
to monotonic feature scaling, but the pipeline's imputation, categorical
encoding, and within-entity panel features (lag/diff/own-history z-score,
seasonality) still materially shape the isolation cuts, so preprocess first.

The `(entity_id, period)` keys are held aside by the pipeline (`keys`, returned
alongside `X`) — entity/time are keys, not features. Joining detector scores
back to the separate ground-truth file via those keys is the evaluation
module's responsibility, not this module's.

---

## 2b. Parameter reference (verified against installed `scikit-learn==1.7.2`)

Every value below was read from `inspect.signature(sklearn.ensemble.IsolationForest.__init__)`
on this environment on 2026-08-19, not recalled from memory — sklearn has
changed these defaults across versions before (`contamination` used to
default to `0.1`; `max_samples` and `contamination` both moved to `"auto"`
in different releases), so a number quoted from a different installed
version would be wrong here. Two defaults differ: **sklearn's own default**
(what `IsolationForest()` alone would use) vs. **this project's default**
(what `IsolationForestDetector()` actually uses, `src/models/iforest.py:78`)
— the project overrides `n_estimators` and `contamination` deliberately; the
rest pass through unchanged.

| Parameter | sklearn default | Project default | Meaning | Alternatives and what they change | Trade-off |
| --- | --- | --- | --- | --- | --- |
| `n_estimators` | `100` | **`200`** | Number of isolation trees averaged into one score. | Any positive int. Optuna search space: `[100, 600]` step 50 (`tune_iforest`). | More trees → lower score variance across random seeds, linear cost increase (fit and score both scale ~linearly). 200 was chosen as a floor above sklearn's 100 because score-stability tests in this project's own docs (§3) showed rank agreement among the top anomalies is still seed-sensitive at 100. Diminishing returns past a few hundred; the real ceiling is wall-clock budget, not accuracy. |
| `max_samples` | `"auto"` (= `min(256, n_samples)`) | `"auto"` (unchanged) | Rows drawn (with replacement iff `bootstrap=True`) to build *each* tree. | `"auto"`, an int (exact row count), or a float in `(0, 1]` (fraction of rows). Optuna search space: categorical `{"auto", float in [0.3, 1.0]}`. | **Smaller, not larger, is the "more power" direction** for this algorithm — the original Isolation Forest paper's central result is that small sub-samples isolate anomalies in *fewer* splits because the normal bulk swamps large samples ("swamping"/"masking"). `"auto"`'s 256-row cap is deliberately small regardless of dataset size; raising it moves splits toward normal-region boundaries and typically *degrades* anomaly separation, it does not just cost more compute. |
| `contamination` | `"auto"` (sets an offset via an internal heuristic, not a literal rate) | **fixed by `tune_iforest(contamination=...)`, default `0.02`** | Only feeds `predict()`'s and `decision_function()`'s 0-centering; **does not affect `score_samples()`, the value this project ranks and thresholds on.** | Any float in `(0, 0.5]`, or `"auto"`. `IsolationForestAssumptionError` (this project's assumption gate, `src/utils/assumptions.py::validate_iforest_config`) blocks anything outside `(0, 0.5]`. | This is *not* searched by Optuna in this project, and the reason is load-bearing, not an oversight: `score_samples` doesn't consult it, and every rank-based tuning objective (`_rank_agreement`, `_tail_separation`) is invariant to it too, so a search over `contamination` would optimize nothing about the forest — see §3 "contamination is not searched" below and the retired `_separation_margin` postmortem in `CONTEXT.md`. **Do not read `contamination` as "the model's estimate of the true anomaly rate"** — it is an operating-point choice for `predict()`/`decision_function()` only, not a fitted or validated quantity. |
| `max_features` | `1.0` (all features per split) | `1.0` (unchanged) | Fraction of *columns* (not rows) considered when picking each split's random feature. | Float in `(0, 1]`. Optuna search space: `[0.3, 1.0]`. | Lower values add a second, independent source of tree diversity beyond row sub-sampling (closer to a Random-Forest-style feature bagging). Interacts with `max_samples`: both control how much of the data any one tree actually sees. |
| `bootstrap` | `False` | `False` (unchanged) | Whether the per-tree row sample is drawn *with* replacement. | `True`/`False`. Optuna search space: both. | `False` (sampling without replacement) is what the original algorithm describes and is the more common choice in practice; `True` changes the effective sample-diversity statistics slightly and is offered only because the tuning search costs nothing to include it. |
| `random_state` | `None` (non-reproducible) | **`42`**, threaded from `PipelineConfig.seed` | Seeds sklearn's internal `RandomState` for the split-feature/split-value draws. | Any int, or `None`. | `None` means **every fit produces a different forest and different scores** — for a project whose CONTEXT.md explicitly tracks measured metrics as evidence (e.g. "OOT ROC-AUC 0.71"), an unfixed seed would make those numbers non-reproducible from one run to the next. Fixed unconditionally here; there is no code path in this project that leaves it as sklearn's `None` default. |
| `n_jobs` | `None` (single-threaded) | **`-1`** (all cores) | Parallelism across trees during fit/score. | Any int, `-1`, or `None`. | Pure wall-clock knob — does not change the fitted forest or its scores (tree construction is embarrassingly parallel and each tree's random draws are independent of core count). `-1` is a safe default precisely because it cannot change results, only fit time. |

**Not exposed by this project's wrapper** (sklearn defaults apply, unmodified): `verbose` (`0`, no progress printing — this project's own `tqdm` bar in `tune_iforest` covers that need at the trial level instead) and `warm_start` (`False` — this project always fits from scratch; incremental forest growth is never used).

**Sensitivity analysis actually run vs. still open.** §3 below and `tests/test_iforest.py` cover: score finiteness/orientation under malformed input, resume/crash-recovery correctness, and rank agreement between two seeds at the tuned configuration (`TestRankAgreement`-style tests). **Not yet measured on this project's data**: a systematic sweep of `n_estimators` alone (holding everything else fixed) to find where score-variance-across-seeds actually plateaus, and a `max_samples` sensitivity curve. Treat "200 trees is enough" as this project's working assumption, justified by the reasoning above, not as a measured optimum — CONTEXT.md's own convention (label a claim as "evidence required" rather than proven) applies here too.

---

## 3. Tuning and crash recovery

`tune_iforest(X, n_trials=50, y=None, ...)` runs an Optuna study over the
detector's hyperparameters, refits the best configuration on all of `X`, and
saves it.

### SQLite study and resume

The study is created against a **persistent SQLite RDBStorage**, default
`sqlite:///artifacts/tuning/optuna_iforest.db`, with `load_if_exists=True`. That is the
recovery mechanism: if the process dies mid-search, re-running `tune_iforest`
with the **same `study_name` + `storage`** reopens the existing study and
continues from the already-completed trials rather than restarting. On top of
that, the current best hyperparameters are checkpointed to
`artifacts/tuning/best_params_iforest.yaml` (atomic write) after **every** completed
trial, so the best-so-far configuration is always durable on disk even if the
run is interrupted before it finishes.

To resume after an interruption, simply call `tune_iforest` again with the same
`study_name` and `storage` (both default, so a plain re-run resumes by default).
`n_trials` is the number of *new* trials to add on top of what already exists.

### Search space

- `n_estimators` — int in `[100, 600]`, step 50
- `max_samples` — categorical mode `{"auto", "float"}`; when `"float"`, a
  fraction in `[0.3, 1.0]`
- `max_features` — float in `[0.3, 1.0]`
- `bootstrap` — `{True, False}`

**`contamination` is not searched.** It is an explicit `tune_iforest` argument
(default `0.10`, the alert budget the OOT deliverable reports on).

> **Why.** sklearn's `score_samples` does not consult `offset_`; contamination
> only shifts the threshold used by `decision_function` / `predict`. Every
> objective here ranks rows by `score_samples`, and PR-AUC, ROC-AUC and Spearman
> agreement are all rank statistics — so contamination is *mathematically
> incapable* of changing an objective value. Searching it burned TPE budget on a
> flat dimension and persisted an arbitrary "best".
> `tests/test_iforest.py::TestContaminationIsNotSearched` asserts both its
> absence from the search space and the invariance premise itself.

### Held-out objective

Each trial fits on a `fit_idx` block and is scored on a disjoint `eval_idx`
block (`_blocked_split`, 70/30). When `groups` is supplied — `main.py` passes
`entity_id` — the split keeps **whole entities** on one side.

> **Why blocked.** All rows of one customer share the latent level that
> generated them, so a row-wise split lets a trial be scored on months of an
> entity it was fitted on. Scoring a trial on the rows it fitted also makes the
> objective an in-sample statistic, which a bigger forest can always improve
> without generalising better.

The final model is refit on all of `X`: the split exists to make model
*selection* honest, not to discard data once selected.

### Study fingerprinting

The effective study name is `f"{study_name}_{fingerprint}"`, hashing
`(X.shape, feature_names, objective mode, direction)`. Pass `study_tag` to
override.

> **Why.** `load_if_exists=True` resumes by name alone, so a fixed name pooled
> trials from a 2,000-entity one-hot panel with trials from a 100,000-entity
> frequency-encoded one, and TPE modelled a response surface stitched from
> incomparable values.

### Objective modes

`direction` defaults to `None` and auto-resolves (mirroring `tune_vae`); a
callable `objective_metric` without an explicit direction logs a warning.

- **Supervised** (`y` given, 0/1 labels aligned row-for-row to `X`): scores the
  **held-out block** against the labels, defaulting to **PR-AUC**
  (`average_precision_score`) — the informative summary for heavily imbalanced
  anomaly detection — switchable to **ROC-AUC** via
  `objective_metric="roc_auc"`. Falls back to the unsupervised objective when
  the held-out block is single-class.
- **Unsupervised** (`y is None`): `_rank_agreement` — refit the same
  configuration on two disjoint halves of `fit_idx` with different seeds, score
  the common held-out block with both, and return
  `max(spearman, 0) * jaccard(top-decile_a, top-decile_b)`. A constant score
  vector returns `0.0`.

> **Why this replaced `_separation_margin`.** The old proxy used the trial's own
> `contamination` as the tail fraction `k` at which it cut scores that are
> invariant to it. Shrinking `k` selects a more extreme tail, mechanically
> raising `mean(top) − mean(rest)` — so the objective was monotone in a knob
> that does not affect the model, and the search was driven to the lower bound
> of the contamination range regardless of forest quality. It optimised the
> metric's own parameter instead of the forest. `_rank_agreement` instead asks
> whether the ranking survives refitting on different data, which is what "this
> detector found real structure" actually means. It is the honest version of
> `evaluation.metrics._rank_stability`, whose docstring concedes it only jitters
> fixed scores because refitting is unavailable at metric-computation time —
> during tuning, it is not.

`objective_metric` may also be a callable `(detector, X) -> float` for a fully
custom objective.

### Measured: which anomaly geometry does it actually recover?

`evaluation.metrics.metrics_by_anomaly_type` breaks recall down by injected
type. Controlled sweep on `--quick` (800 entities × 6 periods, seed 42, fixed
detector hyperparameters — only `numeric_transform` varies), OOT recall@10 %:

| transform | IF PR-AUC | global | local | contextual |
|---|---|---|---|---|
| `robust` / `standard` (shape-preserving) | **0.272** / 0.265 | **1.000** | 0.200 | 0.600 |
| `yeo-johnson` (default) | 0.117 | 0.600 | 0.200 | 0.400 |
| `log1p` | 0.089 | 0.600 | 0.000 | 0.200 |
| `auto` (per-column, min `abs_skewness`) | 0.057 | 0.600 | 0.000 | 0.200 |

Two counter-intuitive conclusions:

1. **The Isolation Forest wants the heavy tail left alone.** Its mechanism is
   that extreme values isolate in few splits; `log1p` and `yeo-johnson` compress
   exactly the right tail where `global` anomalies live, so those anomalies then
   need *more* splits and score *lower*. Shape-preserving affine transforms
   double the PR-AUC and take `global` recall to 1.0 — consistent with
   `TestAffineRescalingIsANoOp`, which proves `robust` and `standard` are the
   same thing to this model.
2. **`local` is the blind spot** — 0.0–0.2 recall under every transform, mean
   score percentile ≈ 0.50, i.e. the forest ranks a local anomaly as no more
   suspicious than a median row. By construction a local anomaly is drawn from
   inside the population `[p2, p95]` band and is anomalous only against the
   entity's own history, so nothing in the marginal geometry separates it. The
   `_own_z` feature is the intended instrument and is evidently not enough. This
   is the open problem — now measured rather than assumed.

**The default is deliberately *not* changed to `robust`.** The VAE produces
100 % NaN scores under `robust`: the untouched tail reaches ~5×10⁵ in scaled
units (vs 47 under `yeo-johnson`) and the MSE gradients overflow. The two models
share one matrix and want opposite treatment; resolving that needs a per-model
shape transform, not a new shared default.

### Outputs

| Artifact | Default path |
| --- | --- |
| Optuna SQLite study | `artifacts/tuning/optuna_iforest.db` |
| Best params (incremental YAML) | `artifacts/tuning/best_params_iforest.yaml` |
| Refitted best detector | `artifacts/models/iforest.joblib` |
| Score-distribution figure | `artifacts/reports/figures/` |

`tune_iforest` returns the Optuna `Study`. `plot_score_distribution(scores, ...)`
writes a histogram (overlaying normal vs. anomaly when labels are supplied) to
`artifacts/reports/figures/` per the project-wide figures rule.

---

## 4. Minimal usage

Unsupervised tune-then-score (labels optional; pass `y` for the supervised
PR-AUC objective):

```python
from src.data import load_or_generate_panel
from src.preprocessing import fit_transform_panel
from src.models import IsolationForestDetector, tune_iforest

# 1. Load (or generate) the panel.
df, schema = load_or_generate_panel(
    data_path="artifacts/data/data.csv", n_individuals=1_000, n_periods=10, seed=42
)

# 2. Preprocess to a model-ready matrix (keys kept aside for later GT join).
X, keys, feature_names = fit_transform_panel(
    df, schema, numeric_transform="yeo-johnson", categorical_encoding="onehot"
)

# 3. Tune with Optuna (SQLite study -> artifacts/tuning/optuna_iforest.db).
#    Re-running with the same study_name/storage resumes after a crash.
#    Best params stream to artifacts/tuning/best_params_iforest.yaml every trial;
#    the refitted best detector is saved to artifacts/models/iforest.joblib.
study = tune_iforest(X, n_trials=25)   # add y=<0/1 labels> for supervised PR-AUC

# 4. Load the refitted best model and score.
detector = IsolationForestDetector.load("artifacts/models/iforest.joblib")
scores = detector.score_samples(X)     # higher = more anomalous
flags = detector.predict(X)            # 1 = anomaly, 0 = normal
```

To get 0/1 labels for the supervised objective, join the separate ground-truth
file to `keys` on `(entity_id, period)` (evaluation-side; see the Data contract
in `CONTEXT.md`). See `src/models/iforest.py` docstrings for the full API.
