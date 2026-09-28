# Isolation Forest — Concept, Implementation, and Tuning

This document covers the Isolation Forest anomaly detector shipped in
`src/models/iforest.py`: the underlying idea, this project's API around it,
how it consumes the preprocessing matrix, and how the Optuna tuning routine
recovers from crashes.

> Las fuentes introductorias consultadas están indexadas en
> `geeksforgeeks_notes.md`; esta guía aplica el concepto al código del proyecto.

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
— the project overrides `n_estimators`, `max_samples` and `contamination`
deliberately (see `PipelineConfig.iforest_params`); the rest pass through unchanged.

| Parameter | sklearn default | Project default | Meaning | Alternatives and what they change | Trade-off |
| --- | --- | --- | --- | --- | --- |
| `n_estimators` | `100` | **`300`, fixed** (never searched) | Number of isolation trees averaged into one score. | Any positive int. `tune_iforest(n_estimators=...)` fixes it for every trial and the refit. | More trees → lower score variance across seeds, linear cost. It is **fixed, not tuned**: the previous label-free objective (`rank_agreement`) rose mechanically with the tree count (Spearman 0.87 with `n_estimators`), so a searched `n_estimators` just pushed the search to its upper bound at ~5× the cost with no gain in true detection quality (measured, §3 "Measured"). 300 sits past the point where seed-to-seed score variance stops mattering for the top of the ranking. |
| `max_samples` | `"auto"` (= `min(256, n_samples)`) | **absolute integer** (untuned fallback `4096`, capped to the fit rows; tuner range `1 024–32 768`) | Rows drawn to build *each* tree (ψ). | An int (exact row count). **Never a fraction in this project**: a fraction is a different absolute ψ in the trial (share of the fit block), the refit (all in-time rows) and the stacking forest (train rows) — a 2.5× mismatch measured before the redesign, so what was validated was not what was deployed. `tune_iforest(max_samples_range=(lo, hi))`, log-uniform, capped to the rows each trial fits on. | The paper's argument for a small ψ (256: swamping/masking) holds when anomalies are extreme and rare. **It is regime-dependent, not a law**: on this project's synthetic panel (weak signal, `yeo-johnson`) true average precision *rose* with ψ up to a few thousand rows per tree, and the direction reverses under `robust` scaling. So the tuner searches ψ on an absolute log scale and always re-evaluates the paper default (ψ=256) as its reference; if nothing beats it by more than the measured noise, the default is kept (§3 selection rule). |
| `contamination` | `"auto"` (sets an offset via an internal heuristic, not a literal rate) | **`0.02`, an operating point — not tuned** (`tune_iforest(contamination=...)`) | Only feeds `predict()`'s and `decision_function()`'s 0-centering; **does not affect `score_samples()`, the value this project ranks and thresholds on.** | Any float in `(0, 0.5]`, or `"auto"`. `IsolationForestAssumptionError` (`src/utils/assumptions.py::validate_iforest_config`) blocks anything outside `(0, 0.5]`. | Not searched, and not "tuned to start at 0.02 then 0.01, 0.005": every score and every rank-based objective is invariant to it, so a search would optimise nothing about the forest. What *can* be examined is how the **alert set** changes when the cut-off moves over the same score: the diagnostic's §9 *Sensibilidad del punto de operación* compares the top-2 %, 1 % and 0.5 % of the IF score with the production alert set (no refits; the old refit sweep returned the identical forest three times). **Do not read `contamination` as the model's estimate of the true anomaly rate.** |
| `max_features` | `1.0` (all features per split) | `1.0` (unchanged) | Fraction of *columns* (not rows) considered when picking each split's random feature. | Float in `[0.5, 1.0]` in the tuner. | Lower values add a second source of tree diversity beyond row sub-sampling. Interacts with `max_samples`: both control how much of the data any one tree sees. Searched jointly with ψ. |
| `bootstrap` | `False` | `False`, fixed | Whether the per-tree row sample is drawn *with* replacement. | `True`/`False` via `tune_iforest(bootstrap=...)`; not searched. | `False` (sampling without replacement) is what the original algorithm describes; searching it added a dimension with no measurable effect on the objective. |
| `random_state` | `None` (non-reproducible) | **`42`**, threaded from `PipelineConfig.seed` | Seeds sklearn's internal `RandomState` for the split-feature/split-value draws. | Any int, or `None`. | `None` means **every fit produces a different forest and different scores** — for a project whose CONTEXT.md explicitly tracks measured metrics as evidence (e.g. "OOT ROC-AUC 0.71"), an unfixed seed would make those numbers non-reproducible from one run to the next. Fixed unconditionally here; there is no code path in this project that leaves it as sklearn's `None` default. |
| `n_jobs` | `None` (single-threaded) | **`-1`** (all cores) | Parallelism across trees during fit/score. | Any int, `-1`, or `None`. | Pure wall-clock knob — does not change the fitted forest or its scores (tree construction is embarrassingly parallel and each tree's random draws are independent of core count). `-1` is a safe default precisely because it cannot change results, only fit time. |

**Not exposed by this project's wrapper** (sklearn defaults apply, unmodified): `verbose` (`0`, no progress printing — this project's own `tqdm` bar in `tune_iforest` covers that need at the trial level instead) and `warm_start` (`False` — this project always fits from scratch; incremental forest growth is never used).

**Sensitivity analysis actually run.** The redesign is grounded in a 156-configuration grid on two labelled synthetic panels plus 15-trial study replays (§3 "Measured: the tuner redesign"). Everything measured is on **synthetic** data; the *direction* of the ψ effect must be re-checked on real data with `--tune` before anything is claimed about it.

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
`n_trials` is the **total** trial budget of the study: a resumed study runs only
the missing `n_trials − completed` (repeating a run does not silently grow it),
and trials a crash left `RUNNING` are closed as failed first.

### Search space

Two dimensions, nothing else:

- `max_samples` (ψ) — **absolute integer**, log-uniform in
  `max_samples_range` (default `1 024–32 768`, CLI `--iforest-max-samples-range`),
  capped to the rows each trial fits on.
- `max_features` — float in `max_features_range` (default `0.5–1.0`).

Fixed for every trial **and** the refit: `n_estimators=300`, `bootstrap=False`.

**`contamination` is not searched.** It is an explicit `tune_iforest` argument
(the deployed detector's `predict()` operating point; `contamination_tuned: false`
is written to the YAML).

> **Why.** sklearn's `score_samples` does not consult `offset_`; contamination
> only shifts the threshold used by `decision_function` / `predict`. Every
> objective here ranks rows by `score_samples`, and PR-AUC, ROC-AUC, Spearman
> agreement and the tail statistics are all rank/quantile statistics — so
> contamination is *mathematically incapable* of changing an objective value.

**Sampler.** A low-discrepancy Sobol sweep (`QMCSampler`), not TPE: with 15
noisy trials on an almost one-dimensional response, a model-based sampler has
nothing to model. On a fresh study the first trials are **anchors** at
ψ ∈ {1 024, 4 096, 16 384, 32 768} (clipped to the fit rows, all features), so
even a `--quick` run covers the axis that matters.

### Selection rule (why "argmax" is not enough)

The objective is noisy and the argmax of noisy values is biased upward: the trial that
"wins" the tuning search may just have gotten a lucky forest draw, not a genuinely
better configuration. After the trials, the tuner re-evaluates the **paper default**
(ψ=256, all features) with `noise_seeds` (default 3) different seeds — their spread is
the noise floor `sd` — and, alongside it, only the **top `selection_top_k`** (default 5)
completed trials by their single-seed tuning value (2026-09-27; a trial ranked below that
cutoff has no realistic chance of being the true best once noise is accounted for, so
replicating it spends the same cost for no new decision). Then, among the replicated
trials:

1. **1-standard-error rule** — among trials within one `sd` of the best value,
   take the *cheapest* (smallest ψ × `max_features`).
2. **Margin rule** — keep that pick only if it beats the reference mean by more
   than `margin_sd` (1.0) `sd`; otherwise the reference default is deployed.

Cost: `(min(selection_top_k, n_trials) + 1) * noise_seeds` full fit+score cycles —
bounded independent of `n_trials`, so a bigger tuning budget (typically wanted for a
bigger real panel) no longer multiplies this phase's wall-clock cost along with it. Each
cycle still scores the *full* validation set (unlike ψ, which has an absolute cap), so on
a large real panel this phase still takes real time — lower `selection_top_k` (down to 1,
which keeps rule 2 but drops rule 1) or `noise_seeds` (minimum 2) to bound it further, or
`selection_top_k=0` to skip the whole phase and deploy the tuning search's raw single-seed
winner directly (no noise floor, no margin check — an explicit trade, never the default).
`--iforest-selection-top-k` / `--iforest-noise-seeds`, or `iforest.selection_top_k` /
`iforest.noise_seeds` in `configs/pipeline.yaml`.

The decision (`selection` block: best/picked trial, noise, reference values,
`deployed: tuned | reference_default | tuned_unreplicated`, `beats_reference`,
`n_trials_completed`, `n_trials_replicated`) is written to the YAML and to
`study.user_attrs["selection"]`. A tuned configuration that cannot be told apart from
the default is not a finding.

### Held-out objective

Each trial fits on a `fit_idx` block and is scored on a disjoint `eval_idx`
block. `main.py` passes `valid_mask` (the chronological validation months), which
has priority: fit on the training months, score on the validation months — what
deployment looks like. Without it, `_blocked_split` (70/30) keeps **whole
entities** on one side when `groups` is supplied.

> **Why blocked.** All rows of one customer share the latent level that
> generated them, so a row-wise split lets a trial be scored on months of an
> entity it was fitted on. Scoring a trial on the rows it fitted also makes the
> objective an in-sample statistic, which a bigger forest can always improve
> without generalising better.

The final model is refit on all of `X`: the split exists to make model
*selection* honest, not to discard data once selected.

### Study fingerprinting

The effective study name is `f"{study_name}_{fingerprint}"`, hashing
`(X.shape, feature_names, objective mode, direction)` **plus** the fixed
`n_estimators`, the ψ / `max_features` ranges, `bootstrap`, the hash of the
held-out split and of the labels used, and the top-k fractions. Pass `study_tag`
to override.

> **Why.** `load_if_exists=True` resumes by name alone, so a fixed name pooled
> trials with incomparable objective values (different data, split, objective or
> search space) into one study.

### Objective modes

`direction` defaults to `None` → `maximize` (every built-in objective is
higher-is-better); a callable `objective_metric` without an explicit direction
logs a warning.

- **Labelled** (`y` given, aligned to `X`, `NaN` = unknown row, ignored): scores
  the held-out block against the labels on the **known rows only**, default
  **average precision**, switchable to ROC-AUC via `objective_metric="roc_auc"`.
  With reviewed labels this is the only real signal. It is used only when the
  held-out block has at least `min_eval_positives` (10) positives and both
  classes; otherwise the study says so in the log and falls back to label-free.
  `main.py` supplies it in this precedence: explicit `--supervised` ground truth
  → reviewed labels **when gate 4.5 authorises them** (level above `rojo`, no
  veto) and the validation months hold enough mature positive rows (Phase 5b,
  `--tune-with-labels auto|off`, `--tune-min-positive-rows`) → label-free. The
  reviewed labels reach the tuner only as a held-out target of the validation
  months, never as training data.
- **Label-free (default)**: `tail_separation` — `(p95 − p50) / IQR` of the
  held-out scores of a single forest fitted on the fit block. Both cut points are
  fixed constants, so the only way to raise it is to push the tail away from the
  bulk. It rewards *separation*, not *correctness* (nothing label-free can verify
  the tail holds the true anomalies), but it is unrelated to the tree count and
  tracked the true quality at Spearman 0.64 across the 156-configuration grid.
- **`objective_metric="rank_agreement"`** keeps the legacy stability objective:
  refit on two disjoint halves of `fit_idx`, score the common held-out block and
  return `max(spearman, 0) × jaccard(top-k_a, top-k_b)`, where
  *k = round(fraction × n_eval)* — a **top-k fraction, not a decile** (earlier
  text said "top-decile"; the code never was) — averaged over `tail_fracs`
  (0.5 %, 1 %, 2 %). Its ψ cap is half the fit block. Kept for comparison: it is
  dominated by `n_estimators` and noisy (sd ≈ 0.034).

> **History.** `_separation_margin` was retired because it cut the tail at the
> trial's own `contamination`, a knob that does not affect the scores, so the
> search optimised the metric's parameter instead of the forest. `_rank_agreement`
> replaced it and was in turn demoted to an option because of its tree-count
> dependence.

`objective_metric` may also be a callable `(detector, X) -> float`.

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
   double the PR-AUC and take `global` recall to 1.0 — consistent with the
   theory above: `robust` and `standard` are the same affine shape to this
   model, so they score identically.
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
