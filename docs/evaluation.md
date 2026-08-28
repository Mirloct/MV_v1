# Evaluation — Chronological Split, Metrics, Threshold, and the Top-N Deliverable

> The split, threshold-calibration and deliverable contracts summarised here are
> derived in full — with the leakage argument behind each — in
> [`docs/leakage_free_pipeline.md`](leakage_free_pipeline.md).

This document covers the evaluation module in `src/evaluation/`: the
out-of-time (OOT) split, the ground-truth join, the supervised and unsupervised
metric suites, the human-readable scored frame, the headline risk-ranked
Excel export (percentile-based by default), and the evaluation figures. It
explains the concepts, the API, and gives a runnable end-to-end snippet.

> Metric concepts adapted from GeeksforGeeks, see
> `geeksforgeeks_notes.md` (section 5, "Evaluation Metrics for
> (Imbalanced) Classification / Anomaly Detection"). Source URLs are cited in
> [Section 3](#3-metric-definitions).

---

## 1. Concepts

### Out-of-time (OOT) evaluation

The banking panel is a *balanced* panel keyed by `(entity_id, period)`: every
entity is observed once per period. A genuine **out-of-time** holdout treats the
**last period** in the panel as the OOT month. Detectors are fit on the in-time
rows (all earlier periods) and score every row; evaluation and the business
deliverable focus on the OOT month. This mimics production: score a future month
the model never saw at fit time, rank its individuals by anomaly score, and hand
the riskiest tail to analysts.

`src/evaluation/splits.py` is the single source of truth for that split, so the
OOT month is never hard-coded elsewhere:

- `oot_period(keys, time_col="period", n_oot_periods=1)` returns the last
  `n_oot_periods` distinct period value(s), in ascending order.
- `oot_split(keys, time_col="period", n_oot_periods=1)` returns
  `(in_time_mask, oot_mask)` — complementary boolean numpy arrays row-aligned
  with `keys` (hence with the feature matrix `X`).

### Ground truth is joined, not assumed

Detection is unsupervised by default; anomaly labels live in a **separate**
ground-truth file (parquet, or a CSV sibling when no parquet engine is
installed), never in `data.csv`. For offline evaluation,
`load_ground_truth_labels(schema, keys)` reads `schema.ground_truth_path` and
left-joins `is_anomaly` onto the `keys` frame, returning a 0/1 int vector aligned
row-for-row with `X`. The join is coercion-robust (entity ids compared as
strings, periods as datetimes) so a parquet/CSV round-trip cannot break the
match. When no ground-truth file is available — a genuinely unlabeled real
dataset — an all-zero vector is returned with a logged warning, and supervised
metrics simply degrade to `NaN` and can be skipped.

---

## 2. Module API

| Function | File | Purpose |
| --- | --- | --- |
| `chronological_split(keys, time_col, n_val_periods, n_test_periods)` | `splits.py` | **Train / validation / test masks, strictly by period.** The pipeline's split. |
| `oot_period(keys, time_col, n_oot_periods)` | `splits.py` | OOT period value(s) = last `n_oot_periods` distinct periods. |
| `oot_split(keys, time_col, n_oot_periods)` | `splits.py` | 2-way `(in_time_mask, oot_mask)`; superseded by `chronological_split` in `main.py`. |
| `load_ground_truth_labels(schema, keys)` | `labels.py` | 0/1 `is_anomaly` vector aligned to `X` rows; all-zeros when unlabeled. |
| `load_ground_truth_types(schema, keys)` | `labels.py` | Per-row `anomaly_type` string, for the per-geometry breakdown. |
| `supervised_metrics(y_true, scores, ...)` | `metrics.py` | Ranking + threshold + cost-sensitive metrics (labels required). |
| `unsupervised_metrics(X, scores, ...)` | `metrics.py` | Label-free cluster-quality + rank-stability proxies. |
| `metrics_by_anomaly_type(y_true, y_type, scores, ...)` | `metrics.py` | Recall@k per anomaly geometry, against the **global** top-k. |
| `calibrate_threshold(scores, method="pot"\|"percentile", ...)` | `thresholds.py` | **Alert cut-off fitted on validation scores** (GPD/POT or percentile). |
| `apply_threshold(scores, threshold)` | `thresholds.py` | 0/1 alert flags. |
| `build_scored_frame(raw_df, keys, scores, schema)` | `scoring.py` | Attach `anomaly_score` + **raw** features to `(entity_id, period)`. |
| `export_oot_top_anomalies(scored_df, schema, min_percentile=90.0, ...)` | `oot_report.py` | **The deliverable:** distinct individuals at/above a score percentile (default P90), ID–PERIOD–SCORE–BAND–VARIABLES `.xlsx`. `top_n`/`top_fraction` switch to a fixed headcount instead. |
| `export_oot_top_decile(scored_df, schema, ...)` | `oot_report.py` | Percentage-based wrapper kept for backwards compatibility. |
| `plot_embedding(X, scores_or_labels, method=...)` | `visualize.py` | 2D PCA/t-SNE/UMAP scatter coloured by score. |
| `plot_roc_pr(y_true, scores)` | `visualize.py` | ROC + PR curves side by side. |

**Score convention (project-wide): higher score = more anomalous.** Both
detectors follow it — the Isolation Forest exposes `-sklearn.score_samples`, and
the VAE exposes the per-row MSE reconstruction error. Every ranking in this
module sorts descending accordingly.

---

## 3. Metric definitions

`supervised_metrics(y_true, scores, k_fractions=(0.01, 0.05, 0.10), cost_fp=1.0,
cost_fn=10.0)` returns a flat `dict` of floats (JSON/YAML serializable). All
metrics are built from the **confusion matrix** — True/False Positives and
Negatives (TP, FP, TN, FN) — because on an extremely imbalanced problem (positives
can be well under 1%) plain accuracy is misleading: always predicting "normal"
scores highly. Every quantity degrades gracefully to `NaN` (never a
`ZeroDivisionError`) when there are no positives or only a single class.

- **`roc_auc`** — ROC-AUC: the area under the curve of true-positive rate vs
  false-positive rate across all thresholds. ~1.0 is strong separation, 0.5 is
  random. On heavily imbalanced data it can look overly optimistic, because a
  large true-negative count suppresses the false-positive rate.
- **`pr_auc`** — Precision-Recall AUC (average precision). Because it ignores
  true negatives and focuses on the rare positive class, it is the more
  informative summary than ROC-AUC for anomaly detection.
- **`best_f1` / `best_f2`** (+ `*_threshold`) — the best achievable F1 and F2
  over the full threshold sweep. F1 = harmonic mean of precision and recall
  (`2PR / (P + R)`); it is only high when both are high. F-beta =
  `(1 + beta^2) PR / (beta^2 P + R)` generalizes F1: F2 (beta=2) up-weights
  **recall**, matching the fact that missing an anomaly is costlier than a false
  alarm.
- **`precision_at_best_f1` / `recall_at_best_f1`** — precision and recall at the
  best-F1 operating point.
- **`mcc`** — Matthews correlation coefficient at the best-F1 threshold (the
  documented operating point).
- **`precision_at_<k>` / `recall_at_<k>` / `lift_at_<k>`** — for each `k` in
  `k_fractions`, with `K = ceil(k * N)` top-scored rows. When only the top-scored
  candidates can be reviewed (a common operational constraint), rank by score and
  evaluate the top-K. **Precision@K** = fraction of the top-K that are true
  anomalies; **Recall@K** = fraction of all true anomalies captured within the
  top-K; **Lift@K = Precision@K / base_rate** — how many times better than random
  selection the top-K list is (>1 means the ranking concentrates anomalies). Tags
  are compact percentages, e.g. `precision_at_1pct`, `lift_at_10pct`.
- **`expected_loss`** (+ `_total`, `_threshold`, `_flag_fraction`) —
  cost-sensitive expected per-sample loss at the cost-optimal cut:
  `min over cuts of (cost_fp * FP + cost_fn * FN) / N`. The cost matrix
  `cost_fn > cost_fp` encodes the banking economics that a missed anomaly (false
  negative — undetected fraud/loss) is far more expensive than a false alarm
  (false positive — a wasted analyst review). The default 10:1 ratio is a
  placeholder; override `cost_fn`/`cost_fp` with real per-error costs.
  `expected_loss_flag_fraction` is the share of rows flagged at that optimum;
  `expected_loss_threshold` is the score cut (or `NaN` when flagging nothing is
  cheapest).
- Bookkeeping: `n`, `n_positive`, `base_rate`, `cost_fp`, `cost_fn`.

`unsupervised_metrics(X, scores, contamination=0.05, sample_size=5000,
random_state=42)` needs no labels. It induces a 2-cluster (anomaly/normal) split
at the top-`contamination` scores and, on a random subsample of at most
`sample_size` rows (the cluster indices are ~O(n^2)), returns:

- **`silhouette`** — silhouette score of that labeling (how separated the flagged
  tail is from the bulk in feature space).
- **`calinski_harabasz`** — Calinski-Harabasz index of the same labeling.
- **`rank_stability`** — a bootstrap-jitter proxy: fix a reference subsample of
  scores, perturb them with Gaussian jitter scaled to 1% of the score standard
  deviation, re-rank, and report the mean Spearman correlation against the
  un-perturbed ranking over `n_boot` bootstraps. Well-separated scores give ~1.0
  (a small nudge cannot reorder them); heavily tied/flat distributions — whose
  top-anomaly ordering is fragile — score lower. It is a heuristic (true stability
  would require re-fitting the model on resamples, which is unavailable at
  metric-computation time).
- Bookkeeping: `contamination`, `n`, `n_flagged`, `subsample_size`.

Degenerate single-cluster subsamples (all-normal or all-anomaly) yield `NaN`
cluster metrics rather than raising.

**Sources (GeeksforGeeks, paraphrased in `geeksforgeeks_notes.md`
section 5):**

- ROC / ROC-AUC — https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/
- F1 / F-beta — https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/
- Confusion-matrix metrics — https://www.geeksforgeeks.org/machine-learning/evaluation-metrics-for-classification-model-in-python/
- Imbalanced-data handling — https://www.geeksforgeeks.org/machine-learning/handling-imbalanced-data-for-classification/

---

## 4. The risk-ranked Excel deliverable

This is the project's headline business artifact.

`build_scored_frame(raw_df, keys, scores, schema)` first joins the per-row
anomaly `scores` back onto the **raw** panel (`raw_df`, pre-preprocessing) via the
`(entity_id, period)` keys, returning a tidy frame of
`[entity_id, period, anomaly_score, <raw feature columns...>]`. The raw features
matter: the deliverable must be readable by a human reviewer, so it carries the
**original** feature values, not the scaled/encoded model matrix. `scores` must be
row-aligned with `keys` (the `fit_transform_panel` order); a mismatch raises.

`export_oot_top_anomalies(scored_df, schema, min_percentile=90.0, top_n=None,
top_fraction=0.10, model_name="model", n_oot_periods=1, score_col="anomaly_score",
threshold=None)` writes every individual **at or above the 90th percentile of
the OOT-block score** by default — a percentile rather than a fixed headcount
because the queue then scales with the portfolio ("the riskiest 10%" holds at
2,000 customers or 200,000). Sorted by score descending (ties broken by
`entity_id` ascending, stable sort). Passing `top_n` (or setting
`min_percentile=None` and leaving `top_fraction`) instead selects a fixed
headcount/fraction, the older behavior. Returns `(path, table_df)`.

**Exact column layout (hard requirement):**

| Column 1 | Column 2 | Column 3 | Column 4 | Columns 5+ |
| --- | --- | --- | --- | --- |
| **ID** (`entity_id`) | **PERIOD** (`time_col`) | **SCORE** (`anomaly_score`) | **BAND** (`p90`/`p95`/`p99`, the highest reached) | **VARIABLES** (raw, human-readable feature values) |

An `alert` column is inserted after BAND when a calibrated `threshold` is
passed. `main.py` (Phase 9) also attaches a `top_5_variables` column — the
per-row explanation of *why that individual* scored high (`explain_rows_
iforest`/`explain_rows_vae`, `src/interpretability/`), as opposed to the
population-level ranking `shap_summary_iforest`/`reconstruction_error_by_
feature` produce — computed only for the OOT rows this deliverable draws
from, not the whole panel. It lands among the VARIABLES columns (last, since
`export_oot_top_anomalies` preserves `scored_df`'s existing column order),
not in the fixed ID–PERIOD–SCORE–BAND layout above, and is best-effort: a
failure computing it logs a warning and the export still proceeds without
it. See `CONTEXT.md` "Per-row explanation" and `CHANGELOG.md` 2026-08-28.

Percentile cut-offs (and the band assignment) are computed over the
**full** de-duplicated OOT population before any selection — never over the
exported subset, which would be circular. If `out_path` is left at the
default and a `model_name` is given, the file becomes
`artifacts/reports/oot_p<percentile>_<model>.xlsx` (e.g.
`artifacts/reports/oot_p90_iforest.xlsx`) under the percentile default, or
`oot_top<N>_<model>.xlsx` under the fixed-headcount mode. The `.xlsx` write
uses the `openpyxl` engine. Output lands under `artifacts/reports/`.

---

## 5. Figures

Per the project-wide **hard figures rule**, all evaluation figures are written
under `artifacts/reports/figures/`:

- `plot_embedding(X, scores_or_labels, method="pca"|"tsne"|"umap", y=None)` —
  a 2D scatter coloured by anomaly score, with optional ground-truth anomalies
  drawn as ring markers. `umap` falls back to PCA (with a warning) when
  `umap-learn` is not importable. Returns the PNG path.
- `plot_roc_pr(y_true, scores)` — ROC and PR curves side by side. Returns `None`
  (with a warning) when the labels contain a single class, in which case the
  curves are undefined.
- `plot_score_comparison(scores_a, scores_b=None, names=("iforest", "vae"))` —
  overlays one or two detectors' score distributions.

---

## 6. End-to-end snippet

Load → preprocess → OOT split → fit on in-time → score all rows → metrics →
export the top-N risk-ranked Excel → figures.

```python
import numpy as np

from src.data import load_or_generate_panel
from src.preprocessing import fit_transform_panel
from src.models import IsolationForestDetector
from src.evaluation import (
    oot_split,
    load_ground_truth_labels,
    supervised_metrics,
    unsupervised_metrics,
    build_scored_frame,
    export_oot_top_anomalies,
    calibrate_threshold,
    plot_embedding,
    plot_roc_pr,
)

# 1. Load the panel and preprocess to the model matrix (keys kept aside).
df, schema = load_or_generate_panel(data_path="artifacts/data/data.csv",
                                    n_individuals=1_000, n_periods=10, seed=42)
X, keys, feature_names = fit_transform_panel(df, schema)

# 2. OOT split: last period held out from fitting.
in_time_mask, oot_mask = oot_split(keys, time_col=schema.time_col or "period")

# 3. Fit the detector on the in-time rows only, then score EVERY row.
detector = IsolationForestDetector(random_state=42)
detector.fit(X[in_time_mask])
scores = detector.score_samples(X)          # higher = more anomalous

# 4. Metrics. Labels come from the SEPARATE ground-truth file (all-zeros if none).
y = load_ground_truth_labels(schema, keys)
sup = supervised_metrics(y[oot_mask], scores[oot_mask])     # NaN-safe if unlabeled
uns = unsupervised_metrics(X[oot_mask], scores[oot_mask])
print("PR-AUC:", sup["pr_auc"], "Lift@10pct:", sup["lift_at_10pct"])
print("rank_stability:", uns["rank_stability"])

# 5. Human-readable scored frame (RAW features) and the OOT Excel deliverable.
scored = build_scored_frame(df, keys, scores, schema)
path, table = export_oot_top_anomalies(scored, schema, model_name="iforest")
print("risk-ranked Excel:", path)              # -> artifacts/reports/oot_p90_iforest.xlsx

# 6. Figures -> artifacts/reports/figures/
plot_embedding(X[oot_mask], scores[oot_mask], method="pca", y=y[oot_mask])
plot_roc_pr(y[oot_mask], scores[oot_mask])   # None if OOT month has no positives
```

Swap `IsolationForestDetector` for `VAEDetector` to score with the VAE; the score
convention (higher = more anomalous) and the whole downstream flow are identical.

See `docs/models_isolation_forest.md` and `docs/models_vae.md` for the detectors,
and `CONTEXT.md` for the data and preprocessing contracts.
