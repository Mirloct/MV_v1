"""Evaluation-metric suite for anomaly detection (project spec section 6).

Two entry points, both returning plain ``dict``s of Python floats so reporting
can dump them straight to JSON/YAML:

* :func:`supervised_metrics` -- computed only when ground-truth labels exist.
  Ranking + threshold + cost-sensitive metrics for a heavily imbalanced,
  rare-positive problem.
* :func:`unsupervised_metrics` -- label-free cluster-quality and rank-stability
  proxies, computed on a subsample because the cluster indices are roughly
  O(n^2).

Score convention (project-wide): **higher score = more anomalous**. Every
ranking here sorts descending accordingly.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    silhouette_score,
)

__all__ = ["supervised_metrics", "unsupervised_metrics", "metrics_by_anomaly_type"]

_NAN = float("nan")


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def _pct_tag(frac: float) -> str:
    """Compact percentage tag for a k-fraction, e.g. 0.01 -> '1pct', 0.005 -> '0.5pct'."""
    pct = frac * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}pct"
    return f"{pct:g}pct"


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    """Row indices ordered most-anomalous first (stable, project-wide convention)."""
    return np.argsort(-np.asarray(scores, dtype=float).ravel(), kind="stable")


def _top_k(frac: float, n: int) -> int:
    """Alert-budget size for a k-fraction: at least 1 row, at most ``n``."""
    if n <= 0:
        return 0
    return min(n, max(1, math.ceil(frac * n)))


def supervised_metrics(
    y_true,
    scores,
    k_fractions: Sequence[float] = (0.01, 0.05, 0.10),
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
) -> dict:
    """Supervised metrics for anomaly scores against 0/1 labels.

    Score convention: higher = more anomalous (rows are ranked descending).

    Metrics
    -------
    * ``roc_auc`` / ``pr_auc`` -- ROC-AUC and PR-AUC (average precision).
      PR-AUC is the more informative summary under extreme imbalance.
    * ``best_f1`` / ``best_f2`` (+ their thresholds) -- best achievable F1 and
      F2 over the full threshold sweep. F2 up-weights recall (missing an anomaly
      is costlier than a false alarm).
    * ``mcc`` -- Matthews correlation coefficient **at the best-F1 threshold**
      (the documented operating point).
    * ``precision_at_<k>`` / ``recall_at_<k>`` / ``lift_at_<k>`` -- for each
      ``k`` in ``k_fractions`` with ``K = ceil(k * N)`` top-scored rows.
      ``Lift@K = Precision@K / base_rate`` -- how many times better than random
      selection the top-K list is (>1 means the ranking concentrates anomalies).
    * ``expected_loss`` -- cost-sensitive expected per-sample loss at the
      cost-optimal threshold: ``min_t (cost_fp * FP(t) + cost_fn * FN(t)) / N``.
      The cost matrix ``cost_fn > cost_fp`` encodes the banking economics that a
      missed anomaly (false negative -- e.g. undetected fraud / loss) is far more
      expensive than a false alarm (false positive -- e.g. a wasted analyst
      review). The default 10:1 ratio (``cost_fn=10``, ``cost_fp=1``) is a
      placeholder; callers should override it with real per-error costs. The
      sweep therefore favours flagging more of the ranking than a plain
      accuracy-optimal cut would. ``expected_loss_total`` is the same quantity
      un-normalized, and ``expected_loss_flag_fraction`` is the share of rows
      flagged at that optimum.

    Every quantity degrades gracefully to ``NaN`` (never a ZeroDivisionError)
    when there are no positives / a single class.

    Args:
        y_true: 0/1 labels, length N.
        scores: Anomaly scores, length N (higher = more anomalous).
        k_fractions: Top-fractions for Precision/Recall/Lift@K.
        cost_fp: Cost assigned to each false positive.
        cost_fn: Cost assigned to each false negative (should exceed ``cost_fp``).

    Returns:
        A flat ``dict`` of floats (plus a few integer-valued floats / the cost
        settings), JSON/YAML serializable.
    """
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(scores, dtype=float).ravel()
    n = int(s.size)
    n_pos = int(y.sum())
    base_rate = n_pos / n if n else _NAN

    out: dict = {
        "n": float(n),
        "n_positive": float(n_pos),
        "base_rate": _safe_float(base_rate),
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
    }

    both_classes = 0 < n_pos < n
    out["roc_auc"] = _safe_float(roc_auc_score(y, s)) if both_classes else _NAN
    out["pr_auc"] = _safe_float(average_precision_score(y, s)) if both_classes else _NAN

    # --- best-threshold F1 / F2 and MCC at the best-F1 point --------------- #
    if both_classes:
        prec, rec, thr = precision_recall_curve(y, s)
        # prec/rec have length L+1; thr length L. Align on the first L points,
        # each of which corresponds to threshold thr[i].
        p, r = prec[:-1], rec[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
            b2 = 4.0  # beta=2 -> beta^2
            f2 = np.where((b2 * p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
        if f1.size:
            i1 = int(np.argmax(f1))
            i2 = int(np.argmax(f2))
            out["best_f1"] = _safe_float(f1[i1])
            out["best_f1_threshold"] = _safe_float(thr[i1])
            out["precision_at_best_f1"] = _safe_float(p[i1])
            out["recall_at_best_f1"] = _safe_float(r[i1])
            out["best_f2"] = _safe_float(f2[i2])
            out["best_f2_threshold"] = _safe_float(thr[i2])
            y_pred = (s >= thr[i1]).astype(int)
            try:
                out["mcc"] = _safe_float(matthews_corrcoef(y, y_pred))
            except (ValueError, RuntimeWarning):
                out["mcc"] = _NAN
        else:  # pragma: no cover - defensive
            out.update(
                best_f1=_NAN, best_f1_threshold=_NAN, precision_at_best_f1=_NAN,
                recall_at_best_f1=_NAN, best_f2=_NAN, best_f2_threshold=_NAN, mcc=_NAN,
            )
    else:
        out.update(
            best_f1=_NAN, best_f1_threshold=_NAN, precision_at_best_f1=_NAN,
            recall_at_best_f1=_NAN, best_f2=_NAN, best_f2_threshold=_NAN, mcc=_NAN,
        )

    # --- Precision@K / Recall@K / Lift@K ----------------------------------- #
    order = _rank_desc(s)  # most anomalous first
    y_ranked = y[order]
    tp_cumulative = np.cumsum(y_ranked)
    for frac in k_fractions:
        tag = _pct_tag(frac)
        if n == 0:
            out[f"precision_at_{tag}"] = _NAN
            out[f"recall_at_{tag}"] = _NAN
            out[f"lift_at_{tag}"] = _NAN
            continue
        k = _top_k(frac, n)
        tp = int(tp_cumulative[k - 1])
        prec_k = tp / k
        rec_k = tp / n_pos if n_pos > 0 else _NAN
        lift_k = prec_k / base_rate if (n_pos > 0 and base_rate > 0) else _NAN
        out[f"precision_at_{tag}"] = _safe_float(prec_k)
        out[f"recall_at_{tag}"] = _safe_float(rec_k)
        out[f"lift_at_{tag}"] = _safe_float(lift_k)

    # --- cost-sensitive Expected Loss -------------------------------------- #
    # Sweep every "flag the top-i scored rows as anomalies" cut (i = 0..N) and
    # take the minimum of cost_fp*FP + cost_fn*FN. i = 0 flags nothing
    # (FP = 0, FN = n_pos). For i >= 1: FP = (#negatives in top-i),
    # FN = n_pos - (#positives in top-i). O(N).
    if n == 0:
        out.update(
            expected_loss=_NAN, expected_loss_total=_NAN,
            expected_loss_threshold=_NAN, expected_loss_flag_fraction=_NAN,
        )
    else:
        idx = np.arange(1, n + 1)
        fp = idx - tp_cumulative               # negatives among the top-i
        fn = n_pos - tp_cumulative             # missed positives
        loss_flagged = cost_fp * fp + cost_fn * fn
        loss_none = cost_fn * n_pos            # flag nothing
        best_flag_i = int(np.argmin(loss_flagged))       # 0-based -> flags i+1 rows
        best_loss_flagged = float(loss_flagged[best_flag_i])
        if loss_none <= best_loss_flagged:
            total_loss = float(loss_none)
            flag_i = 0
            threshold = _NAN  # flag nothing: no finite score threshold
        else:
            total_loss = best_loss_flagged
            flag_i = best_flag_i + 1
            threshold = _safe_float(s[order][best_flag_i])
        out["expected_loss"] = total_loss / n
        out["expected_loss_total"] = total_loss
        out["expected_loss_threshold"] = threshold
        out["expected_loss_flag_fraction"] = flag_i / n

    return out


def metrics_by_anomaly_type(
    y_true,
    y_type,
    scores,
    k_fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict:
    """Break recall down by anomaly geometry (``global``/``local``/...).

    The generator injects four structurally different anomaly types, but every
    metric so far collapses them into one number. An aggregate PR-AUC of, say,
    0.08 is compatible with "recovers collective spikes perfectly, blind to
    contextual" and with "mediocre at everything" -- and those call for opposite
    fixes. This reports which is which.

    TEORÍA: recall for a type is counted against the **global** top-k, not
    against a ranking recomputed within the type. The alert budget is a single
    shared queue: an analyst reviews the top k rows overall, so the question is
    how many type-t anomalies survive competition with every other row -- not
    how they would rank among themselves. A within-type ranking would report
    near-perfect recall for a type the detector never surfaces.

    Args:
        y_true: 0/1 ground-truth labels.
        y_type: Per-row anomaly type strings, aligned to ``y_true``
            (:func:`src.evaluation.labels.load_ground_truth_types`).
        scores: Anomaly scores, higher = more anomalous.
        k_fractions: Alert-budget fractions to report recall at.

    Returns:
        ``{type: {...}}`` for every type carrying at least one positive, plus an
        ``"__overall__"`` entry. Each holds ``n`` (rows of that type),
        ``n_positive``, ``mean_score_percentile`` (mean percentile rank of the
        type's positives within the full score distribution -- 1.0 means the
        type sits at the very top) and ``recall_at_{k}pct``.
    """
    y = np.asarray(y_true).ravel().astype(int)
    t = np.asarray(y_type, dtype=object).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    n = int(y.size)
    if not (t.size == n and s.size == n):
        raise ValueError(
            f"length mismatch: y_true={n}, y_type={t.size}, scores={s.size}"
        )

    out: dict = {}
    if n == 0:
        return out

    # Percentile rank of every row in the score distribution (1.0 = most
    # anomalous). Ties share the average rank, as elsewhere in this module.
    pct_rank = rankdata(s) / n
    order = _rank_desc(s)
    in_top = {frac: np.zeros(n, dtype=bool) for frac in k_fractions}
    for frac in k_fractions:
        in_top[frac][order[: _top_k(frac, n)]] = True

    positives = y == 1
    observed = [str(v) for v in pd.unique(t[positives])] if positives.any() else []

    def _block(mask_rows: np.ndarray, mask_pos: np.ndarray) -> dict:
        n_pos = int(mask_pos.sum())
        block = {
            "n": float(int(mask_rows.sum())),
            "n_positive": float(n_pos),
            "mean_score_percentile": (
                _safe_float(np.mean(pct_rank[mask_pos])) if n_pos else _NAN
            ),
        }
        for frac in k_fractions:
            caught = int(np.count_nonzero(mask_pos & in_top[frac]))
            block[f"recall_at_{_pct_tag(frac)}"] = (
                _safe_float(caught / n_pos) if n_pos else _NAN
            )
            block[f"n_caught_at_{_pct_tag(frac)}"] = float(caught)
        return block

    for name in sorted(observed):
        is_type = t == name
        out[name] = _block(is_type, is_type & positives)

    out["__overall__"] = _block(np.ones(n, dtype=bool), positives)
    return out


def _to_dense_subsample(X, idx: np.ndarray) -> np.ndarray:
    """Return a dense float array of the selected rows of ``X`` (sparse-safe)."""
    if sp.issparse(X):
        return np.asarray(X[idx].todense(), dtype=float)
    return np.asarray(X, dtype=float)[idx]


def _rank_stability(scores: np.ndarray, sample_size: int, rng: np.random.Generator,
                    n_boot: int = 10) -> float:
    """Rank-stability proxy: mean Spearman correlation under bootstrap jitter.

    True stability of a detector's ranking needs re-fitting the model on
    bootstrap resamples, which is not available at metric-computation time (only
    the final scores are). We therefore approximate scoring variability: fix a
    reference subsample of rows, then in each of ``n_boot`` bootstraps perturb
    those scores with Gaussian jitter scaled to 1% of the score standard
    deviation and re-rank, reporting the mean Spearman rank correlation against
    the un-perturbed reference ranking. Well-separated scores give ~1.0 (a small
    nudge cannot reorder them); heavily tied/flat score distributions -- whose
    top-anomaly ordering is fragile -- score lower. It is a heuristic, documented
    as such.
    """
    n = scores.size
    m = min(sample_size, n)
    if m < 3:
        return _NAN
    ref_idx = rng.choice(n, size=m, replace=False)
    ref_scores = scores[ref_idx]
    std = float(ref_scores.std())
    if std == 0.0:
        return 1.0  # a constant score is (degenerately) perfectly stable
    ref_rank = rankdata(ref_scores)
    corrs = []
    for _ in range(n_boot):
        jitter = rng.normal(0.0, 0.01 * std, size=m)
        boot_rank = rankdata(ref_scores + jitter)
        rho = spearmanr(ref_rank, boot_rank).correlation
        if np.isfinite(rho):
            corrs.append(rho)
    return float(np.mean(corrs)) if corrs else _NAN


def unsupervised_metrics(
    X,
    scores,
    contamination: float = 0.05,
    sample_size: int = 5000,
    random_state: int = 42,
) -> dict:
    """Label-free cluster-quality and rank-stability metrics.

    A binary anomaly/normal split is induced at the top-``contamination`` scores
    (the ``(1 - contamination)`` score quantile is the cut). On a random
    subsample of at most ``sample_size`` rows (the cluster indices are ~O(n^2)):

    * ``silhouette`` -- silhouette score of that 2-cluster labeling (how
      separated the flagged tail is from the bulk in feature space).
    * ``calinski_harabasz`` -- Calinski-Harabasz index of the same labeling.
    * ``rank_stability`` -- see :func:`_rank_stability` (bootstrap-jitter proxy).

    Degenerate single-cluster cases (all-normal or all-anomaly in the subsample,
    or fewer than 2 usable clusters) yield ``NaN`` for the cluster metrics rather
    than raising.

    Returns:
        A ``dict`` of floats: ``silhouette``, ``calinski_harabasz``,
        ``rank_stability``, plus ``contamination``, ``n``, ``n_flagged`` and the
        realized ``subsample_size``.
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = int(s.size)
    rng = np.random.default_rng(random_state)

    out: dict = {"contamination": float(contamination), "n": float(n)}

    if n == 0:
        out.update(silhouette=_NAN, calinski_harabasz=_NAN, rank_stability=_NAN,
                   n_flagged=0.0, subsample_size=0.0)
        return out

    contamination = float(min(max(contamination, 1.0 / n), 0.5))
    threshold = np.quantile(s, 1.0 - contamination)
    labels_full = (s >= threshold).astype(int)
    out["n_flagged"] = float(int(labels_full.sum()))

    # Subsample rows for the O(n^2) cluster indices.
    m = min(sample_size, n)
    sub_idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    sub_labels = labels_full[sub_idx]
    out["subsample_size"] = float(m)

    n_clusters = int(np.unique(sub_labels).size)
    if n_clusters >= 2 and m > n_clusters:
        X_sub = _to_dense_subsample(X, sub_idx)
        # Guard non-finite values that would break the distance computations.
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            out["silhouette"] = _safe_float(silhouette_score(X_sub, sub_labels))
        except (ValueError, RuntimeWarning):
            out["silhouette"] = _NAN
        try:
            out["calinski_harabasz"] = _safe_float(calinski_harabasz_score(X_sub, sub_labels))
        except (ValueError, RuntimeWarning):
            out["calinski_harabasz"] = _NAN
    else:
        # Single cluster in the subsample: cluster indices are undefined.
        out["silhouette"] = _NAN
        out["calinski_harabasz"] = _NAN

    out["rank_stability"] = _rank_stability(s, sample_size, rng)
    return out
