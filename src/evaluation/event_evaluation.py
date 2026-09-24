"""Evaluate scores against reviewed event labels, with the denominators the
temporal-supervision framework asks for.

AUROC alone never decides (Saito & Rehmsmeier 2015): with a rare positive class
the headline numbers are average precision and the operational ``@K`` figures
(precision, recall, episode recall, lift, false positives). Every result carries
the counts behind it, a **cluster bootstrap by entity** interval (rows of one
entity are not independent), and an explicit ``conclusive`` flag: a window with
fewer than ``min_positive_episodes`` positive episodes is *descriptive only* and
must not decide which model wins.

Only rows that are known **and mature** enter (``EventLabels.usable``); unknown
and immature rows are excluded, never counted as negatives.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.progress import track

__all__ = ["calibration_summary", "evaluate_ranking"]

#: Share of eligible rows reviewed when no operational capacity K is configured.
DEFAULT_CAPACITY_SHARE = 0.05


def _topk_stats(y: np.ndarray, scores: np.ndarray, k: int) -> tuple[int, np.ndarray]:
    """True positives among the ``k`` highest scores (stable tie-break) and the
    positions of those ``k`` rows."""
    order = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return int(y[order].sum()), order


def _safe(fn, *args) -> Optional[float]:
    try:
        value = float(fn(*args))
    except ValueError:  # single class, empty, ...
        return None
    return value if math.isfinite(value) else None


def _bootstrap(y, scores, entity, k_share, n_boot, seed) -> dict[str, Optional[list[float]]]:
    """95% percentile intervals for AP and precision@K, resampling ENTITIES."""
    if n_boot <= 0:
        return {"ap": None, "precision_at_k": None}
    groups = pd.Series(np.arange(len(y))).groupby(entity).apply(lambda s: s.to_numpy())
    members = list(groups.to_numpy())
    if len(members) < 2:
        return {"ap": None, "precision_at_k": None}
    rng = np.random.default_rng(seed)
    aps, precs = [], []
    for _ in track(range(n_boot), desc="event_evaluation[bootstrap]", unit="rep"):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[i] for i in pick])
        yb, sb = y[idx], scores[idx]
        if yb.sum() == 0:
            continue
        aps.append(average_precision_score(yb, sb))
        kb = max(1, int(round(k_share * len(idx))))
        tp, _ = _topk_stats(yb, sb, kb)
        precs.append(tp / kb)
    if len(aps) < max(10, n_boot // 5):
        return {"ap": None, "precision_at_k": None}
    interval = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"ap": interval(aps), "precision_at_k": interval(precs)}


def evaluate_ranking(
    scores: np.ndarray,
    y: np.ndarray,
    eligible: np.ndarray,
    entity: np.ndarray,
    episode: np.ndarray,
    *,
    k: Optional[int] = None,
    n_boot: int = 100,
    seed: int = 42,
    min_positive_episodes: int = 20,
) -> dict[str, Any]:
    """Score quality on ``eligible`` rows (see module docstring).

    Args:
        scores: Higher = more anomalous, aligned with the panel rows.
        y: 0/1 target (``NaN`` allowed where not eligible).
        eligible: Rows that may be evaluated (known, mature, inside the window).
        entity: Cluster id per row, for the bootstrap.
        episode: Episode id per row (``""`` where none); positives are counted as
            distinct episodes, not rows.
        k: Operational review capacity in rows; default 5% of the eligible rows.
    """
    mask = np.asarray(eligible, dtype=bool) & np.isfinite(scores) & ~np.isnan(np.asarray(y, dtype=float))
    n = int(mask.sum())
    out: dict[str, Any] = {"eligible_rows": n}
    if n == 0:
        return {**out, "status": "no_eligible_rows", "conclusive": False}
    s, yy = np.asarray(scores, dtype=float)[mask], np.asarray(y, dtype=float)[mask]
    ent, epi = np.asarray(entity)[mask], np.asarray(episode)[mask]
    n_pos = int(yy.sum())
    positive_episodes = int(len(np.unique(epi[(yy == 1) & (epi != "")]))) if n_pos else 0
    out.update(n_positive_rows=n_pos, n_positive_episodes=positive_episodes,
               n_positive_entities=int(len(np.unique(ent[yy == 1]))) if n_pos else 0,
               event_rate_row=n_pos / n)
    if n_pos == 0:
        return {**out, "status": "no_positives", "conclusive": False,
                "note": "Ventana sin positivos maduros: no hay desempeño que medir."}
    k_source = "configurado" if k else f"{DEFAULT_CAPACITY_SHARE:.0%} de las filas elegibles (supuesto)"
    kk = min(int(k) if k else max(1, math.ceil(DEFAULT_CAPACITY_SHARE * n)), n)
    tp, top = _topk_stats(yy, s, kk)
    hit_episodes = len(np.unique(epi[top][(yy[top] == 1) & (epi[top] != "")]))
    prevalence = n_pos / n
    out.update(
        status="ok", k=kk, k_source=k_source,
        ap=_safe(average_precision_score, yy, s), roc_auc=_safe(roc_auc_score, yy, s),
        precision_at_k=tp / kk, recall_at_k=tp / n_pos,
        episode_recall_at_k=(hit_episodes / positive_episodes) if positive_episodes else None,
        lift_at_k=(tp / kk) / prevalence, false_positives_at_k=kk - tp,
        fp_per_1000_eligible=1000.0 * (kk - tp) / n,
        ap_baseline=prevalence,
        ci=_bootstrap(yy, s, ent, kk / n, n_boot, seed),
    )
    out["conclusive"] = positive_episodes >= min_positive_episodes
    if not out["conclusive"]:
        out["note"] = (f"Solo descriptivo: {positive_episodes} episodio(s) positivo(s) "
                       f"(< {min_positive_episodes}); no decide qué modelo gana.")
    return out


def calibration_summary(prob: np.ndarray, y: np.ndarray, min_positives: int = 20) -> dict[str, Optional[float]]:
    """Brier, log-loss and calibration intercept/slope of a *probability*.

    Slope and intercept need enough events to mean anything; below
    ``min_positives`` they are ``None`` (a calibrator on a handful of events only
    overfits).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss

    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    yy = np.asarray(y, dtype=float)
    out: dict[str, Optional[float]] = {
        "brier": _safe(brier_score_loss, yy, p),
        "log_loss": _safe(lambda a, b: log_loss(a, b, labels=[0, 1]), yy, p),
        "calibration_intercept": None, "calibration_slope": None,
    }
    if yy.sum() >= min_positives and (yy == 0).sum() >= min_positives:
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, yy)
        out["calibration_intercept"] = float(fit.intercept_[0])
        out["calibration_slope"] = float(fit.coef_[0][0])
    return out
