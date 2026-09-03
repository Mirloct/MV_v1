from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _stable_rank(frame: pd.DataFrame, score_col: str, id_col: str | None) -> pd.DataFrame:
    columns = [score_col]
    ascending = [False]
    if id_col and id_col in frame:
        columns.append(id_col)
        ascending.append(True)
    return frame.sort_values(columns, ascending=ascending, kind="mergesort")


def _metric_row(
    ranked: pd.DataFrame,
    label_col: str,
    score_name: str,
    k: int,
    average_precision: float,
) -> dict[str, float | int | str]:
    effective_k = min(k, len(ranked))
    labels = ranked[label_col].to_numpy(dtype=int)
    positives = int(labels.sum())
    true_positives = int(labels[:effective_k].sum())
    prevalence = positives / len(ranked) if len(ranked) else np.nan
    precision = true_positives / effective_k if effective_k else np.nan
    recall = true_positives / positives if positives else np.nan
    lift = precision / prevalence if prevalence else np.nan
    return {
        "score_name": score_name,
        "k": effective_k,
        "alerts_requested": k,
        "true_positives": true_positives,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
        "average_precision": average_precision,
    }


def evaluate_scores(
    frame: pd.DataFrame,
    label_col: str,
    score_columns: list[str],
    budgets: list[int],
    id_col: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    labels = frame[label_col].to_numpy(dtype=int)
    for score_col in score_columns:
        scores = frame[score_col].to_numpy(dtype=float)
        ap = float(average_precision_score(labels, scores)) if labels.sum() else np.nan
        ranked = _stable_rank(frame, score_col, id_col)
        rows.extend(_metric_row(ranked, label_col, score_col, k, ap) for k in budgets)
    return pd.DataFrame(rows)


def evaluate_family_scores(
    frame: pd.DataFrame,
    label_col: str,
    family_col: str,
    score_columns: list[str],
    budgets: list[int],
    id_col: str | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    positive = frame[label_col].astype(int) == 1
    families = frame.loc[positive, family_col].dropna().unique().tolist()
    for family in families:
        cohort_mask = ~positive | (frame[family_col] == family)
        cohort = frame.loc[cohort_mask]
        table = evaluate_scores(cohort, label_col, score_columns, budgets, id_col)
        table.insert(0, "cohort_positives", int(cohort[label_col].sum()))
        table.insert(0, "cohort_rows", len(cohort))
        table.insert(0, "group_value", str(family))
        table.insert(0, "group_column", family_col)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _top_mask(scores: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    order = np.argsort(-np.asarray(scores), kind="stable")[: min(k, len(scores))]
    mask[order] = True
    return mask


def positive_coverage(
    labels: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, k: int
) -> dict[str, int]:
    positive = np.asarray(labels, dtype=int) == 1
    top_a = _top_mask(score_a, k)
    top_b = _top_mask(score_b, k)
    return {
        "k": min(k, len(positive)),
        "shared_positive_hits": int(np.sum(positive & top_a & top_b)),
        "a_only_positive_hits": int(np.sum(positive & top_a & ~top_b)),
        "b_only_positive_hits": int(np.sum(positive & ~top_a & top_b)),
        "union_positive_hits": int(np.sum(positive & (top_a | top_b))),
    }
