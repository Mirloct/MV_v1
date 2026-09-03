from __future__ import annotations

from itertools import combinations

import numpy as np


def _top_indices(row: np.ndarray, k: int) -> set[int]:
    order = np.argsort(-row, kind="stable")[: min(k, len(row))]
    return set(order.tolist())


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def top_k_stability(score_runs: np.ndarray, k: int) -> dict[str, object]:
    scores = np.asarray(score_runs, dtype=float)
    if scores.ndim != 2 or scores.shape[0] < 2:
        raise ValueError("score_runs must contain at least two 1-D runs")
    top_sets = [_top_indices(row, k) for row in scores]
    counts = np.zeros(scores.shape[1], dtype=float)
    for selected in top_sets:
        counts[list(selected)] += 1
    pairwise = [_jaccard(a, b) for a, b in combinations(top_sets, 2)]
    return {
        "k": min(k, scores.shape[1]),
        "selection_probability": (counts / scores.shape[0]).tolist(),
        "mean_jaccard": float(np.mean(pairwise)),
        "min_jaccard": float(np.min(pairwise)),
        "runs": int(scores.shape[0]),
    }

