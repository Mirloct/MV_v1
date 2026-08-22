"""Evaluation figures: 2D embeddings + ROC/PR curves + score comparison.

All figures use the non-interactive ``Agg`` backend and are saved UNDER
``reports/figures/evaluation/`` (the hard project-wide figures convention). Large
inputs are subsampled for legible, cheap plots. Every function returns the saved
path(s).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import scipy.sparse as sp

from src.utils import paths
from src.utils.logging_config import setup_logging

__all__ = ["plot_embedding", "plot_roc_pr", "plot_score_comparison"]

_DEFAULT_FIG_DIR = paths.FIGURES_DIR


def _agg_plt():
    """Import matplotlib with the Agg backend and return the pyplot module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _subsample_rows(n: int, max_points: int, random_state: int) -> np.ndarray:
    """Return row indices, randomly subsampled to ``max_points`` if needed."""
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=max_points, replace=False)


def _to_dense(X, idx: np.ndarray) -> np.ndarray:
    if sp.issparse(X):
        Xd = np.asarray(X[idx].todense(), dtype=float)
    else:
        Xd = np.asarray(X, dtype=float)[idx]
    return np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0)


def _reduce_2d(X: np.ndarray, method: str, random_state: int, log) -> tuple[np.ndarray, str]:
    """Reduce ``X`` to 2D via the named method, with a PCA fallback for UMAP."""
    method = (method or "pca").lower()
    if method == "umap":
        try:
            import umap  # type: ignore
            # n_jobs=1 silences UMAP's own warning: a fixed random_state
            # forces single-threaded execution internally regardless, and
            # UMAP warns if n_jobs isn't already set to 1.
            reducer = umap.UMAP(n_components=2, random_state=random_state, n_jobs=1)
            return np.asarray(reducer.fit_transform(X)), "umap"
        except Exception as exc:
            log.warning("UMAP unavailable (%s); falling back to PCA for the embedding", exc)
            method = "pca"
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = float(min(30.0, max(5.0, (X.shape[0] - 1) / 3.0)))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        return np.asarray(reducer.fit_transform(X)), "tsne"
    # PCA (default / fallback).
    from sklearn.decomposition import PCA
    n_comp = 2 if X.shape[1] >= 2 else 1
    coords = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return np.asarray(coords), "pca"


def plot_embedding(
    X,
    scores_or_labels,
    method: str = "pca",
    out_dir: str = _DEFAULT_FIG_DIR,
    y=None,
    max_points: int = 20_000,
    random_state: int = 42,
    filename: Optional[str] = None,
    return_data: bool = False,
):
    """2D embedding scatter coloured by anomaly score (or label).

    Args:
        X: Feature matrix (dense or sparse), row-aligned with ``scores_or_labels``.
        scores_or_labels: Per-row anomaly score or 0/1 label used as the colour.
        method: ``'pca'`` (always available), ``'tsne'`` (sklearn) or ``'umap'``
            (umap-learn; falls back to PCA with a warning if not importable).
        out_dir: Output directory (under ``reports/figures/`` per the project rule).
        y: Optional 0/1 ground-truth labels; drawn as ring markers so true
            anomalies are visible against the score colouring.
        max_points: Subsample cap for the (potentially O(n^2)) reducer.
        random_state: Seed for subsampling and the reducer.
        filename: Optional output filename; defaults to ``embedding_<method>.png``.
        return_data: When True, also return the plotted coordinates so the
            report can re-render this interactively instead of embedding the
            PNG. Off by default so the historical ``-> str`` contract (and
            every existing caller) is untouched.

    Returns:
        The absolute path of the written PNG, or ``(path, data)`` when
        ``return_data`` is set. ``data`` is ``{"x", "y", "color", "labels",
        "method"}`` with plain lists.
    """
    log = setup_logging()
    plt = _agg_plt()

    c = np.asarray(scores_or_labels, dtype=float).ravel()
    n = c.shape[0]
    idx = _subsample_rows(n, max_points, random_state)
    Xd = _to_dense(X, idx)
    c = c[idx]
    y_arr = None if y is None else np.asarray(y).ravel()[idx]

    coords, used = _reduce_2d(Xd, method, random_state, log)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename or f"embedding_{used}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=c, cmap="viridis", s=6, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="anomaly score / label")
    if y_arr is not None and np.any(y_arr == 1):
        pos = y_arr == 1
        ax.scatter(
            coords[pos, 0], coords[pos, 1], facecolors="none", edgecolors="red",
            s=40, linewidths=0.8, label="ground-truth anomaly",
        )
        ax.legend(loc="best")
    ax.set_title(f"{used.upper()} embedding coloured by anomaly score")
    ax.set_xlabel(f"{used}-1")
    ax.set_ylabel(f"{used}-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    log.info("Saved %s embedding (%d points) to %s", used, len(idx), out_path)
    path = os.path.abspath(out_path)
    if not return_data:
        return path
    return path, {
        "x": np.asarray(coords[:, 0], dtype=float).tolist(),
        "y": np.asarray(coords[:, 1], dtype=float).tolist(),
        "color": np.asarray(c, dtype=float).tolist(),
        "labels": None if y_arr is None else np.asarray(y_arr, dtype=int).tolist(),
        "method": used,
    }


def plot_roc_pr(
    y_true,
    scores,
    out_dir: str = _DEFAULT_FIG_DIR,
    filename: str = "roc_pr_curves.png",
) -> Optional[str]:
    """ROC and Precision-Recall curves side by side (needs both label classes).

    Returns the written PNG path, or ``None`` (with a logged warning) when the
    labels contain a single class, in which case the curves are undefined.
    """
    log = setup_logging()
    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(scores, dtype=float).ravel()
    if np.unique(y).size < 2:
        log.warning("plot_roc_pr: only one label class present; skipping ROC/PR plot")
        return None

    plt = _agg_plt()
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = roc_auc_score(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    pr_auc = average_precision_score(y, s)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, color="#4c72b0", label=f"ROC (AUC={roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax1.set_title("ROC curve")
    ax1.set_xlabel("false positive rate")
    ax1.set_ylabel("true positive rate")
    ax1.legend(loc="lower right")

    ax2.plot(rec, prec, color="#c44e52", label=f"PR (AP={pr_auc:.3f})")
    base_rate = float(y.mean())
    ax2.axhline(base_rate, ls="--", color="grey", linewidth=0.8, label=f"base rate={base_rate:.4f}")
    ax2.set_title("Precision-Recall curve")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    log.info("Saved ROC/PR curves to %s (ROC-AUC=%.3f, PR-AUC=%.3f)", out_path, roc_auc, pr_auc)
    return os.path.abspath(out_path)


def plot_score_comparison(
    scores_a,
    scores_b=None,
    names: tuple[str, str] = ("iforest", "vae"),
    out_dir: str = _DEFAULT_FIG_DIR,
    filename: str = "score_comparison.png",
    max_points: int = 200_000,
    random_state: int = 42,
) -> str:
    """Overlay the anomaly-score distributions of one or two detectors.

    Args:
        scores_a: First detector's scores.
        scores_b: Optional second detector's scores (e.g. VAE vs iForest).
        names: Legend labels for the two score vectors.
        out_dir: Output directory (under ``reports/figures/``).
        filename: Output filename.
        max_points: Subsample cap per score vector.
        random_state: Seed for subsampling.

    Returns:
        The absolute path of the written PNG.
    """
    log = setup_logging()
    plt = _agg_plt()
    rng = np.random.default_rng(random_state)

    def _prep(scores):
        s = np.asarray(scores, dtype=float).ravel()
        if s.size > max_points:
            s = s[rng.choice(s.size, size=max_points, replace=False)]
        return s

    a = _prep(scores_a)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(a, bins=60, alpha=0.6, density=True, label=names[0], color="#4c72b0")
    if scores_b is not None:
        b = _prep(scores_b)
        ax.hist(b, bins=60, alpha=0.6, density=True, label=names[1], color="#c44e52")
        ax.legend()
    ax.set_title("Anomaly-score distribution (higher = more anomalous)")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("density")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    log.info("Saved score comparison figure to %s", out_path)
    return os.path.abspath(out_path)
