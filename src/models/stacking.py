"""Feed one detector's anomaly score into another as an extra feature (IF -> VAE).

The two detectors in this project normally run **in parallel** on the same
matrix. This module implements the alternative *stacked* arrangement: the
Isolation Forest runs first, and its per-row anomaly score is appended to the
feature matrix that trains the VAE, so the VAE learns what "normal" looks like
*including* how the forest scores it.

Pipeline (mapping the usual fraud-detection terminology onto this project's
chronological blocks):

===================  =========================  ===================================
Stacking step        Here                       Note
===================  =========================  ===================================
train IF             ``X[train_mask]``          one model, fitted once
score all sets       ``X`` (all rows)           the *same* model scores every block
append as a column   ``build_stacked_matrix``   identical column order everywhere
re-standardise       ``StandardScaler``         fitted on the train block only
train VAE            ``input_dim = n + 1``      picked up automatically from shape
===================  =========================  ===================================

Why the score is appended rather than the models being averaged: the VAE's job
is to model the normal manifold, and "how isolated a row looks to a forest" is a
genuine coordinate of that manifold. A row that is mildly unusual in every raw
feature *and* carries a high forest score is more suspicious than either signal
alone suggests, and only a model that sees both can express that interaction.

.. warning::
   **The appended score is in-sample on the rows the forest was fitted on.**
   A forest scores its own training data differently from unseen data, so the
   feature's distribution is not identical across blocks; the VAE calibrates its
   reconstruction of that column on the training distribution and meets a
   slightly shifted one at test time. The principled fix is out-of-fold
   (cross-fitted) scores, which requires fitting the forest more than once and
   is therefore excluded by the "one model, reused across the three sets"
   constraint. :func:`score_shift_report` quantifies the resulting shift so the
   cost of that trade-off is measured rather than assumed.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import setup_logging

__all__ = [
    "build_stacked_matrix",
    "StackedMatrix",
    "score_shift_report",
    "DEFAULT_SCORE_FEATURE",
]

DEFAULT_SCORE_FEATURE = "iforest_score"


class StackedMatrix:
    """The augmented matrix plus everything needed to reproduce it later."""

    __slots__ = ("X", "feature_names", "scaler", "score_name", "n_original")

    def __init__(self, X, feature_names: list, scaler, score_name: str, n_original: int):
        self.X = X
        self.feature_names = feature_names
        self.scaler = scaler
        self.score_name = score_name
        self.n_original = n_original

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"StackedMatrix(shape={self.X.shape}, n_original={self.n_original}, "
            f"score_name={self.score_name!r})"
        )


def build_stacked_matrix(
    X,
    detector_scores,
    fit_mask: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    score_name: str = DEFAULT_SCORE_FEATURE,
    standardise: bool = True,
    logger: Optional[logging.Logger] = None,
) -> StackedMatrix:
    """Append ``detector_scores`` to ``X`` as one extra column and re-standardise.

    Args:
        X: The preprocessed feature matrix (dense ndarray or scipy sparse), with
            all rows -- train, validation and test -- in their original order.
        detector_scores: One score per row of ``X``, aligned row-for-row, in the
            project convention **higher = more anomalous**.
        fit_mask: Boolean row mask marking the training block. The scaler is
            fitted on these rows only; ``None`` fits on everything (leaky --
            only for callers with no split).
        feature_names: Names of ``X``'s columns; ``score_name`` is appended.
        score_name: Name of the new column.
        standardise: Fit a :class:`~sklearn.preprocessing.StandardScaler` on the
            augmented training block and apply it to every row.
        logger: Optional logger; defaults to the project logger.

    Returns:
        A :class:`StackedMatrix`. The column order is fixed by construction --
        the score is always last -- so train, validation and test are guaranteed
        to line up.

    Raises:
        ValueError: If the score length does not match ``X``'s row count.
    """
    log = logger or setup_logging()
    scores = np.asarray(detector_scores, dtype=float).ravel()
    n_rows = X.shape[0]
    if scores.shape[0] != n_rows:
        raise ValueError(
            f"detector_scores has {scores.shape[0]} entries but X has {n_rows} rows"
        )
    if not np.isfinite(scores).all():
        n_bad = int((~np.isfinite(scores)).sum())
        log.warning(
            "%d non-finite detector score(s) replaced by the finite median before "
            "stacking", n_bad,
        )
        finite = scores[np.isfinite(scores)]
        scores = np.nan_to_num(
            scores, nan=float(np.median(finite)) if finite.size else 0.0,
            posinf=float(finite.max()) if finite.size else 0.0,
            neginf=float(finite.min()) if finite.size else 0.0,
        )

    n_original = int(X.shape[1])
    col = scores.reshape(-1, 1)

    # Sparse input keeps a sparse result; the score column is dense but a single
    # column costs one dense vector, not a densified matrix.
    if sp.issparse(X):
        X_aug = sp.hstack([X.tocsr(), sp.csr_matrix(col)], format="csr")
    else:
        X_aug = np.hstack([np.asarray(X, dtype=float), col])

    scaler = None
    if standardise:
        if fit_mask is None:
            fit_rows = X_aug
            log.warning(
                "build_stacked_matrix: no fit_mask given, the scaler is fitted on "
                "ALL rows -- this leaks validation/test statistics into training."
            )
        else:
            mask = np.asarray(fit_mask, dtype=bool).ravel()
            if mask.shape[0] != n_rows:
                raise ValueError(
                    f"fit_mask has {mask.shape[0]} entries but X has {n_rows} rows"
                )
            if not mask.any():
                raise ValueError("fit_mask selects no rows")
            fit_rows = X_aug[mask]
        # with_mean=False keeps a sparse matrix sparse (centring would fill it).
        scaler = StandardScaler(with_mean=not sp.issparse(X_aug))
        scaler.fit(fit_rows)
        X_aug = scaler.transform(X_aug)

    names = list(feature_names) if feature_names is not None else [
        f"f{i}" for i in range(n_original)
    ]
    names.append(score_name)

    log.info(
        "Stacked matrix: %d x %d -> %d x %d (appended %r as the last column; "
        "scaler fitted on %s)",
        n_rows, n_original, X_aug.shape[0], X_aug.shape[1], score_name,
        "all rows" if fit_mask is None else f"{int(np.asarray(fit_mask).sum())} train rows",
    )
    return StackedMatrix(
        X=X_aug, feature_names=names, scaler=scaler,
        score_name=score_name, n_original=n_original,
    )


def score_shift_report(
    detector_scores,
    fit_mask: np.ndarray,
    eval_masks: dict,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Quantify how much the stacked score's distribution moves across blocks.

    The stacked feature is in-sample on the rows the forest was fitted on. This
    reports, per block, the mean/std of the score and a standardised gap versus
    the fit block::

        shift = (mean_block - mean_fit) / std_fit

    A shift of a fraction of a standard deviation is the ordinary train/test
    difference; a shift of one or more means the VAE is being asked to
    reconstruct a column whose location it never saw, and the resulting
    reconstruction error is an artefact of the stacking rather than a signal.

    Returns:
        ``{block_name: {"mean", "std", "shift"}}``, plus ``"fit"`` for the
        reference block.
    """
    log = logger or setup_logging()
    s = np.asarray(detector_scores, dtype=float).ravel()
    fit = np.asarray(fit_mask, dtype=bool).ravel()
    fit_scores = s[fit]
    mu, sd = float(np.mean(fit_scores)), float(np.std(fit_scores))
    out = {"fit": {"mean": mu, "std": sd, "shift": 0.0, "n": float(fit.sum())}}

    for name, mask in eval_masks.items():
        m = np.asarray(mask, dtype=bool).ravel()
        if not m.any():
            continue
        block = s[m]
        block_mu = float(np.mean(block))
        out[name] = {
            "mean": block_mu,
            "std": float(np.std(block)),
            "shift": (block_mu - mu) / sd if sd > 0 else float("nan"),
            "n": float(m.sum()),
        }

    log.info(
        "Stacked-score distribution shift vs the fit block: %s",
        {k: f"{v['shift']:+.3f} sd" for k, v in out.items() if k != "fit"},
    )
    worst = max(
        (abs(v["shift"]) for k, v in out.items() if k != "fit" and np.isfinite(v["shift"])),
        default=0.0,
    )
    if worst >= 1.0:
        log.warning(
            "The stacked score shifts by %.2f sd between blocks. The VAE will "
            "reconstruct that column badly on every row of the shifted block, "
            "not just the anomalous ones -- consider out-of-fold scores.", worst,
        )
    return out
