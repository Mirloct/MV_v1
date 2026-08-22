"""Calibrate the anomaly-score cut-off on the validation block.

An anomaly detector produces a *ranking*; turning it into an alert needs a
threshold, and where that threshold comes from decides whether the reported
performance means anything.

**The rule this module exists to enforce: the threshold is computed on
validation scores and only ever applied to test.** Choosing the cut-off on the
same rows you then report on is the threshold-fitting form of data leakage --
you would be reporting the best cut-off *in hindsight*, which no deployed system
can have.

Two methods:

* ``"percentile"`` -- flag the top ``100 - percentile`` % of validation scores.
  A business rule: simple, transparent, and it fixes the alert volume rather
  than the false-alarm risk.
* ``"pot"`` (Peaks-Over-Threshold, the default) -- fit a Generalized Pareto
  distribution to the validation scores that exceed a high quantile, then invert
  it for a target exceedance probability.

TEORÍA (why POT): the Pickands-Balkema-de Haan theorem says that for a broad
class of distributions the conditional excess ``X - u | X > u`` converges to a
Generalized Pareto as the threshold ``u`` grows. So instead of assuming the
score distribution's shape everywhere, POT models only its *tail* -- which is
the only part a threshold interacts with -- and lets the data pick the tail
index. That buys two things a raw percentile cannot:

1. **Extrapolation past the data.** A p99 estimated from 1,600 validation rows
   is the 16th largest observation, and there is no p99.9 at all. The fitted GPD
   yields a principled quantile beyond the observed range.
2. **A false-alarm budget instead of an alert budget.** You state "1 false alarm
   per 1,000 rows" and get the cut-off, rather than stating "flag 1 %" and
   discovering the risk after the fact.

The survival function above ``u`` is

    P(X > x) = (Nu / N) * [1 + xi * (x - u) / sigma] ** (-1 / xi)

so setting it to the target probability ``p`` and solving for ``x`` gives

    x = u + (sigma / xi) * [ (p * N / Nu) ** (-xi) - 1 ]        (xi != 0)
    x = u + sigma * ln( Nu / (p * N) )                          (xi -> 0)

Degenerate inputs (too few exceedances, a failed fit, a non-finite result) fall
back to the plain percentile and say so in the returned dict and the log -- a
silently wrong threshold is worse than an honest simple one.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from src.utils.logging_config import setup_logging

__all__ = ["calibrate_threshold", "apply_threshold", "THRESHOLD_METHODS"]

THRESHOLD_METHODS: tuple[str, ...] = ("pot", "percentile")

# Minimum exceedances worth fitting a 2-parameter GPD to. Below this the shape
# parameter is noise and the extrapolation is not credible.
_MIN_EXCEEDANCES = 30


def _percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(scores, float(percentile)))


def calibrate_threshold(
    scores,
    method: str = "pot",
    percentile: float = 99.0,
    tail_percentile: float = 95.0,
    target_far: float = 1e-3,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Return the alert cut-off calibrated on ``scores`` (the validation block).

    Args:
        scores: Validation anomaly scores (higher = more anomalous).
        method: ``"pot"`` (default) or ``"percentile"``.
        percentile: Cut-off percentile for ``method="percentile"``, and the
            fallback whenever the POT fit is not usable.
        tail_percentile: Quantile defining the POT threshold ``u``; only scores
            above it enter the GPD fit.
        target_far: Target exceedance probability for POT -- the false-alarm
            rate you are willing to accept (``1e-3`` = 1 in 1,000 rows).
        logger: Optional logger; defaults to the project logger.

    Returns:
        A ``dict`` with ``threshold``, ``method`` (the method actually used),
        ``requested_method``, ``n``, ``n_flagged``, ``flagged_rate``, plus
        ``u``/``xi``/``sigma``/``n_exceedances``/``target_far`` for POT and
        ``fallback_reason`` when POT degraded to the percentile.
    """
    log = logger or setup_logging()
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = int(s.size)

    requested = str(method).lower()
    if requested not in THRESHOLD_METHODS:
        raise ValueError(
            f"Unknown threshold method {method!r}; choose from {THRESHOLD_METHODS}"
        )

    out: dict = {
        "requested_method": requested,
        "method": requested,
        "n": float(n),
        "percentile": float(percentile),
    }
    if n == 0:
        out.update(threshold=float("nan"), method="percentile", n_flagged=0.0,
                   flagged_rate=float("nan"), fallback_reason="no finite scores")
        log.warning("Threshold calibration got no finite scores; threshold is NaN.")
        return out

    threshold = _percentile_threshold(s, percentile)

    if requested == "pot":
        u = _percentile_threshold(s, tail_percentile)
        exceedances = s[s > u] - u
        n_exc = int(exceedances.size)
        out.update(u=float(u), n_exceedances=float(n_exc), target_far=float(target_far),
                   tail_percentile=float(tail_percentile))

        reason = None
        if n_exc < _MIN_EXCEEDANCES:
            reason = f"only {n_exc} exceedances above p{tail_percentile:g} (need {_MIN_EXCEEDANCES})"
        else:
            try:
                from scipy.stats import genpareto

                # floc=0: the excesses are measured from u by construction, so
                # the location is known and must not be re-estimated.
                xi, _, sigma = genpareto.fit(exceedances, floc=0.0)
                if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
                    reason = f"degenerate GPD fit (xi={xi}, sigma={sigma})"
                else:
                    ratio = target_far * n / n_exc
                    if ratio <= 0:
                        reason = "non-positive target exceedance probability"
                    else:
                        if abs(xi) < 1e-8:
                            pot_threshold = u + sigma * np.log(1.0 / ratio)
                        else:
                            pot_threshold = u + (sigma / xi) * (ratio ** (-xi) - 1.0)
                        if not np.isfinite(pot_threshold):
                            reason = "non-finite POT quantile"
                        else:
                            threshold = float(pot_threshold)
                            out.update(xi=float(xi), sigma=float(sigma))
            except Exception as exc:  # pragma: no cover - scipy edge cases
                reason = f"GPD fit failed ({exc})"

        if reason is not None:
            out["method"] = "percentile"
            out["fallback_reason"] = reason
            log.warning(
                "POT calibration unusable (%s); falling back to the p%.6g percentile.",
                reason, percentile,
            )

    n_flagged = int((s >= threshold).sum())
    out.update(
        threshold=float(threshold),
        n_flagged=float(n_flagged),
        flagged_rate=float(n_flagged) / n,
    )
    log.info(
        "Threshold calibrated on %d validation scores via %s: threshold=%.6f "
        "-> %d flagged (%.3f%% of validation)",
        n, out["method"], out["threshold"], n_flagged, 100.0 * out["flagged_rate"],
    )
    return out


def apply_threshold(scores, threshold: float) -> np.ndarray:
    """Binary alert flags (``1`` = alert) for ``scores`` at ``threshold``."""
    s = np.asarray(scores, dtype=float).ravel()
    return (s >= float(threshold)).astype(int)
