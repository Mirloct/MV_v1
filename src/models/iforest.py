"""Isolation Forest anomaly detector with Optuna tuning and crash recovery.

This module wraps scikit-learn's :class:`~sklearn.ensemble.IsolationForest`
in a thin, project-consistent detector and provides an Optuna tuning routine
that is resilient to crashes: the study lives in a persistent SQLite RDB and
the best-so-far hyperparameters are checkpointed to YAML after every trial.

Design boundary
---------------
The detector is deliberately decoupled from the data / out-of-time (OOT)
logic. It consumes an already-preprocessed feature matrix ``X`` (a dense
:class:`numpy.ndarray` **or** a :mod:`scipy.sparse` CSR matrix, exactly as
produced by :func:`src.preprocessing.pipeline.fit_transform_panel`) and
returns a per-row anomaly score. The OOT split and the join back to the
separate ground-truth file are the evaluation module's responsibility, not
this module's. Here we only ``fit`` on a given ``X_train`` and ``score`` any
``X``.

Algorithm note (Liu, Ting & Zhou, 2008)
---------------------------------------
Isolation Forest isolates points with random axis-parallel splits; anomalies
are isolated with *shorter* average path lengths. Each isolation tree is grown
to an implicit height limit of ``ceil(log2(max_samples))`` (the expected depth
of a balanced binary tree over the sub-sample), because beyond that depth only
the normal bulk remains and extra depth adds no isolation signal. Raw path
lengths are normalized by ``c(n)``, the average path length of an unsuccessful
BST search over ``n`` points, so that the anomaly score ``s = 2**(-E[h]/c(n))``
is comparable across sub-sample sizes. scikit-learn implements exactly this;
we only fix the *sign* of the exposed score (see the convention below).

Score-sign convention
----------------------
Throughout this project the anomaly score follows **higher = more anomalous**.
scikit-learn's ``score_samples`` uses the opposite sign (higher = more normal)
and its ``decision_function`` is that value shifted by the contamination
offset (negative = predicted outlier). :meth:`IsolationForestDetector.score_samples`
returns ``-clf.score_samples(X)`` so the exposed score increases with
anomalousness; :meth:`decision_function` is passed through unchanged (raw
scikit-learn semantics: negative = outlier) for callers that want the
threshold-centered value.

Data sources / inputs: preprocessed dense/CSR feature matrices, optional
reviewed labels and temporal validation masks; writes Optuna SQLite, YAML and
joblib artifacts under the configured paths.
Created: 2026-08-22
Last modified: 2026-09-26
Changelog:
- 2026-09-25: Bound resumed studies to complete matrix/label contents and made
  a study with no completed trials fail explicitly instead of exposing stale outputs.
- 2026-09-26: Fixed the production operating point at contamination 0.005;
  tuning remains restricted to max_samples and max_features.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Optional, Sequence, Union

import joblib
import numpy as np
import scipy.sparse as sp
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils import paths
from src.utils.atomic_io import atomic_replace
from src.utils.logging_config import log_phase, setup_logging
from src.utils.progress import Bar, show_best_trial as _show_best_trial

__all__ = [
    "IsolationForestDetector",
    "tune_iforest",
    "plot_score_distribution",
]

# Fraction of entities (or rows) held out from every trial's fit so the
# objective is never scored in-sample. See `_blocked_split`.
_DEFAULT_HOLDOUT_FRAC = 0.3

# Production operating point is a review-capacity decision, never a search
# dimension. Alternative values remain descriptive score-cut sweeps only.
_DEFAULT_CONTAMINATION = 0.005

# Tuner design constants (see `tune_iforest`). psi = rows per tree, ABSOLUTE.
_DEFAULT_N_ESTIMATORS = 300            # fixed: searching it only inflated the old objective
_DEFAULT_PSI_RANGE = (1024, 32768)     # absolute max_samples range, capped to the fit rows
_DEFAULT_MF_RANGE = (0.5, 1.0)
_REFERENCE_MAX_SAMPLES = 256           # Liu et al. (2008) paper default = the reference
_PSI_ANCHORS = (1024, 4096, 16384, 32768)
_DEFAULT_TAIL_FRACS = (0.005, 0.01, 0.02)
_DEFAULT_NOISE_SEEDS = 3
_MIN_EVAL_POSITIVES = 10

# Label-free objective names accepted by `objective_metric`. Passing one of
# these selects it even when labels are available, so an unlabelled deployment
# can be rehearsed on a labelled dataset.
_UNSUPERVISED_METRICS: tuple[str, ...] = ("rank_agreement", "tail_separation")

# Default on-disk locations. All artifact paths live in `src.utils.paths`, the
# single place that knows the `artifacts/` layout.
_DEFAULT_STORAGE_DB = paths.IFOREST_STUDY_DB
_DEFAULT_BEST_PARAMS = paths.IFOREST_BEST_PARAMS
_DEFAULT_MODEL_OUT = paths.IFOREST_MODEL
_DEFAULT_FIG_DIR = paths.FIGURES_DIR

ArrayLike = Union[np.ndarray, "sp.spmatrix"]


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _n_samples_features(X: ArrayLike) -> tuple[int, int]:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        arr = np.asarray(X)
        return arr.shape[0], (arr.shape[1] if arr.ndim == 2 else 1)
    return int(shape[0]), int(shape[1])


# --------------------------------------------------------------------------- #
# Detector                                                                    #
# --------------------------------------------------------------------------- #
class IsolationForestDetector:
    """A thin, project-consistent wrapper around ``sklearn.IsolationForest``.

    The wrapper standardises the anomaly-score sign (higher = more anomalous),
    logs fit time and matrix shape via :func:`log_phase`, transparently accepts
    sparse (CSR) or dense input, and persists via joblib. It is intentionally
    stateless beyond the fitted scikit-learn estimator so it stays picklable.
    """

    def __init__(
        self,
        n_estimators: int = _DEFAULT_N_ESTIMATORS,
        max_samples: Union[str, int, float] = "auto",
        max_features: float = 1.0,
        contamination: Union[str, float] = "auto",
        bootstrap: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: Optional[IsolationForest] = None

    # -- construction ------------------------------------------------------- #
    def _build(self) -> IsolationForest:
        return IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    @staticmethod
    def _as_model_input(X: ArrayLike) -> ArrayLike:
        """Coerce input to a scikit-learn-friendly form (CSR if sparse)."""
        if sp.issparse(X):
            return X.tocsr()
        return np.asarray(X)

    # -- fit ---------------------------------------------------------------- #
    def fit(self, X: ArrayLike) -> "IsolationForestDetector":
        """Fit the underlying Isolation Forest on ``X`` and return ``self``."""
        log = setup_logging()
        Xm = self._as_model_input(X)
        n_samples, n_features = _n_samples_features(Xm)
        with log_phase("iforest.fit", log):
            log.info(
                "Fitting IsolationForest on %d samples x %d features "
                "(n_estimators=%s, max_samples=%s, max_features=%s, "
                "contamination=%s, bootstrap=%s)",
                n_samples, n_features, self.n_estimators, self.max_samples,
                self.max_features, self.contamination, self.bootstrap,
            )
            self.model_ = self._build()
            self.model_.fit(Xm)
        return self

    def _check_fitted(self) -> IsolationForest:
        if self.model_ is None:
            raise RuntimeError("IsolationForestDetector is not fitted; call fit(X) first.")
        return self.model_

    # -- scoring ------------------------------------------------------------ #
    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Return anomaly scores where **higher = more anomalous**.

        This is ``-sklearn.score_samples`` so that the returned value grows
        with the likelihood of a point being an anomaly (the project-wide
        convention).
        """
        model = self._check_fitted()
        return -model.score_samples(self._as_model_input(X))

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Raw scikit-learn ``decision_function`` (negative = predicted outlier).

        Kept in scikit-learn's native sign/centering (shifted by the
        contamination offset) for callers that want the threshold-centered
        value. For the project-standard "higher = more anomalous" score use
        :meth:`score_samples`.
        """
        model = self._check_fitted()
        return model.decision_function(self._as_model_input(X))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Binary anomaly flag: ``1`` = anomaly, ``0`` = normal.

        A convenience passthrough over scikit-learn's ``predict`` (which
        returns ``-1`` for outliers and ``+1`` for inliers), remapped to the
        anomaly-positive ``1``/``0`` convention used across the project.
        """
        model = self._check_fitted()
        raw = model.predict(self._as_model_input(X))
        return (raw == -1).astype(int)

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str = _DEFAULT_MODEL_OUT) -> str:
        """Serialize the detector with joblib. Returns the written path."""
        self._check_fitted()
        _ensure_parent_dir(path)
        joblib.dump(self, path)
        setup_logging().info("Saved IsolationForestDetector to %s", path)
        return path

    @classmethod
    def load(cls, path: str = _DEFAULT_MODEL_OUT) -> "IsolationForestDetector":
        """Load a detector previously written by :meth:`save`."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain an {cls.__name__}")
        return obj


# --------------------------------------------------------------------------- #
# Optuna tuning helpers                                                        #
# --------------------------------------------------------------------------- #
def _default_storage_uri(db_path: str = _DEFAULT_STORAGE_DB) -> str:
    """Build a SQLite RDBStorage URI, creating the parent directory.

    SQLAlchemy needs forward slashes in the SQLite URL, so backslashes from
    Windows paths are normalised.
    """
    _ensure_parent_dir(db_path)
    return "sqlite:///" + db_path.replace("\\", "/")


def _detector_kwargs_from_params(
    params: dict, contamination: float = _DEFAULT_CONTAMINATION
) -> dict:
    """Translate an Optuna trial's params into ``IsolationForestDetector`` kwargs.

    ``max_samples`` is stored across two params (``max_samples_mode`` plus, when
    the mode is ``float``, ``max_samples``) so both ``'auto'`` and a numeric
    fraction can be explored; here they are collapsed back to a single value.

    ``contamination`` is **not** a searched parameter -- it is supplied by the
    caller (see :func:`tune_iforest`). A value carried in an older ``params``
    dict is still honoured so YAML checkpoints written before that change keep
    loading.

    TEORÍA: sklearn's ``IsolationForest.score_samples`` does not consult
    ``offset_``; contamination only shifts the threshold used by
    ``decision_function`` / ``predict``. Every objective in this module ranks
    rows by ``score_samples``, and PR-AUC, ROC-AUC and Spearman agreement are
    all rank-based -- so contamination is mathematically incapable of changing
    an objective value. Searching it burns TPE budget on a dimension with a flat
    response surface and persists an arbitrary "best" value.
    """
    if "max_samples_mode" in params:  # legacy layout (studies/YAML written before the redesign)
        ms_mode = params["max_samples_mode"]
        if ms_mode == "auto":
            max_samples: Union[str, int, float] = "auto"
        elif ms_mode == "int":
            max_samples = int(params.get("max_samples_int", 256))
        else:
            max_samples = float(params.get("max_samples", 1.0))
    else:  # current layout: max_samples is an absolute integer (rows per tree)
        max_samples = int(params["max_samples"])
    return {
        "n_estimators": int(params.get("n_estimators", _DEFAULT_N_ESTIMATORS)),
        "max_samples": max_samples,
        "max_features": float(params["max_features"]),
        "contamination": float(params.get("contamination", contamination)),
        "bootstrap": bool(params.get("bootstrap", False)),
    }


def _blocked_split(
    n_samples: int,
    groups: Optional[np.ndarray],
    holdout_frac: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into ``(fit_idx, eval_idx)``, blocking on ``groups``.

    TEORÍA: in a panel, all rows of one entity share the latent level that
    generated them (a per-entity balance/transaction scale). A plain row-wise
    split therefore leaks: the model sees months 1-5 of an entity while being
    scored on month 6 of the *same* entity, and an objective computed that way
    rewards memorising entity levels rather than isolating anomalies. Splitting
    whole entities makes the held-out block genuinely unseen.
    """
    rng = np.random.default_rng(random_state)
    if groups is None:
        idx = rng.permutation(n_samples)
        n_eval = max(1, int(round(holdout_frac * n_samples)))
        n_eval = min(n_eval, n_samples - 1)
        return np.sort(idx[n_eval:]), np.sort(idx[:n_eval])

    groups = np.asarray(groups).ravel()
    if groups.shape[0] != n_samples:
        raise ValueError(
            f"groups has {groups.shape[0]} entries but X has {n_samples} rows"
        )
    uniq = np.unique(groups)
    if uniq.size < 2:
        return _blocked_split(n_samples, None, holdout_frac, random_state)
    shuffled = rng.permutation(uniq)
    n_eval_groups = max(1, int(round(holdout_frac * uniq.size)))
    n_eval_groups = min(n_eval_groups, uniq.size - 1)
    eval_groups = set(shuffled[:n_eval_groups].tolist())
    is_eval = np.fromiter((g in eval_groups for g in groups), dtype=bool, count=n_samples)
    return np.flatnonzero(~is_eval), np.flatnonzero(is_eval)


def _top_k_set(scores: np.ndarray, k: int) -> set:
    """Indices of the ``k`` highest scores (most anomalous)."""
    if k <= 0:
        return set()
    k = min(k, scores.size)
    return set(np.argpartition(scores, -k)[-k:].tolist())


def _rank_agreement(
    detector_kwargs: dict,
    X: ArrayLike,
    fit_idx: np.ndarray,
    ref_idx: np.ndarray,
    random_state: int,
    contamination: Union[float, Sequence[float]] = _DEFAULT_CONTAMINATION,
) -> float:
    """Label-free objective: out-of-sample stability of the anomaly ranking.

    Splits ``fit_idx`` in half, fits the *same* configuration on each half with
    a different seed, scores the common held-out ``ref_idx`` block with both,
    and returns

    ``max(spearman(scores_a, scores_b), 0) * jaccard(top-k_a, top-k_b)``

    where ``k = round(contamination * n_eval)``; ``contamination`` may be a sequence of
    fractions, in which case the Jaccard is averaged over them (one alert budget is a
    noisy summary of the head). Note it is the top **k-fraction**, not a decile.

    TEORÍA: a detector that has found real structure produces a ranking that
    does not depend on which half of the data it was fitted on; one that is
    fitting sampling noise produces a ranking that reshuffles. Spearman's rho
    measures agreement over the whole ordering, but the decision only ever uses
    the head of it, so the rho is weighted by the Jaccard overlap of the two
    top-deciles -- a configuration whose global ordering is stable but whose
    *alert set* is not gets penalised.

    This replaces :func:`_separation_margin`, which was monotone in its own
    ``contamination`` knob and therefore optimised the metric rather than the
    forest. It is also the honest version of
    ``src.evaluation.metrics._rank_stability``: that one perturbs fixed scores
    with jitter because refitting is unavailable at metric-computation time,
    whereas during tuning refitting is exactly what we can afford.

    Degeneracy guard: a constant (or near-constant) score vector has an
    undefined/perfect rank correlation, which would make a collapsed model look
    maximally stable. Such configurations return ``0.0``.
    """
    if fit_idx.size < 4 or ref_idx.size < 3:
        return 0.0
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(fit_idx)
    half = shuffled.size // 2
    halves = (np.sort(shuffled[:half]), np.sort(shuffled[half:]))

    scores = []
    for offset, part in enumerate(halves):
        detector = IsolationForestDetector(
            random_state=random_state + offset, n_jobs=-1, **detector_kwargs
        )
        detector.model_ = detector._build()
        detector.model_.fit(detector._as_model_input(X[part]))
        s = detector.score_samples(X[ref_idx])
        # A collapsed score distribution cannot be "stable" in any useful sense.
        if not np.isfinite(s).all() or float(np.std(s)) < 1e-12:
            return 0.0
        scores.append(s)

    rho = spearmanr(scores[0], scores[1]).correlation
    if not np.isfinite(rho):
        return 0.0

    fracs = [float(contamination)] if np.isscalar(contamination) else [float(c) for c in contamination]
    jaccards = []
    for frac in fracs:
        k = max(1, int(round(frac * ref_idx.size)))
        top_a, top_b = _top_k_set(scores[0], k), _top_k_set(scores[1], k)
        union = len(top_a | top_b)
        jaccards.append((len(top_a & top_b) / union) if union else 0.0)
    jaccard = float(np.mean(jaccards))

    return float(max(rho, 0.0) * jaccard)


def _study_fingerprint(
    X: ArrayLike,
    feature_names: Optional[Sequence[str]],
    mode: str,
    direction: str,
    extra: str = "",
) -> str:
    """Short hash identifying the data + objective a study's trials belong to.

    TEORÍA: Optuna's ``load_if_exists=True`` resumes a study by *name* only. A
    fixed name means trials produced on a 2,000-entity panel with one-hot
    encoding are pooled with trials from a 100,000-entity panel with frequency
    encoding, and TPE then models a response surface stitched together from
    incomparable objective values. Binding the name to the matrix shape, the
    feature list and the objective mode keeps resume working within a
    configuration while isolating different ones.
    """
    n_samples, n_features = _n_samples_features(X)
    names = "" if feature_names is None else ",".join(str(f) for f in feature_names)
    h = hashlib.sha1()
    h.update(f"{n_samples}|{n_features}|{mode}|{direction}|{names}|{extra}|".encode("utf-8"))
    if sp.issparse(X):
        matrix = X.tocsr()
        h.update(str(matrix.dtype).encode("ascii"))
        for part in (matrix.data, matrix.indices, matrix.indptr):
            arr = np.ascontiguousarray(part)
            h.update(memoryview(arr).cast("B"))
    else:
        arr = np.ascontiguousarray(np.asarray(X))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(memoryview(arr).cast("B"))
    return h.hexdigest()[:10]


def _separation_margin(scores: np.ndarray, contamination: float) -> float:
    """DEPRECATED unsupervised proxy: normalized gap between tail and bulk.

    .. deprecated::
       No longer the default unsupervised objective -- superseded by
       :func:`_rank_agreement`. Kept importable for reference and for callers
       that pass it explicitly via ``objective_metric``.

    Splits the anomaly scores at the ``contamination`` quantile (top fraction
    treated as the putative anomalies) and returns
    ``(mean(top) - mean(rest)) / std(all)``.

    Why it was retired -- TEORÍA: the scores returned by ``score_samples`` are
    invariant to ``contamination``, but this metric uses that same
    ``contamination`` as the tail fraction ``k`` at which it cuts them. Shrinking
    ``k`` selects a more extreme tail, which mechanically raises
    ``mean(top) - mean(rest)``. The objective is therefore monotonically
    increasing in a knob that does not affect the model at all, so a search over
    it is driven to the lower bound of the ``contamination`` range regardless of
    forest quality: it optimises the metric's own parameter instead of the
    forest. It also rewards *separation* rather than *correctness* -- nothing
    verifies that the isolated tail are the true anomalies.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    if n == 0:
        return 0.0
    frac = contamination if isinstance(contamination, (int, float)) else 0.05
    frac = float(min(max(frac, 1.0 / n), 0.5))
    k = max(1, int(round(frac * n)))
    order = np.argsort(scores)  # ascending; most anomalous at the tail
    top = scores[order[-k:]]
    rest = scores[order[:-k]]
    if rest.size == 0:
        return 0.0
    std = float(scores.std()) + 1e-12
    return float((top.mean() - rest.mean()) / std)


def _tail_separation(scores: np.ndarray, hi: float = 95.0, lo: float = 50.0) -> float:
    """Label-free proxy: how far the score tail sits above the normal bulk.

    ``percentile(scores, hi) - percentile(scores, lo)``, normalised by the
    interquartile range so configurations with different score scales stay
    comparable.

    TEORÍA: a useful detector produces a score distribution with a *detached*
    upper tail -- a small set of easily-isolated points well above the mass of
    normal ones. Both cut points are **fixed constants**, which is what makes
    this safe to optimise: the retired ``_separation_margin`` cut the tail at the
    trial's own ``contamination`` while the scores were invariant to it, so
    shrinking that knob mechanically inflated the metric and the search
    optimised the metric's parameter instead of the forest. With p95 and p50
    fixed, the only way to raise this number is to actually push the tail away
    from the bulk.

    Honest caveat: it still rewards *separation*, not *correctness*. Nothing
    label-free can verify that the separated tail holds the true anomalies.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 4:
        return 0.0
    p_hi, p_lo = np.percentile(s, [float(hi), float(lo)])
    q75, q25 = np.percentile(s, [75.0, 25.0])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        # A collapsed score distribution has no meaningful separation.
        return 0.0
    return float((p_hi - p_lo) / scale)


def _supervised_score(
    y: np.ndarray, scores: np.ndarray, objective_metric: Optional[str]
) -> float:
    """PR-AUC (default) or ROC-AUC of anomaly scores against binary labels."""
    metric = (objective_metric or "average_precision").lower()
    if metric in ("average_precision", "pr_auc", "prauc", "ap"):
        return float(average_precision_score(y, scores))
    if metric in ("roc_auc", "rocauc", "auc", "roc"):
        return float(roc_auc_score(y, scores))
    raise ValueError(
        f"Unknown supervised objective_metric {objective_metric!r}; "
        "use 'average_precision' or 'roc_auc'."
    )


def _json_safe(value):
    """Best-effort conversion of numpy scalars/arrays for YAML/JSON payloads."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _select_trial(
    completed: list,
    reference: dict,
    *,
    sign: float,
    one_se_rule: bool,
    margin_sd: float,
    cost_of: Callable[[dict], float],
    evaluations: Optional[dict[int, dict]] = None,
) -> dict:
    """Pick the trial to deploy using replicated means and a real 1-SE band.

    ``evaluations`` maps each trial number to replicated ``values``, ``mean`` and
    ``se`` from the same seeds used for the paper default. The standard-error band
    therefore describes uncertainty around the best configuration itself; it is
    not the old shortcut that reused the default's standard deviation.

    * **1-SE rule** -- among replicated trial means within one SE of the best, take
      the *cheapest* (smallest ``max_samples * max_features``): configurations
      that cannot be told apart from the best should not be paid for.
    * **Margin rule** -- the pick is kept only if it beats the reference by more
      than ``margin_sd`` noise ``sd``; otherwise the reference (default) is kept.
      A tuned config that cannot be told apart from the default is not a finding.
    """
    evaluations = evaluations or {
        t.number: {"mean": float(t.value), "se": float(reference.get("se", reference["sd"])),
                   "values": [float(t.value)]}
        for t in completed
    }
    best = max(completed, key=lambda t: sign * evaluations[t.number]["mean"])
    best_mean = float(evaluations[best.number]["mean"])
    best_se = float(evaluations[best.number]["se"])
    picked = best
    if one_se_rule:
        band = [t for t in completed
                if sign * evaluations[t.number]["mean"] >= sign * best_mean - best_se]
        picked = min(
            band,
            key=lambda t: (cost_of(t.params), -sign * evaluations[t.number]["mean"]),
        )
    picked_eval = evaluations[picked.number]
    picked_values = np.asarray(picked_eval.get("values", [picked_eval["mean"]]), dtype=float)
    reference_values = np.asarray(reference.get("values", [reference["mean"]]), dtype=float)
    if len(picked_values) == len(reference_values) and len(picked_values) > 1:
        differences = sign * (picked_values - reference_values)
        improvement = float(np.mean(differences))
        improvement_se = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    else:
        improvement = sign * (float(picked_eval["mean"]) - float(reference["mean"]))
        improvement_se = float(np.hypot(picked_eval.get("se", 0.0), reference.get("se", 0.0)))
    beats_reference = improvement > margin_sd * improvement_se
    return {
        "best_trial": best.number, "best_value": best_mean, "best_se": best_se,
        "picked_trial": picked.number, "picked_value": float(picked_eval["mean"]),
        "noise_sd": float(reference["sd"]), "reference_mean": float(reference["mean"]),
        "improvement_se": improvement_se,
        "margin_sd": float(margin_sd), "one_se_rule": bool(one_se_rule),
        "beats_reference": bool(beats_reference),
        "trials_within_noise_of_best": int(sum(
            sign * evaluations[t.number]["mean"] >= sign * best_mean - best_se for t in completed)),
        "params": dict(picked.params),
    }


def tune_iforest(
    X: ArrayLike,
    n_trials: int = 50,
    y: Optional[np.ndarray] = None,
    storage: Optional[str] = None,
    study_name: str = "iforest",
    direction: Optional[str] = None,
    objective_metric: Optional[Union[str, Callable[["IsolationForestDetector", ArrayLike], float]]] = None,
    best_params_path: str = _DEFAULT_BEST_PARAMS,
    model_out: str = _DEFAULT_MODEL_OUT,
    random_state: int = 42,
    timeout: Optional[float] = None,
    groups: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    contamination: float = _DEFAULT_CONTAMINATION,
    holdout_frac: float = _DEFAULT_HOLDOUT_FRAC,
    feature_names: Optional[Sequence[str]] = None,
    study_tag: Optional[str] = None,
    early_stopping_patience: Optional[int] = 10,
    early_stopping_min_delta: float = 0.005,
    early_stopping_min_trials: int = 10,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
    max_samples_range: tuple[int, int] = _DEFAULT_PSI_RANGE,
    max_features_range: tuple[float, float] = _DEFAULT_MF_RANGE,
    bootstrap: bool = False,
    tail_fracs: Sequence[float] = _DEFAULT_TAIL_FRACS,
    noise_seeds: int = _DEFAULT_NOISE_SEEDS,
    margin_sd: float = 1.0,
    one_se_rule: bool = True,
    min_eval_positives: int = _MIN_EVAL_POSITIVES,
    label_source: str = "labels",
):
    """Tune :class:`IsolationForestDetector` with Optuna and crash recovery.

    What is searched, and why so little
    -----------------------------------
    Two dimensions only:

    * ``max_samples`` -- an **absolute integer** (rows per tree), log-uniform in
      ``max_samples_range`` (default 1 024-32 768), capped at the number of rows
      each trial is fitted on. It is *never* a fraction of the data: a fraction
      is a different absolute size in the trial (half of the fit block), in the
      final refit (all in-time rows) and in the stacking forest (train rows), so
      what was validated would not be what is deployed.
    * ``max_features`` -- float in ``max_features_range`` (default 0.5-1.0).

    ``n_estimators`` is a **fixed** argument (default 300): the previous
    label-free objective (``rank_agreement``) rose mechanically with the tree
    count (Spearman 0.87), so a searched ``n_estimators`` just drove the search
    to its upper bound at ~5x the cost with no gain in detection quality.
    ``bootstrap`` is fixed as well. ``contamination`` is **not** a model
    hyper-parameter: it only shifts ``offset_`` and cannot change any score or
    ranking; it is the deployed detector's ``predict()`` operating point and has
    no influence on which configuration wins.

    Search design
    -------------
    The budget is spent on a low-discrepancy (Sobol/QMC) sweep of the 2-D space,
    preceded by *anchor* trials at ``max_samples`` in {1 024, 4 096, 16 384,
    32 768} (all features), so even a 5-trial ``--quick`` run covers the axis
    that matters. A model-based sampler (TPE) has nothing to model with 15 noisy
    trials on an almost one-dimensional response.

    Selection rule (not just "argmax")
    ----------------------------------
    The objective is noisy, and the argmax of noisy values is biased upward. After
    the trials, the **paper default** (``max_samples=256``, all features) is
    re-evaluated with ``noise_seeds`` different seeds. Every trial is re-evaluated
    on those same seeds; the deployed configuration is the *cheapest* replicated
    mean within one standard error of the best replicated mean. It is kept only
    when its paired improvement over the reference exceeds ``margin_sd`` standard
    errors; otherwise the reference default is kept. The decision is written to YAML.

    Objective (held-out, always out-of-sample)
    ------------------------------------------
    Every trial fits on ``fit_idx`` and is scored on the disjoint ``eval_idx``
    (the chronological validation months when ``valid_mask`` is given).

    * **Labelled** (``y`` given, aligned to ``X``; ``NaN`` = unknown row, ignored):
      average precision (or ROC-AUC via ``objective_metric='roc_auc'``) of the
      held-out scores against the labels. With reviewed labels this is the only
      real signal; it is used only when ``eval_idx`` holds at least
      ``min_eval_positives`` positives and both classes, otherwise the study falls
      back to the label-free objective and says so.
    * **Label-free** (default): ``tail_separation`` -- how far the score tail sits
      above the bulk, with fixed cut points (p95, p50) so nothing is gameable.
      It rewards separation, not correctness (nothing label-free can verify
      that); it tracked the true quality across a 156-configuration grid at
      Spearman 0.64 while being unrelated to the tree count.
    * ``objective_metric='rank_agreement'`` keeps the legacy stability objective
      (averaged over ``tail_fracs`` top-k sizes instead of a single one).
    * ``objective_metric`` may also be a callable ``(detector, X) -> float``.

    Crash recovery and resume
    -------------------------
    The study lives in a persistent SQLite RDBStorage with ``load_if_exists``; the
    best-so-far configuration is checkpointed to YAML after every trial.
    ``n_trials`` is the *total* budget: a resumed study only runs the missing
    trials (``n_trials - completed``), so repeating a run does not silently grow
    it, and trials left ``RUNNING`` by a crash are closed as failed. The study
    name carries a fingerprint of the data shape, features, objective, search
    space and validation split, so a changed setup starts a fresh study instead
    of mixing non-comparable objective values.

    Args:
        X: Preprocessed feature matrix (dense ndarray or scipy sparse CSR).
        n_trials: Total trial budget of the study (resumed trials count).
        y: Optional 0/1 labels aligned to ``X`` rows, ``NaN`` for unknown rows.
        storage: Optuna storage URI; defaults to the SQLite DB above.
        study_name: Study name *prefix* (suffixed with a fingerprint).
        direction: 'maximize' / 'minimize'; ``None`` = 'maximize' (every built-in
            objective is higher-is-better).
        objective_metric: Metric name or custom callable (see above).
        best_params_path: YAML path for the incremental best-params checkpoint.
        model_out: joblib path for the final refitted detector.
        random_state: Seed for the sampler, the split and every fitted forest.
        timeout: Optional wall-clock budget (seconds) for ``study.optimize``.
        groups: Per-row groups used only when no ``valid_mask`` is given.
        valid_mask: Chronological validation rows (priority over ``groups``).
        contamination: Operating point stored on the deployed detector (does not
            affect selection).
        holdout_frac: Held-out fraction when neither mask nor groups define it.
        feature_names: Feature list folded into the study fingerprint.
        study_tag: Explicit study-name suffix, overriding the fingerprint.
        early_stopping_patience, early_stopping_min_delta, early_stopping_min_trials:
            Trial-level early stopping, see
            :class:`src.models._tuning_stop.TrialPatienceStopper`.
        n_estimators: Fixed number of trees.
        max_samples_range: Inclusive absolute range searched for ``max_samples``.
        max_features_range: Range searched for ``max_features``.
        bootstrap: Fixed bootstrap flag.
        tail_fracs: Top-k fractions averaged by the legacy ``rank_agreement``.
        noise_seeds: Seeds used to measure the reference's noise floor (>= 2).
        margin_sd: Noise sds by which the pick must beat the reference.
        one_se_rule: Apply the cheapest-within-noise rule.
        min_eval_positives: Minimum held-out positives for the labelled objective.
        label_source: Provenance tag of ``y`` (goes into the study fingerprint and
            the YAML), e.g. ``'reviewed_labels'``.

    Returns:
        The Optuna :class:`~optuna.study.Study`; the decision is in
        ``study.user_attrs['selection']`` and in the YAML.
    """
    import warnings

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    warnings.filterwarnings("ignore", message=".*balance properties of Sobol.*")

    log = setup_logging()
    contamination = _DEFAULT_CONTAMINATION
    n_estimators = _DEFAULT_N_ESTIMATORS
    if storage is None:
        storage = _default_storage_uri()
    _ensure_parent_dir(best_params_path)
    noise_seeds = max(2, int(noise_seeds))

    # Normalise once so the row-subsetting below (`X[fit_idx]`) is valid for
    # every accepted input type; CSR is the only sparse layout that supports it.
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)

    y_arr = None if y is None else np.asarray(y, dtype=float).ravel()
    custom_objective = objective_metric if callable(objective_metric) else None
    metric_name = objective_metric if isinstance(objective_metric, str) else None
    forced_unsupervised = metric_name in _UNSUPERVISED_METRICS

    if direction is None:
        direction = "maximize"
        if custom_objective is not None:
            log.warning(
                "tune_iforest: a callable objective_metric was supplied without an "
                "explicit direction; assuming 'maximize'. Pass direction='minimize' "
                "for a loss-style objective."
            )
    sign = 1.0 if direction == "maximize" else -1.0

    # One split for every objective. `valid_mask` (the chronological validation
    # months) has priority: the model is applied to *future* periods, so selecting
    # hyper-parameters on future rows is the only honest measurement. Entity
    # blocking defends a different leak and remains the fallback without a time split.
    n_samples, _ = _n_samples_features(X)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool).ravel()
        if vm.shape[0] != n_samples:
            raise ValueError(f"valid_mask has {vm.shape[0]} entries but X has {n_samples} rows")
        if not vm.any() or vm.all():
            raise ValueError("valid_mask must select some -- but not all -- rows")
        eval_idx = np.flatnonzero(vm)
        fit_idx = np.flatnonzero(~vm)
        split_kind = "temporal (valid_mask)"
    else:
        fit_idx, eval_idx = _blocked_split(n_samples, groups, holdout_frac, random_state)
        split_kind = "entity-blocked" if groups is not None else "row-wise (no groups given)"
    log.info("Objective split: %d fit rows / %d held-out rows (%s)",
             fit_idx.size, eval_idx.size, split_kind)

    # -- which objective actually runs ---------------------------------------- #
    use_supervised = False
    y_eval = keep_eval = None
    if y_arr is not None and custom_objective is None and not forced_unsupervised:
        y_eval_all = y_arr[eval_idx]
        keep_eval = ~np.isnan(y_eval_all)
        y_eval = y_eval_all[keep_eval]
        n_pos, n_neg = int((y_eval == 1).sum()), int((y_eval == 0).sum())
        use_supervised = n_pos >= int(min_eval_positives) and n_neg >= 1
        if not use_supervised:
            log.warning(
                "Held-out block has %d labelled positive(s) and %d negative(s) (need >= %d "
                "positives and >= 1 negative); falling back to the label-free objective.",
                n_pos, n_neg, min_eval_positives,
            )
    elif y_arr is not None and forced_unsupervised:
        log.info("objective_metric=%r is label-free; the supplied labels are ignored for "
                 "model selection.", metric_name)
    unsupervised_kind = metric_name if forced_unsupervised else "tail_separation"
    if custom_objective is not None:
        objective_name = "custom"
    elif use_supervised:
        objective_name = f"supervised({label_source})"
    else:
        objective_name = f"unsupervised({unsupervised_kind})"
    mode = objective_name

    # -- absolute max_samples range: capped at what each trial really fits on -------- #
    halves = (
        custom_objective is None and not use_supervised and unsupervised_kind == "rank_agreement"
    )
    fit_cap = fit_idx.size // 2 if halves else fit_idx.size
    psi_hi = max(2, min(int(max_samples_range[1]), fit_cap))
    psi_lo = max(2, min(int(max_samples_range[0]), psi_hi))
    mf_lo, mf_hi = float(max_features_range[0]), float(max_features_range[1])
    reference_params = {"max_samples": min(_REFERENCE_MAX_SAMPLES, psi_hi), "max_features": mf_hi}

    def kwargs_of(params: dict) -> dict:
        return {
            "n_estimators": int(n_estimators),
            "max_samples": int(params["max_samples"]),
            "max_features": float(params["max_features"]),
            "contamination": float(contamination),
            "bootstrap": bool(bootstrap),
        }

    def fit_on(rows: np.ndarray, params: dict, seed: int) -> "IsolationForestDetector":
        detector = IsolationForestDetector(random_state=seed, n_jobs=-1, **kwargs_of(params))
        detector.model_ = detector._build()
        detector.model_.fit(detector._as_model_input(X[rows]))
        return detector

    def evaluate(params: dict, seed: int) -> float:
        """The objective of one configuration at one seed (out-of-sample)."""
        if unsupervised_kind == "rank_agreement" and not use_supervised and custom_objective is None:
            return _rank_agreement(kwargs_of(params), X, fit_idx, eval_idx, seed, tuple(tail_fracs))
        detector = fit_on(fit_idx, params, seed)
        if custom_objective is not None:
            return float(custom_objective(detector, X))
        scores = detector.score_samples(X[eval_idx])
        if use_supervised:
            return _supervised_score(y_eval, scores[keep_eval], metric_name)
        return _tail_separation(scores)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "max_samples": trial.suggest_int("max_samples", psi_lo, psi_hi, log=True),
            "max_features": trial.suggest_float("max_features", mf_lo, mf_hi),
        }
        value = evaluate(params, random_state)
        log.debug("Trial %d: value=%.6f params=%s", trial.number, value, trial.params)
        return value

    # -- study ----------------------------------------------------------------------- #
    split_hash = hashlib.sha1(np.ascontiguousarray(eval_idx).tobytes()).hexdigest()[:8]
    if use_supervised:
        label_identity = hashlib.sha1()
        label_identity.update(np.ascontiguousarray(keep_eval, dtype=bool).tobytes())
        label_identity.update(np.ascontiguousarray(y_eval, dtype=float).tobytes())
        label_hash = label_identity.hexdigest()[:8]
    else:
        label_hash = "-"
    fingerprint_extra = (
        f"ne={n_estimators}|psi={psi_lo}-{psi_hi}|mf={mf_lo}-{mf_hi}|boot={bootstrap}|"
        f"split={split_hash}|labels={label_hash}|obj={objective_name}|tails={tuple(tail_fracs)}"
    )
    suffix = study_tag or _study_fingerprint(X, feature_names, mode, direction, fingerprint_extra)
    study_name = f"{study_name}_{suffix}"

    from src.models._optuna_storage import resolve_storage

    study = optuna.create_study(
        study_name=study_name,
        storage=resolve_storage(storage),
        direction=direction,
        sampler=optuna.samplers.QMCSampler(qmc_type="sobol", scramble=True, seed=random_state),
        load_if_exists=True,  # <-- crash-recovery / resume switch
    )
    # Trials a crash left RUNNING would otherwise stay unresolved forever.
    for stale in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        try:
            study.tell(stale.number, state=optuna.trial.TrialState.FAIL)
        except Exception:  # noqa: BLE001 - best effort
            pass
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    to_run = max(0, int(n_trials) - n_done)
    if not study.trials:
        anchors = sorted({int(min(max(a, psi_lo), psi_hi)) for a in _PSI_ANCHORS})
        for anchor in anchors[: max(0, to_run)]:
            study.enqueue_trial({"max_samples": anchor, "max_features": mf_hi})

    log.info(
        "Optuna tuning: study=%r storage=%r mode=%s direction=%s n_estimators=%d (fixed) "
        "max_samples=[%d, %d] (absolute) max_features=[%.2f, %.2f] contamination=%.4g "
        "(operating point, not searched) budget=%d completed=%d -> running %d",
        study_name, storage, mode, direction, n_estimators, psi_lo, psi_hi, mf_lo, mf_hi,
        contamination, n_trials, n_done, to_run,
    )

    def payload_for(trial_params: dict, value: float, number: int, extra: Optional[dict] = None) -> dict:
        kwargs = kwargs_of(trial_params)
        payload = {
            "study_name": study_name,
            "direction": direction,
            "best_value": float(value),
            "best_trial_number": number,
            "n_trials_completed": sum(
                t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
            "objective_mode": mode,
            "label_source": label_source if use_supervised else None,
            "random_state": random_state,
            # `contamination` is the deployed detector's operating point. It is not
            # tuned and cannot change any score or ranking.
            "contamination": float(contamination),
            "contamination_tuned": False,
            "holdout_frac": float(holdout_frac),
            "n_estimators_fixed": int(n_estimators),
            "max_samples_absolute": kwargs["max_samples"],
            "best_params": kwargs,
            "raw_optuna_params": dict(trial_params),
        }
        if extra:
            payload.update(extra)
        return _json_safe(payload)

    def write_payload(payload: dict) -> None:
        tmp_path = best_params_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        atomic_replace(tmp_path, best_params_path)  # atomic swap, Windows-lock-safe

    progress = Bar(desc=f"optuna[{study_name}]", total=to_run, unit="trial")

    def _progress_callback(study_: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:
        progress.update(1)
        _show_best_trial(progress, study_)

    def _persist_best_callback(study_: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:
        """Checkpoint the best-so-far configuration to YAML after each trial."""
        try:
            best = study_.best_trial
        except (ValueError, RuntimeError):
            return  # no completed trial yet
        write_payload(payload_for(best.params, best.value, best.number,
                                  {"status": "in_progress_best_so_far"}))
        log.info("Checkpointed best params (trial %d, value=%.6f) -> %s",
                 best.number, best.value, best_params_path)

    callbacks = [_progress_callback, _persist_best_callback]
    stopper = None
    if early_stopping_patience is not None and to_run > 0:
        from src.models._tuning_stop import TrialPatienceStopper

        stopper = TrialPatienceStopper(
            direction=direction, model_name="iforest",
            n_trials_requested=to_run + len(study.trials),
            patience=early_stopping_patience, min_delta=early_stopping_min_delta,
            min_trials=early_stopping_min_trials,
        )
        callbacks.append(stopper)

    with log_phase("iforest.tune (optuna)", log):
        try:
            if to_run > 0:
                study.optimize(objective, n_trials=to_run, timeout=timeout,
                               callbacks=callbacks, gc_after_trial=True)
        finally:
            progress.close()
        if stopper is not None and stopper.stopped:
            log.info("iForest tuning stopped early: %s (%d trial(s) skipped).",
                     stopper.stop_reason, stopper.trials_skipped)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"Isolation Forest tuning study {study_name!r} has no completed trials; "
            "no model or final parameter file was produced."
        )

    # -- selection: replicated 1-SE rule + paired margin over paper default ---------- #
    with log_phase("iforest.select (reference + noise)", log):
        selection_seeds = [random_state + 10_000 + i for i in range(noise_seeds)]
        ref_values = [
            evaluate(reference_params, seed) for seed in selection_seeds
        ]
        trial_evaluations = {}
        for trial in completed:
            values = [evaluate(trial.params, seed) for seed in selection_seeds]
            trial_evaluations[trial.number] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                "values": [float(v) for v in values],
            }
    reference = {"mean": float(np.mean(ref_values)), "sd": float(np.std(ref_values, ddof=1)),
                  "se": float(np.std(ref_values, ddof=1) / np.sqrt(len(ref_values))),
                  "values": [float(v) for v in ref_values], "params": dict(reference_params)}
    selection = _select_trial(
        completed, reference, sign=sign, one_se_rule=one_se_rule, margin_sd=margin_sd,
        cost_of=lambda p: float(p["max_samples"]) * float(p["max_features"]),
        evaluations=trial_evaluations,
    )
    selection["trial_evaluations"] = trial_evaluations
    selection["reference"] = reference
    if selection["beats_reference"]:
        final_params, final_value, final_trial = selection["params"], selection["picked_value"], selection["picked_trial"]
        selection["deployed"] = "tuned"
    else:
        final_params, final_value, final_trial = reference_params, reference["mean"], -1
        selection["deployed"] = "reference_default"
        log.warning(
            "The selected trial (replicated mean %.4f) does not beat the paper default "
            "(%.4f) by more than %.1f paired standard error(s); keeping %s.",
            selection["picked_value"], reference["mean"], margin_sd, reference_params,
        )
    try:
        study.set_user_attr("selection", _json_safe(selection))
    except Exception:  # noqa: BLE001 - metadata only
        pass

    best_kwargs = kwargs_of(final_params)
    log.info("Deployed configuration (%s): value=%.6f params=%s",
             selection["deployed"], final_value, best_kwargs)
    write_payload(payload_for(final_params, final_value, final_trial,
                              {"status": "final", "selection": selection}))

    # The final model is refit on ALL of X (fit + held-out): the split exists to
    # make model *selection* honest, not to throw away data once selected. The
    # absolute `max_samples` is the same integer the trials were scored with.
    with log_phase("iforest.refit_best", log):
        best_detector = IsolationForestDetector(random_state=random_state, n_jobs=-1, **best_kwargs)
        best_detector.fit(X)
        best_detector.save(model_out)

    return study


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #
def plot_score_distribution(
    scores: np.ndarray,
    out_dir: str = _DEFAULT_FIG_DIR,
    filename: str = "iforest_score_distribution.png",
    y: Optional[np.ndarray] = None,
    max_points: int = 200_000,
    random_state: int = 42,
) -> str:
    """Plot a histogram of anomaly scores and save it under ``reports/figures``.

    Uses the non-interactive ``Agg`` backend. When ``y`` (0/1 labels) is given,
    the normal and anomaly score distributions are overlaid. Very large score
    vectors are randomly subsampled to ``max_points`` for a legible, cheap plot.
    Figures always land under ``reports/figures/`` per the project rule.

    Returns:
        The absolute path of the written PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=float).ravel()
    y_arr = None if y is None else np.asarray(y).ravel()

    if scores.size > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(scores.size, size=max_points, replace=False)
        scores = scores[idx]
        if y_arr is not None:
            y_arr = y_arr[idx]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    fig, ax = plt.subplots(figsize=(8, 5))
    if y_arr is not None and np.unique(y_arr).size > 1:
        normal = scores[y_arr == 0]
        anomaly = scores[y_arr == 1]
        bins = np.histogram_bin_edges(scores, bins=60)
        ax.hist(normal, bins=bins, alpha=0.6, density=True, label="normal", color="#4c72b0")
        ax.hist(anomaly, bins=bins, alpha=0.6, density=True, label="anomaly", color="#c44e52")
        ax.legend()
    else:
        ax.hist(scores, bins=60, alpha=0.8, color="#4c72b0")

    ax.set_title("Isolation Forest anomaly-score distribution (higher = more anomalous)")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("density" if y_arr is not None else "count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    setup_logging().info("Saved score distribution figure to %s", out_path)
    return os.path.abspath(out_path)
