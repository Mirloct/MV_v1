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

__all__ = [
    "IsolationForestDetector",
    "tune_iforest",
    "plot_score_distribution",
]

# Fraction of entities (or rows) held out from every trial's fit so the
# objective is never scored in-sample. See `_blocked_split`.
_DEFAULT_HOLDOUT_FRAC = 0.3

# Operating point used for the final refit and for the top-k overlap term of
# the unsupervised objective: the report's headline deliverable is the OOT top
# decile, so 0.10 is the alert budget the model is actually judged on.
_DEFAULT_CONTAMINATION = 0.10

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
        n_estimators: int = 200,
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
    ms_mode = params.get("max_samples_mode", "float")
    if ms_mode == "auto":
        max_samples: Union[str, int, float] = "auto"
    elif ms_mode == "int":
        max_samples = int(params.get("max_samples_int", 256))
    else:
        max_samples = float(params.get("max_samples", 1.0))
    return {
        "n_estimators": int(params["n_estimators"]),
        "max_samples": max_samples,
        "max_features": float(params["max_features"]),
        "contamination": float(params.get("contamination", contamination)),
        "bootstrap": bool(params["bootstrap"]),
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
    contamination: float = _DEFAULT_CONTAMINATION,
) -> float:
    """Label-free objective: out-of-sample stability of the anomaly ranking.

    Splits ``fit_idx`` in half, fits the *same* configuration on each half with
    a different seed, scores the common held-out ``ref_idx`` block with both,
    and returns

    ``max(spearman(scores_a, scores_b), 0) * jaccard(top-decile_a, top-decile_b)``

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

    k = max(1, int(round(float(contamination) * ref_idx.size)))
    top_a, top_b = _top_k_set(scores[0], k), _top_k_set(scores[1], k)
    union = len(top_a | top_b)
    jaccard = (len(top_a & top_b) / union) if union else 0.0

    return float(max(rho, 0.0) * jaccard)


def _study_fingerprint(
    X: ArrayLike,
    feature_names: Optional[Sequence[str]],
    mode: str,
    direction: str,
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
    payload = f"{n_samples}|{n_features}|{mode}|{direction}|{names}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


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
):
    """Tune :class:`IsolationForestDetector` with Optuna and crash recovery.

    Crash recovery
    --------------
    The study is created against a **persistent SQLite RDBStorage** (default
    ``sqlite:///configs/optuna_iforest.db``) with
    ``optuna.create_study(..., load_if_exists=True)``. This is the recovery
    mechanism: if the process dies mid-search and ``tune_iforest`` is called
    again with the same ``study_name`` + ``storage``, Optuna *reopens* the
    existing study and continues from the already-completed trials rather than
    restarting from scratch. On top of that, the current best hyperparameters
    are written to ``best_params_path`` (YAML) after **every** completed trial,
    so the best-so-far configuration is always durable on disk even if the run
    is interrupted before it finishes.

    Search space
    ------------
    * ``n_estimators`` -- int in [100, 600] (step 50)
    * ``max_samples`` -- categorical mode {'auto', 'float'}; when 'float', a
      fraction in [0.3, 1.0]
    * ``max_features`` -- float in [0.3, 1.0]
    * ``bootstrap`` -- {True, False}

    ``contamination`` is deliberately **not** searched -- it cannot move any
    rank-based objective (see :func:`_detector_kwargs_from_params`) -- and is
    instead fixed by the ``contamination`` argument at the operating point the
    model is judged on.

    Held-out objective
    ------------------
    Every trial fits on ``fit_idx`` and is scored on the disjoint ``eval_idx``
    produced by :func:`_blocked_split`; when ``groups`` is supplied the split
    keeps whole entities on one side. Scoring a trial on the rows it was fitted
    on measures how well the forest memorised the sample, not how well it
    generalises.

    Objective modes
    ----------------
    * **Supervised** (``y`` given, aligned row-for-row to ``X``): scores the
      study on the held-out anomaly scores vs. the 0/1 labels, defaulting to
      **average_precision_score (PR-AUC)** -- the informative summary for
      heavily imbalanced anomaly detection -- switchable to ROC-AUC via
      ``objective_metric='roc_auc'``. Falls back to the unsupervised objective
      for a trial whose held-out block contains a single class.
    * **Unsupervised** (``y is None``): :func:`_rank_agreement` -- the
      out-of-sample stability of the anomaly ranking under refitting.

    ``objective_metric`` may also be a callable ``(detector, X) -> float`` to
    plug in a fully custom objective (evaluated in either mode).

    Args:
        X: Preprocessed feature matrix (dense ndarray or scipy sparse CSR).
        n_trials: Number of *new* trials to run in this call.
        y: Optional 0/1 anomaly labels aligned to ``X`` rows (supervised mode).
        storage: Optuna storage URI; defaults to the SQLite DB above.
        study_name: Study name *prefix*; the effective name is suffixed with
            ``study_tag`` or a data/objective fingerprint (see
            :func:`_study_fingerprint`) so resume stays scoped to a
            configuration.
        direction: 'maximize' / 'minimize'. ``None`` (default) auto-resolves,
            mirroring :func:`src.models.vae.tune_vae`.
        objective_metric: Metric name or custom callable (see above).
        best_params_path: YAML path for the incremental best-params checkpoint.
        model_out: joblib path for the final refitted detector.
        random_state: Seed for the sampler, the split and every fitted forest.
        timeout: Optional wall-clock budget (seconds) for ``study.optimize``.
        groups: Optional per-row group labels (typically ``entity_id``) used to
            block the fit/eval split and the rank-agreement refits.
        contamination: Fixed operating point for the final detector and for the
            top-k overlap term of the unsupervised objective.
        holdout_frac: Fraction of entities (or rows) held out from each trial's
            fit.
        feature_names: Optional feature list folded into the study fingerprint,
            so a change of encoding starts a fresh study.
        study_tag: Explicit study-name suffix, overriding the fingerprint.
        early_stopping_patience: Stop the study (skip remaining trials) after
            this many consecutive trials with no >= ``early_stopping_min_delta``
            relative improvement in ``study.best_value``. ``None`` disables
            trial-level early stopping (all ``n_trials`` always run). This is
            independent of, and not a substitute for, the VAE's *per-epoch*
            early stopping inside a single fit -- an Optuna trial is an
            independent draw from the search space, not one more step of the
            same optimization, so there is no epoch-style convergence to test
            for; see :class:`src.models._tuning_stop.TrialPatienceStopper`.
        early_stopping_min_delta: Relative improvement threshold (e.g. ``0.005``
            = 0.5%) below which a trial does not reset the patience counter.
        early_stopping_min_trials: Never stop before this many trials have
            completed, resumed trials included.

    Returns:
        The Optuna :class:`~optuna.study.Study` (completed + resumed trials).
    """
    import optuna
    from tqdm.auto import tqdm

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log = setup_logging()
    if storage is None:
        storage = _default_storage_uri()
    _ensure_parent_dir(best_params_path)

    # Normalise once so the row-subsetting below (`X[fit_idx]`) is valid for
    # every accepted input type; CSR is the only sparse layout that supports it.
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)

    y_arr = None if y is None else np.asarray(y).ravel()
    supervised = y_arr is not None
    custom_objective = objective_metric if callable(objective_metric) else None

    _metric_tag = objective_metric if isinstance(objective_metric, str) else None
    _label_free = _metric_tag in _UNSUPERVISED_METRICS
    mode = (
        "supervised(custom)" if (supervised and custom_objective is not None)
        else "custom" if custom_objective is not None
        # A label-free metric is its own mode even when labels exist: the study
        # must not pool trials scored on PR-AUC with trials scored on a
        # separation proxy, so the objective name goes into the fingerprint.
        else f"unsupervised({_metric_tag})" if _label_free
        else "supervised" if supervised
        else "unsupervised"
    )

    # Auto-select the optimization direction from the objective mode (mirrors
    # `tune_vae`). Every built-in iForest objective is higher-is-better:
    # PR-AUC / ROC-AUC and rank agreement all improve upward.
    if direction is None:
        direction = "maximize"
        if custom_objective is not None:
            log.warning(
                "tune_iforest: a callable objective_metric was supplied without an "
                "explicit direction; assuming 'maximize'. Pass direction='minimize' "
                "for a loss-style objective."
            )

    # One split, shared by every objective mode: the supervised objective scores
    # `eval_idx`, and `_rank_agreement` refits on the two halves of `fit_idx` and
    # compares them on `eval_idx`.
    #
    # TEORÍA: `valid_mask` (the chronological validation months) takes priority
    # over any random/entity split, because it is the one that matches
    # deployment -- the model will be applied to *future* periods, so selecting
    # hyperparameters on future rows is the only honest measurement. Entity
    # blocking defends a different leak (rows of one customer share a latent
    # level) and remains the fallback when no time split is supplied.
    n_samples, _ = _n_samples_features(X)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool).ravel()
        if vm.shape[0] != n_samples:
            raise ValueError(
                f"valid_mask has {vm.shape[0]} entries but X has {n_samples} rows"
            )
        if not vm.any() or vm.all():
            raise ValueError("valid_mask must select some -- but not all -- rows")
        eval_idx = np.flatnonzero(vm)
        fit_idx = np.flatnonzero(~vm)
        split_kind = "temporal (valid_mask)"
    else:
        fit_idx, eval_idx = _blocked_split(n_samples, groups, holdout_frac, random_state)
        split_kind = "entity-blocked" if groups is not None else "row-wise (no groups given)"
    log.info(
        "Objective split: %d fit rows / %d held-out rows (%s)",
        fit_idx.size, eval_idx.size, split_kind,
    )

    y_eval = None if y_arr is None else y_arr[eval_idx]
    eval_single_class = supervised and np.unique(y_eval).size < 2
    if eval_single_class:
        log.warning(
            "Held-out block contains a single class; the supervised objective is "
            "undefined there, falling back to the unsupervised objective."
        )

    metric_name = objective_metric if isinstance(objective_metric, str) else None
    forced_unsupervised = metric_name in _UNSUPERVISED_METRICS
    use_supervised = supervised and not eval_single_class and not forced_unsupervised
    if forced_unsupervised and supervised:
        log.info(
            "objective_metric=%r is label-free; ignoring the supplied labels for "
            "model selection (they remain available for reporting).", metric_name,
        )
    unsupervised_kind = metric_name if forced_unsupervised else "rank_agreement"

    def objective(trial: "optuna.trial.Trial") -> float:
        n_estimators = trial.suggest_int("n_estimators", 100, 600, step=50)
        # TEORÍA: `max_samples` is the anti-swamping knob. Isolation Forest was
        # designed around *small* sub-samples: with too many points the normal
        # mass crowds the anomalies ("swamping") and every path gets long, which
        # is exactly what happens on an autocorrelated panel where each entity
        # contributes many near-duplicate rows. Fixed absolute sizes (64/128/256,
        # the values from the original paper's regime) are therefore offered
        # alongside the fraction-of-N options.
        ms_mode = trial.suggest_categorical("max_samples_mode", ["auto", "int", "float"])
        if ms_mode == "float":
            max_samples: Union[str, int, float] = trial.suggest_float("max_samples", 0.3, 1.0)
        elif ms_mode == "int":
            max_samples = trial.suggest_categorical("max_samples_int", [64, 128, 256])
        else:
            max_samples = "auto"
        max_features = trial.suggest_float("max_features", 0.3, 1.0)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])

        detector_kwargs = {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "contamination": contamination,
            "bootstrap": bootstrap,
        }

        needs_detector = (
            custom_objective is not None
            or use_supervised
            or unsupervised_kind == "tail_separation"
        )
        if needs_detector:
            detector = IsolationForestDetector(
                random_state=random_state, n_jobs=-1, **detector_kwargs
            )
            detector.model_ = detector._build()
            # TEORÍA: fit on `fit_idx` only. Fitting and scoring the same rows
            # makes the objective an in-sample statistic, which a larger forest
            # can always improve without generalising any better.
            detector.model_.fit(detector._as_model_input(X[fit_idx]))

        if custom_objective is not None:
            value = float(custom_objective(detector, X))
        elif use_supervised:
            scores = detector.score_samples(X[eval_idx])
            value = _supervised_score(y_eval, scores, metric_name)
        elif unsupervised_kind == "tail_separation":
            scores = detector.score_samples(X[eval_idx])
            value = _tail_separation(scores)
        else:
            value = _rank_agreement(
                detector_kwargs, X, fit_idx, eval_idx, random_state, contamination
            )

        log.debug(
            "Trial %d: value=%.6f params=%s", trial.number, value, trial.params
        )
        return value

    suffix = study_tag or _study_fingerprint(X, feature_names, mode, direction)
    study_name = f"{study_name}_{suffix}"

    from src.models._tuning_budget import tpe_startup_trials

    # Scaled to the budget rather than Optuna's fixed default of 10, which
    # would leave only 5 of the default 15 trials actually guided by TPE (and
    # 0 of 5 under --quick). See `_tuning_budget` for the numbers.
    sampler = optuna.samplers.TPESampler(
        seed=random_state, n_startup_trials=tpe_startup_trials(n_trials),
    )
    from src.models._optuna_storage import resolve_storage

    study = optuna.create_study(
        study_name=study_name,
        storage=resolve_storage(storage),
        direction=direction,
        sampler=sampler,
        load_if_exists=True,  # <-- crash-recovery / resume switch
    )

    log.info(
        "Optuna tuning: study=%r storage=%r mode=%s direction=%s "
        "contamination=%.4g (fixed) new_trials=%d existing_trials=%d",
        study_name, storage, mode, direction, contamination,
        n_trials, len(study.trials),
    )

    progress = tqdm(total=n_trials, desc=f"optuna[{study_name}]", unit="trial")

    def _progress_callback(study_: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:
        progress.update(1)

    def _persist_best_callback(study_: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:
        """Checkpoint the current best hyperparameters to YAML after each trial."""
        try:
            best = study_.best_trial
        except (ValueError, RuntimeError):
            return  # no completed trial yet
        payload = {
            "study_name": study_name,
            "direction": direction,
            "best_value": float(best.value) if best.value is not None else None,
            "best_trial_number": best.number,
            "n_trials_completed": len(
                [t for t in study_.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "objective_mode": mode,
            "random_state": random_state,
            # `contamination` is reported as a fixed operating point, not as a
            # tuned value -- it is absent from `raw_optuna_params` by design.
            "contamination": float(contamination),
            "holdout_frac": float(holdout_frac),
            "best_params": _detector_kwargs_from_params(best.params, contamination),
            "raw_optuna_params": dict(best.params),
        }
        tmp_path = best_params_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        atomic_replace(tmp_path, best_params_path)  # atomic swap, Windows-lock-safe
        log.info(
            "Checkpointed best params (trial %d, value=%.6f) -> %s",
            best.number, best.value if best.value is not None else float("nan"),
            best_params_path,
        )

    callbacks = [_progress_callback, _persist_best_callback]
    stopper = None
    if early_stopping_patience is not None:
        from src.models._tuning_stop import TrialPatienceStopper

        stopper = TrialPatienceStopper(
            direction=direction, model_name="iforest",
            n_trials_requested=n_trials + len(study.trials),
            patience=early_stopping_patience, min_delta=early_stopping_min_delta,
            min_trials=early_stopping_min_trials,
        )
        callbacks.append(stopper)

    with log_phase("iforest.tune (optuna)", log):
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks,
                gc_after_trial=True,
            )
        finally:
            progress.close()
        if stopper is not None and stopper.stopped:
            log.info(
                "iForest tuning stopped early: %s (%d trial(s) skipped).",
                stopper.stop_reason, stopper.trials_skipped,
            )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        log.warning("No completed trials; skipping final refit/save.")
        return study

    best_kwargs = _detector_kwargs_from_params(study.best_params, contamination)
    log.info(
        "Best trial %d value=%.6f params=%s",
        study.best_trial.number, study.best_value, best_kwargs,
    )
    # The final model is refit on ALL of X (fit + held-out): the split exists to
    # make model *selection* honest, not to throw away data once selected.
    with log_phase("iforest.refit_best", log):
        best_detector = IsolationForestDetector(
            random_state=random_state, n_jobs=-1, **best_kwargs
        )
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
