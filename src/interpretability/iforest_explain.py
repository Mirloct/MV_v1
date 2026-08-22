"""Interpretability for the Isolation Forest detector.

Two entry points, both saving figures under ``reports/figures/interpretability/``
(the project-wide figures rule) and returning plain ``dict``s of Python floats
so reporting can dump them straight to JSON/YAML:

* :func:`shap_summary_iforest` -- feature attribution via SHAP. It *tries*
  :class:`shap.TreeExplainer` first (native, exact tree attributions); whether
  ``TreeExplainer`` accepts a scikit-learn ``IsolationForest`` is
  **shap-version dependent**, so on failure it falls back to a model-agnostic
  :class:`shap.Explainer` over the detector's ``score_samples``, and finally to
  a manual permutation-importance measured directly on ``score_samples``. The
  path actually used is logged.
* :func:`path_length_analysis` -- ties the anomaly score back to the paper's
  isolation mechanism (anomalies have *shorter* normalized average path
  lengths) using scikit-learn's exact closed-form score, and saves a figure.

Score convention (project-wide): **higher score = more anomalous**
(``detector.score_samples`` = ``-sklearn.score_samples``).

The path-length mechanism paraphrased here follows the Isolation Forest concept
summarized in ``docs/geeksforgeeks_notes.md`` section 2 (anomalies are
isolated with shorter average path lengths); no verbatim copy.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import scipy.sparse as sp

from src.utils import paths
from src.utils.logging_config import log_phase, setup_logging

__all__ = ["shap_summary_iforest", "path_length_analysis"]

_DEFAULT_FIG_DIR = paths.FIGURES_DIR


def _densify(X) -> np.ndarray:
    """Return a dense contiguous ``float64`` 2D array (densifying sparse X)."""
    if sp.issparse(X):
        X = X.toarray()
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.ascontiguousarray(arr)


def _subsample(X: np.ndarray, max_samples: int, random_state: int = 42):
    """Row-subsample a dense matrix to ``max_samples`` (returns X_sub, idx)."""
    n = X.shape[0]
    if max_samples is None or n <= max_samples:
        return X, np.arange(n)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=int(max_samples), replace=False)
    idx.sort()
    return X[idx], idx


def _resolve_feature_names(feature_names, n_features):
    if feature_names is not None and len(feature_names) == n_features:
        return [str(f) for f in feature_names]
    return [f"f{i}" for i in range(n_features)]


def _importance_dict(names, importances) -> dict:
    """Build a feature->importance dict sorted by descending importance."""
    pairs = sorted(
        zip(names, (float(v) for v in importances)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {k: v for k, v in pairs}


def _save_bar_plot(names, importances, out_path, title, xlabel, top_n=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(importances)[::-1][:top_n]
    sel_names = [names[i] for i in order]
    sel_vals = [float(importances[i]) for i in order]

    height = max(3.0, 0.32 * len(sel_names) + 1.0)
    fig, ax = plt.subplots(figsize=(9, height))
    y_pos = np.arange(len(sel_names))
    ax.barh(y_pos, sel_vals, color="#4c72b0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sel_names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _permutation_importance(detector, X, random_state=42, n_repeats=3):
    """Manual permutation importance on ``score_samples``.

    For each feature, its column is randomly permuted and the mean absolute
    change in the (higher=more-anomalous) anomaly score is recorded, averaged
    over ``n_repeats`` shuffles. This is label-free and only reads how much each
    feature drives the anomaly score.
    """
    from tqdm.auto import tqdm

    rng = np.random.default_rng(random_state)
    base = detector.score_samples(X)
    n, d = X.shape
    imp = np.zeros(d, dtype=np.float64)
    for j in tqdm(range(d), desc="permutation_importance", unit="feature"):
        acc = 0.0
        col = X[:, j].copy()
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            X[:, j] = col[perm]
            shuffled = detector.score_samples(X)
            acc += float(np.mean(np.abs(shuffled - base)))
        X[:, j] = col  # restore
        imp[j] = acc / max(n_repeats, 1)
    return imp


def shap_summary_iforest(
    detector,
    X,
    feature_names=None,
    out_dir: str = _DEFAULT_FIG_DIR,
    max_samples: int = 2000,
    filename: str = "iforest_shap_summary.png",
    random_state: int = 42,
) -> dict:
    """Feature attribution for the Isolation Forest, saved as a figure.

    Tries, in order, until one works:

    1. :class:`shap.TreeExplainer` on ``detector.model_`` (native tree SHAP).
       Support for scikit-learn ``IsolationForest`` depends on the shap
       version, so this may raise and trigger the fallback.
    2. Model-agnostic :class:`shap.Explainer` over ``detector.score_samples``
       with a subsample masker.
    3. A manual permutation importance on ``detector.score_samples``.

    A SHAP beeswarm (paths 1-2) or a bar chart (path 3) is written under
    ``out_dir``; the logged message states which path was used.

    Args:
        detector: A fitted :class:`IsolationForestDetector` (``.model_`` is the
            underlying scikit-learn ``IsolationForest``).
        X: Preprocessed feature matrix (dense ndarray or scipy sparse).
        feature_names: Optional names aligned to ``X``'s columns.
        out_dir: Output directory for the figure (under ``reports/figures``).
        max_samples: Row subsample cap for speed.
        filename: Output PNG filename.
        random_state: Seed for subsampling / permutation.

    Returns:
        ``{feature: mean_abs_contribution}`` sorted descending.
    """
    log = setup_logging()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    Xd = _densify(X)
    Xd, _ = _subsample(Xd, max_samples, random_state)
    names = _resolve_feature_names(feature_names, Xd.shape[1])

    with log_phase("interpretability.shap_iforest", log):
        model = getattr(detector, "model_", None)
        if model is None:
            raise RuntimeError(
                "shap_summary_iforest requires a fitted detector (.model_ is None)."
            )

        shap_values = None
        used_path = None

        # -- path 1: native TreeExplainer (version-dependent) --------------- #
        try:
            import shap

            log.info(
                "Computing SHAP values via shap.TreeExplainer over %d row(s) x %d "
                "feature(s) (usually seconds, not minutes, on this path)...",
                Xd.shape[0], Xd.shape[1],
            )
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xd, check_additivity=False)
            log.info("shap.TreeExplainer finished.")
            if isinstance(sv, list):  # some versions wrap in a list
                sv = sv[0]
            shap_values = np.asarray(sv, dtype=np.float64)
            if shap_values.ndim == 3:  # (n, d, outputs)
                shap_values = shap_values[..., 0]
            used_path = "shap.TreeExplainer"
        except Exception as exc:  # noqa: BLE001 - version/support guard
            log.warning(
                "shap.TreeExplainer unavailable for IsolationForest (%s); "
                "falling back to model-agnostic Explainer.", exc,
            )

        # -- path 2: model-agnostic Explainer over score_samples ------------ #
        if shap_values is None:
            try:
                import shap

                # Background: a small subsample as the masker reference.
                bg_n = min(100, Xd.shape[0])
                bg_idx = np.random.default_rng(random_state).choice(
                    Xd.shape[0], size=bg_n, replace=False
                )
                background = Xd[bg_idx]

                def _score_fn(data):
                    return np.asarray(detector.score_samples(data), dtype=np.float64)

                masker = shap.maskers.Independent(background)
                explainer = shap.Explainer(_score_fn, masker)
                # This path calls `_score_fn` (which re-scores the whole
                # forest) many times per explained row and has no built-in
                # progress log of its own -- on a few thousand rows this can
                # run for minutes with nothing printed, which reads as a
                # hang. TreeExplainer (path 1, tried first) normally avoids
                # this path entirely; log explicitly for when it doesn't.
                log.info(
                    "shap.TreeExplainer unavailable -- falling back to the "
                    "model-agnostic Explainer over %d row(s) x %d feature(s); "
                    "this path re-scores the forest many times per row and can "
                    "take minutes with no further log output until it finishes.",
                    Xd.shape[0], Xd.shape[1],
                )
                explanation = explainer(Xd)
                shap_values = np.asarray(explanation.values, dtype=np.float64)
                if shap_values.ndim == 3:
                    shap_values = shap_values[..., 0]
                used_path = "shap.Explainer(score_samples)"
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "shap.Explainer fallback failed (%s); using permutation "
                    "importance on score_samples.", exc,
                )

        # -- attribution + beeswarm when SHAP produced values --------------- #
        if shap_values is not None and shap_values.shape[1] == Xd.shape[1]:
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            importance = _importance_dict(names, mean_abs)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import shap

                log.info(
                    "Rendering SHAP beeswarm plot (%d rows x %d features)...",
                    Xd.shape[0], Xd.shape[1],
                )
                plt.figure()
                # `shap.summary_plot` seeds the *global* NumPy RNG internally,
                # which NumPy now warns about. The call site cannot avoid it --
                # it is inside the library -- and the warning is emitted on
                # every run, burying the real log output. Suppressed narrowly
                # (this one call, this one category) rather than globally, so a
                # genuine FutureWarning from our own code still surfaces.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=FutureWarning,
                        message=".*global RNG.*",
                    )
                    shap.summary_plot(
                        shap_values, Xd, feature_names=names, show=False,
                        plot_size=None,
                    )
                plt.tight_layout()
                plt.savefig(out_path, dpi=120, bbox_inches="tight")
                plt.close("all")
                log.info("SHAP beeswarm plot saved.")
            except Exception as exc:  # noqa: BLE001 - beeswarm may be unavailable
                log.warning(
                    "SHAP beeswarm plot failed (%s); writing mean|SHAP| bar "
                    "chart instead.", exc,
                )
                _save_bar_plot(
                    names, mean_abs, out_path,
                    title="Isolation Forest -- mean|SHAP| feature importance",
                    xlabel="mean |SHAP value|",
                )
            log.info(
                "SHAP summary (%s) saved to %s; top feature=%s.",
                used_path, out_path, next(iter(importance)) if importance else "n/a",
            )
            return importance

        # -- path 3: permutation importance fallback ------------------------ #
        used_path = "permutation_importance(score_samples)"
        imp = _permutation_importance(detector, Xd, random_state=random_state)
        importance = _importance_dict(names, imp)
        _save_bar_plot(
            names, imp, out_path,
            title="Isolation Forest -- permutation importance (score_samples)",
            xlabel="mean |Δ anomaly score| when feature permuted",
        )
        log.info(
            "Feature importance (%s) saved to %s; top feature=%s.",
            used_path, out_path, next(iter(importance)) if importance else "n/a",
        )
        return importance


def path_length_analysis(
    detector,
    X,
    out_dir: str = _DEFAULT_FIG_DIR,
    filename: str = "iforest_path_length.png",
    max_samples: int = 20000,
    random_state: int = 42,
) -> dict:
    """Relate the anomaly score to the (normalized) average path length.

    Mechanism (Liu, Ting & Zhou, 2008): a point isolated with a *shorter*
    average path length is more anomalous. scikit-learn's score is exactly
    ``s_sklearn = 2 ** (-E[h] / c(n))`` where ``E[h]`` is the mean path length
    over the forest and ``c(n)`` the expected unsuccessful-BST-search depth. The
    project detector exposes ``score_samples = -sklearn.score_samples``, i.e.
    ``s = 2 ** (-nd)`` with ``nd = E[h] / c(n)`` the *normalized average path
    length*. Inverting gives ``nd = -log2(s)`` -- an exact, closed-form
    per-sample normalized path length, no private-API access needed.

    A two-panel figure is saved: the anomaly-score distribution and a scatter of
    normalized path length vs. anomaly score (monotonically decreasing).

    Returns:
        Summary stats: score/path-length min/mean/max, their (negative)
        correlation, and ``n_samples``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = setup_logging()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    with log_phase("interpretability.path_length_iforest", log):
        Xd = _densify(X)
        Xd, _ = _subsample(Xd, max_samples, random_state)
        scores = np.asarray(detector.score_samples(Xd), dtype=np.float64).ravel()

        # s = 2 ** (-nd)  ->  nd = -log2(s). Guard against non-positive scores.
        safe = np.clip(scores, 1e-12, None)
        norm_path_len = -np.log2(safe)

        if scores.size > 1 and np.std(scores) > 0 and np.std(norm_path_len) > 0:
            corr = float(np.corrcoef(scores, norm_path_len)[0, 1])
        else:
            corr = float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(scores, bins=60, color="#4c72b0", alpha=0.85)
        axes[0].set_title("Anomaly-score distribution")
        axes[0].set_xlabel("anomaly score (higher = more anomalous)")
        axes[0].set_ylabel("count")

        n_plot = min(scores.size, 20000)
        if scores.size > n_plot:
            pidx = np.random.default_rng(random_state).choice(
                scores.size, size=n_plot, replace=False
            )
        else:
            pidx = np.arange(scores.size)
        axes[1].scatter(
            scores[pidx], norm_path_len[pidx], s=6, alpha=0.35,
            color="#c44e52", linewidths=0,
        )
        axes[1].set_title(
            "Normalized average path length vs. anomaly score\n"
            "(shorter path <-> higher score <-> more anomalous)"
        )
        axes[1].set_xlabel("anomaly score")
        axes[1].set_ylabel("normalized avg path length  nd = -log2(score)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        summary = {
            "n_samples": int(scores.size),
            "score_min": float(np.min(scores)),
            "score_mean": float(np.mean(scores)),
            "score_max": float(np.max(scores)),
            "path_length_min": float(np.min(norm_path_len)),
            "path_length_mean": float(np.mean(norm_path_len)),
            "path_length_max": float(np.max(norm_path_len)),
            "score_pathlen_corr": corr,
            "figure_path": os.path.abspath(out_path),
            # The plotted points themselves, already subsampled to `pidx`, so
            # the report can re-render this as an interactive chart instead of
            # embedding the PNG. Additive: every key above is unchanged.
            "plot_scores": scores[pidx].tolist(),
            "plot_path_lengths": norm_path_len[pidx].tolist(),
        }
        log.info(
            "Path-length analysis saved to %s (corr(score, path_len)=%.4f).",
            out_path, corr,
        )
        return summary
