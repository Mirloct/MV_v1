"""Interpretability for the VAE detector.

Two entry points, both saving figures under ``reports/figures/interpretability/``:

* :func:`latent_space_plot` -- encode ``X`` to latent means, reduce to 2D (UMAP
  when importable, else PCA) and scatter, colored by labels ``y`` when given
  else by the reconstruction-error anomaly score.
* :func:`reconstruction_error_by_feature` -- the per-*feature* mean squared
  reconstruction error, i.e. which columns the VAE reconstructs worst and thus
  which features drive the anomaly score. Runs the torch model in eval / no-grad
  in batches, densifying sparse input.
* :func:`explain_rows_vae` -- *per-row* explanation (which features drove
  *this specific* row's reconstruction error), the complement of
  `reconstruction_error_by_feature`'s aggregate, population-level ranking.

Score convention (project-wide): **higher score = more anomalous**, where the
VAE score is the per-row MSE reconstruction error.

The reconstruction-error interpretation paraphrased here follows the autoencoder
concept summarized in ``docs/geeksforgeeks_notes.md`` section 3 (a
model trained on the normal bulk reconstructs anomalous inputs poorly); no
verbatim copy.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
import torch

from src.preprocessing.pipeline import (
    aggregate_attribution_by_source,
    categorical_feature_mask,
    group_name_by_source,
)
from src.utils import observability, paths
from src.utils.logging_config import log_phase, setup_logging

__all__ = [
    "latent_space_plot",
    "reconstruction_error_by_feature",
    "explain_rows_vae",
]

_DEFAULT_FIG_DIR = paths.FIGURES_DIR


def _checkpoint(name: str, **observed) -> None:
    """Always-passing progress checkpoint -- see the twin helper in
    `iforest_explain.py` for why (diagnosing *where* a stall happened from
    the last-recorded name in `run_events.jsonl`, not a pass/fail verdict).
    """
    observability.check(
        name=f"interpretability.vae_explain.{name}",
        category="validation",
        definition="Progress checkpoint inside the VAE interpretability "
                    "functions -- always passes; its presence (or absence) "
                    "in the health-check log is the diagnostic.",
        expected="reached without hanging",
        severity="info",
        passed=True,
        observed=observed,
    )


def _densify(X) -> np.ndarray:
    """Return a dense contiguous ``float32`` 2D array (densifying sparse X)."""
    if sp.issparse(X):
        X = X.toarray()
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _n_rows(X) -> int:
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return int(np.asarray(X).shape[0])


def latent_space_plot(
    detector,
    X,
    out_dir: str = _DEFAULT_FIG_DIR,
    y=None,
    filename: str = "vae_latent_space.png",
    max_points: int = 50000,
    random_state: int = 42,
    return_data: bool = False,
):
    """Encode ``X`` to latent means, reduce to 2D (UMAP else PCA) and scatter.

    Colored by ``y`` (0/1 labels) when given, otherwise by the reconstruction
    -error anomaly score. Uses the non-interactive ``Agg`` backend and writes
    under ``reports/figures/interpretability/``.

    Args:
        return_data: When True, also return the plotted coordinates so the
            report can re-render this interactively instead of embedding the
            PNG. Off by default so the historical ``-> str`` contract (and
            every existing caller) is untouched.

    Returns:
        The absolute path of the written PNG, or ``(path, data)`` when
        ``return_data`` is set. ``data`` is
        ``{"x", "y", "color", "color_label", "method"}`` with plain lists.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = setup_logging()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    with log_phase("interpretability.vae_latent", log):
        _checkpoint("started", n_rows=int(_n_rows(X)))
        y_arr = None if y is None else np.asarray(y).ravel()

        n = _n_rows(X)
        idx = None
        if n > max_points:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=max_points, replace=False)
            idx.sort()
            X_sub = X[idx]
            if y_arr is not None:
                y_arr = y_arr[idx]
        else:
            X_sub = X

        latents = detector.encode(X_sub)  # (n_sub, latent_dim), encoder means
        _checkpoint("encoded", n_rows=int(X_sub.shape[0]), latent_dim=int(latents.shape[1]))

        # Reduce to 2D: UMAP if importable, else PCA (1D latent -> pad).
        method = "raw"
        if latents.shape[1] == 2:
            coords = np.asarray(latents, dtype=np.float64)
            method = "latent"
        elif latents.shape[1] == 1:
            coords = np.hstack(
                [latents, np.zeros((latents.shape[0], 1), dtype=latents.dtype)]
            ).astype(np.float64)
            method = "latent"
        else:
            coords = None
            try:
                import umap  # umap-learn

                # UMAP's first call in a process pays a one-time numba JIT-
                # compilation cost (measured ~24s for 6000 x 8 on this
                # machine, before any actual fit work starts) with zero
                # console output the whole time -- log this explicitly so a
                # silent multi-second/multi-minute gap here reads as "still
                # working" rather than "the pipeline froze".
                log.info(
                    "Running UMAP on %d point(s) (latent_dim=%d) -- the first UMAP "
                    "call in a process includes a one-time numba compilation warmup "
                    "with no further log output until it finishes; this is normal.",
                    latents.shape[0], latents.shape[1],
                )
                _checkpoint("umap_started", n_points=int(latents.shape[0]))
                # n_jobs=1 silences UMAP's own warning: a fixed random_state
                # forces single-threaded execution internally regardless, and
                # UMAP warns if n_jobs isn't already set to 1.
                reducer = umap.UMAP(n_components=2, random_state=random_state, n_jobs=1)
                coords = np.asarray(reducer.fit_transform(latents), dtype=np.float64)
                method = "UMAP"
                _checkpoint("umap_done")
            except Exception as exc:  # noqa: BLE001 - fall back to PCA
                log.warning("UMAP unavailable (%s); using PCA for latent 2D.", exc)
                _checkpoint("umap_failed", error=str(exc))
            if coords is None:
                from sklearn.decomposition import PCA

                coords = PCA(
                    n_components=2, random_state=random_state
                ).fit_transform(latents)
                method = "PCA"

        fig, ax = plt.subplots(figsize=(7.5, 6))
        if y_arr is not None and np.unique(y_arr).size > 1:
            normal = y_arr == 0
            anomaly = y_arr == 1
            ax.scatter(coords[normal, 0], coords[normal, 1], s=6, alpha=0.35,
                       label="normal", color="#4c72b0", linewidths=0)
            ax.scatter(coords[anomaly, 0], coords[anomaly, 1], s=12, alpha=0.75,
                       label="anomaly", color="#c44e52", linewidths=0)
            ax.legend()
            cbar = None
            colour = y_arr.astype(float)
            colour_label = "etiqueta (1 = anomalía)"
        else:
            scores = np.asarray(
                detector.score_samples(X_sub), dtype=np.float64
            ).ravel()
            sc = ax.scatter(
                coords[:, 0], coords[:, 1], s=8, alpha=0.5, c=scores,
                cmap="viridis", linewidths=0,
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("anomaly score (recon error)")
            colour = scores
            colour_label = "puntaje (error de reconstrucción)"

        ax.set_title(f"VAE latent space ({method} of encoder means)")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        log.info("Saved VAE latent-space figure (%s) to %s.", method, out_path)
        _checkpoint("completed", method=method)
        path = os.path.abspath(out_path)
        if not return_data:
            return path
        return path, {
            "x": np.asarray(coords[:, 0], dtype=float).tolist(),
            "y": np.asarray(coords[:, 1], dtype=float).tolist(),
            "color": np.asarray(colour, dtype=float).tolist(),
            "color_label": colour_label,
            "method": method,
        }


def reconstruction_error_by_feature(
    detector,
    X,
    feature_names=None,
    out_dir: str = _DEFAULT_FIG_DIR,
    filename: str = "vae_recon_by_feature.png",
    max_samples: int = 2000,
    top_n: int = 30,
    random_state: int = 42,
    categorical_columns=None,
) -> dict:
    """Per-feature mean squared reconstruction error of the VAE.

    Computes, for each input column ``j``, ``mean_i (x_ij - x_recon_ij)^2`` over
    the (subsampled) rows, using the encoder mean (deterministic, eval / no-grad)
    exactly as :meth:`VAEDetector.score_samples` does per row. The largest
    per-feature errors identify which features the VAE reconstructs worst and
    therefore which drive the anomaly score. A bar chart of the top features is
    saved under ``reports/figures/interpretability/``.

    Args:
        categorical_columns: Optional list of the *original* (pre-transform)
            categorical column names (e.g.
            ``df.select_dtypes(include=["object", "category"]).columns``).
            When given, two things change: (1) the bar chart ranks and labels
            by *original* variable, summing one-hot-derived columns back
            together, since a high-cardinality categorical otherwise crowds
            the top-N purely by column count, not by how much it actually
            costs to reconstruct; (2) a checkpoint records what share of the
            total reconstruction error (and thus of the VAE's anomaly score,
            since that error sums directly into it) comes from
            categorical-derived columns vs. everything else, so "are my
            string columns dominating the score, not just the report" is a
            measured fact instead of a guess. Omit to keep the historical
            per-column-only behavior (every existing caller's chart is
            unaffected; only a caller that opts in gets the grouped view).

    Returns:
        ``{feature: mean_recon_error}`` sorted descending -- always the full,
        ungrouped per-column detail, regardless of ``categorical_columns``
        (the uncropped `attribution_export.py` workbook needs this; grouping
        is applied only to the chart / diagnostic, never silently to the
        returned data).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = setup_logging()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    with log_phase("interpretability.vae_recon_by_feature", log):
        model = getattr(detector, "model_", None)
        if model is None:
            raise RuntimeError(
                "reconstruction_error_by_feature requires a fitted detector "
                "(.model_ is None)."
            )

        Xd = _densify(X)
        n = Xd.shape[0]
        if max_samples is not None and n > max_samples:
            rng = np.random.default_rng(random_state)
            sel = rng.choice(n, size=int(max_samples), replace=False)
            sel.sort()
            Xd = Xd[sel]

        n_rows, n_features = Xd.shape
        device = getattr(detector, "device", "cpu")
        batch_size = int(getattr(detector, "batch_size", 256) or 256)
        _checkpoint("started", n_rows=int(n_rows), n_features=int(n_features))

        sq_err_sum = np.zeros(n_features, dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for start in range(0, n_rows, batch_size):
                chunk = Xd[start:start + batch_size]
                xb = torch.from_numpy(chunk).to(device)
                mu, _ = model.encode(xb)
                x_recon = model.decode(mu)
                sq = (xb - x_recon) ** 2  # (batch, n_features)
                sq_err_sum += sq.sum(dim=0).cpu().numpy().astype(np.float64)

        mean_err = sq_err_sum / max(n_rows, 1)
        _checkpoint("batches_done", n_rows=int(n_rows))

        if feature_names is not None and len(feature_names) == n_features:
            names = [str(f) for f in feature_names]
        else:
            names = [f"f{i}" for i in range(n_features)]

        pairs = sorted(
            zip(names, (float(v) for v in mean_err)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        result = {k: v for k, v in pairs}

        # -- categorical-vs-numeric contribution diagnostic ------------------ #
        # This is not just a chart-cosmetics measurement: `mean_err` sums
        # directly into `score_samples` (the VAE's anomaly score), so the
        # categorical share of the total *is* the categorical share of what
        # drives every ranking downstream, not only this chart's.
        if categorical_columns:
            cat_mask = categorical_feature_mask(names)
            total_err = float(np.sum(mean_err))
            cat_err = float(np.sum(mean_err[cat_mask])) if cat_mask.any() else 0.0
            cat_share = (cat_err / total_err) if total_err > 0 else float("nan")
            n_cat_cols = int(cat_mask.sum())
            col_share = n_cat_cols / max(n_features, 1)
            log.info(
                "VAE reconstruction error by source: categorical-derived "
                "columns are %d/%d (%.1f%%) of features but %.1f%% of total "
                "reconstruction error (score contribution) -- %s.",
                n_cat_cols, n_features, 100.0 * col_share, 100.0 * cat_share,
                "over-represented" if cat_share > col_share + 0.10
                else "roughly proportional" if abs(cat_share - col_share) <= 0.10
                else "under-represented",
            )
            _checkpoint(
                "categorical_contribution",
                n_categorical_columns=n_cat_cols, n_total_columns=int(n_features),
                column_share=round(col_share, 2), error_share=round(cat_share, 2),
            )

        # -- chart: grouped by original source column when available -------- #
        # A high-cardinality categorical otherwise crowds the top-N purely by
        # column count (20 one-hot slices of "region" vs. 1 "income" column),
        # not by how much it actually costs to reconstruct.
        if categorical_columns:
            chart_values = aggregate_attribution_by_source(result, categorical_columns)
            chart_title = (
                "VAE reconstruction error by source variable "
                "(categorical columns summed back together)"
            )
        else:
            chart_values = result
            chart_title = "VAE per-feature reconstruction error (worst-reconstructed first)"

        chart_pairs = list(chart_values.items())[:top_n]
        sel_names = [k for k, _ in chart_pairs]
        sel_vals = [v for _, v in chart_pairs]
        height = max(3.0, 0.32 * len(sel_names) + 1.0)
        fig, ax = plt.subplots(figsize=(9, height))
        y_pos = np.arange(len(sel_names))
        ax.barh(y_pos, sel_vals, color="#c44e52")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sel_names)
        ax.invert_yaxis()
        ax.set_xlabel("mean squared reconstruction error")
        ax.set_title(chart_title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        log.info(
            "Saved VAE per-feature reconstruction-error figure to %s; "
            "worst feature=%s.", out_path,
            sel_names[0] if sel_names else "n/a",
        )
        _checkpoint("completed")
        return result


def explain_rows_vae(
    detector,
    X,
    feature_names=None,
    top_k: int = 5,
    categorical_columns=None,
    batch_size=None,
):
    """Per-row explanation: the ``top_k`` features (by squared reconstruction
    error) driving *each individual row's* anomaly score.

    Complement of `reconstruction_error_by_feature`: that function aggregates
    over rows to answer "which features matter most overall"; this answers
    "why is *this* row flagged" for a small, specific set of rows (typically
    an OOT alert queue) -- suitable for a spreadsheet column read next to
    each case. Unlike the Isolation Forest's SHAP-based
    `explain_rows_iforest`, this is always exactly computable (no fallback
    needed): squared reconstruction error is deterministic and additive by
    construction, the same quantity `score_samples` sums.

    Args:
        detector: A fitted :class:`VAEDetector`.
        X: Preprocessed feature matrix for exactly the rows to explain (dense
            ndarray or scipy sparse) -- every row is used, not subsampled.
        feature_names: Optional names aligned to ``X``'s columns.
        top_k: How many feature names to report per row.
        categorical_columns: Optional original (pre-transform) categorical
            column names. When given, one-hot-derived columns are summed
            back under their source variable **per row** before ranking --
            same rationale as `reconstruction_error_by_feature`'s aggregate
            diagnostic (`CONTEXT.md` "VAE feature attribution: categorical
            granularity"), applied row-by-row instead of in aggregate.
        batch_size: Rows per forward pass; defaults to the detector's own.

    Returns:
        A list of length ``len(X)``, same row order: each entry is a
        comma-joined string of the ``top_k`` feature/source names for that
        row (descending by squared reconstruction error).
    """
    model = getattr(detector, "model_", None)
    n_rows_requested = int(np.asarray(X).shape[0]) if hasattr(X, "shape") else len(X)
    if model is None or n_rows_requested == 0:
        return [None] * n_rows_requested

    _checkpoint("explain_rows_started", n_rows=n_rows_requested)

    Xd = _densify(X)
    n_rows, n_features = Xd.shape
    names = (
        [str(f) for f in feature_names]
        if feature_names is not None and len(feature_names) == n_features
        else [f"f{i}" for i in range(n_features)]
    )
    device = getattr(detector, "device", "cpu")
    bs = int(batch_size or getattr(detector, "batch_size", 256) or 256)
    k = max(1, min(top_k, n_features))

    # Precompute a per-column -> group-index map once, so per-row aggregation
    # is a cheap segment-sum instead of recomputing string matching per row.
    if categorical_columns:
        groups = [group_name_by_source(n, categorical_columns) for n in names]
        unique_groups = sorted(set(groups))
        group_of = {g: i for i, g in enumerate(unique_groups)}
        col_to_group = np.array([group_of[g] for g in groups], dtype=np.int64)
        report_names = unique_groups
    else:
        col_to_group = None
        report_names = names

    explanations = []
    n_batches = max(1, -(-n_rows // bs))  # ceil division
    # A checkpoint roughly every quarter of the batches -- always fast in
    # practice (one forward pass, no subprocess), but a very large alert
    # queue should still leave a trace partway through, not just at the end.
    checkpoint_every = max(1, n_batches // 4)
    model.eval()
    with torch.no_grad():
        for b, start in enumerate(range(0, n_rows, bs)):
            chunk = Xd[start:start + bs]
            xb = torch.from_numpy(chunk).to(device)
            mu, _ = model.encode(xb)
            x_recon = model.decode(mu)
            sq = ((xb - x_recon) ** 2).cpu().numpy().astype(np.float64)  # (batch, n_features)
            for row in sq:
                if col_to_group is not None:
                    grouped = np.zeros(len(report_names), dtype=np.float64)
                    np.add.at(grouped, col_to_group, row)
                    order = np.argsort(-grouped)[:k]
                else:
                    order = np.argsort(-row)[:k]
                explanations.append(", ".join(report_names[j] for j in order))
            if (b + 1) % checkpoint_every == 0 or (b + 1) == n_batches:
                _checkpoint(
                    "explain_rows_progress", rows_done=len(explanations),
                    rows_total=n_rows_requested,
                )
    _checkpoint("explain_rows_completed", n_explained=len(explanations), n_requested=n_rows_requested)
    return explanations
