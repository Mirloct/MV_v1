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
* :func:`explain_rows_iforest` -- *per-row* explanation (which features drove
  *this specific* row's score), for a small, specific set of rows (typically
  an alert queue) rather than a representative subsample -- the complement of
  :func:`shap_summary_iforest`'s aggregate, population-level ranking.

Score convention (project-wide): **higher score = more anomalous**
(``detector.score_samples`` = ``-sklearn.score_samples``).

The path-length mechanism paraphrased here follows the Isolation Forest concept
summarized in ``docs/geeksforgeeks_notes.md`` section 2 (anomalies are
isolated with shorter average path lengths); no verbatim copy.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import scipy.sparse as sp

from src.utils import observability, paths
from src.utils.logging_config import log_phase, setup_logging

__all__ = ["shap_summary_iforest", "path_length_analysis", "explain_rows_iforest"]

_DEFAULT_FIG_DIR = paths.FIGURES_DIR


def _checkpoint(name: str, **observed) -> None:
    """Record a lightweight, always-passing progress checkpoint.

    Shared by every function in this module (`shap_summary_iforest`,
    `path_length_analysis`, and `explain_rows_iforest`) so every part of
    Isolation Forest interpretability that actually runs leaves a trace, not
    just the SHAP paths.

    Purely diagnostic -- these never fail a run (this module already has its
    own try/except-per-path recovery in `main.py`'s caller). The point is
    what happens when the *process itself* stalls or is killed: the health
    checks in `artifacts/logs/run_events.jsonl` (and the console dashboard's
    live "Supuestos" panel, which shows every `observability.check(...)`
    project-wide) are append-only and flushed per call, so whichever
    `interpretability.iforest.*` name is *last* in the log is exactly the
    step that was in flight when things stopped -- e.g. a "..._calibration_
    started" with no matching "..._calibrated" after it means the calibration
    call itself (not the bounded, budgeted continuation) is what is hanging.
    """
    observability.check(
        name=f"interpretability.iforest.{name}",
        category="validation",
        definition="Progress checkpoint inside Isolation Forest "
                    "interpretability (shap_summary_iforest / "
                    "path_length_analysis) -- always passes; its presence "
                    "(or absence) in the health-check log is the diagnostic, "
                    "not a pass/fail verdict.",
        expected="reached without hanging",
        severity="info",
        passed=True,
        observed=observed,
    )

# Both SHAP fallback paths (model-agnostic Explainer, permutation importance)
# call `detector.score_samples` many times -- once per (row x ~2*n_features)
# evaluation for the former, once per (feature x repeat) for the latter -- so
# their wall-clock cost scales with the number of *features*, not just rows.
# Measured on this project's synthetic panel: at 180 features the
# model-agnostic path costs ~4.8s per explained row (~2.7 hours for the 2000
# rows `shap_summary_iforest` explains by default), which is exactly the
# "the process hangs / seems to crash during interpretability" failure a
# feature-rich real dataset (150-200 columns) triggers, since the 22-67
# feature synthetic panel this project was developed against never spent
# enough time in either fallback to notice. Both fallbacks are therefore kept
# time/call-budgeted below instead of running unbounded.
_MODEL_AGNOSTIC_TIME_BUDGET_S = 60
_MODEL_AGNOSTIC_CALIBRATION_ROWS = 2
_MODEL_AGNOSTIC_BACKGROUND_ROWS = 100
_PERM_IMPORTANCE_MAX_SAMPLES = 1000
_PERM_IMPORTANCE_CALL_BUDGET = 150  # total score_samples() calls across (features x repeats)

# shap.TreeExplainer (path 1) was assumed fast enough not to need budgeting --
# true on every configuration measured directly (up to 600 trees, 200
# features, max_features down to 0.3, bootstrap either way: 2-23s). Confirmed
# in production against a real feature-rich dataset that it can still hang
# well past that on tree/data shapes not reproduced in those tests (root
# cause found 2026-08-28: `max_samples` tuned as a float fraction of a
# multi-month training block builds much deeper trees than the `"auto"`
# default, and TreeExplainer's cost scales with tree depth/leaf count), so it
# now gets the same calibrate-then-bound treatment as path 2.
_TREE_EXPLAINER_TIME_BUDGET_S = 90
_TREE_EXPLAINER_CALIBRATION_ROWS = 5

# A soft time budget only protects against slowness *after* calibration
# returns. Confirmed in production (2026-08-28) that this is not always
# enough on its own: a real run hung 3+ hours with no forward progress, past
# every soft budget above. No in-process timer can stop a blocked native
# call (shap's C/Cython internals) from Python -- the only way to actually
# stop one is to run it in a separate OS process and kill that process if it
# overruns. Both `shap.TreeExplainer` and the model-agnostic `shap.Explainer`
# therefore run in a child process (`_run_with_hard_kill`), forcibly
# terminated (SIGTERM, then `Process.kill()` after a grace period) if they
# have not sent a final result within their ceiling -- at which point that
# path is treated as failed and the next one is tried, exactly as if it had
# raised an exception. The child still reports its calibration measurement
# back before starting the (bounded) full explain, so a kill does not erase
# the fine-grained checkpoints -- only the "done" one never arrives.
_TREE_EXPLAINER_HARD_KILL_S = 180   # ~2x the soft budget + spawn/pickling overhead margin
_MODEL_AGNOSTIC_HARD_KILL_S = 150
_HARD_KILL_GRACE_S = 5              # time given to exit cleanly after terminate() before kill()
_HARD_KILL_POLL_S = 1               # how often the parent checks the child for a message


def _tree_explainer_child(model, Xd, calib_rows, time_budget_s, conn) -> None:
    """Runs in a child process spawned by `_run_with_hard_kill`.

    Isolated so the parent can enforce a real, wall-clock kill on
    `shap.TreeExplainer` that no in-process timer could apply to itself.
    Sends a `"calibrated"` progress message before attempting the full
    (budget-bounded) explain, so a later hard-kill still leaves the parent
    with that measurement instead of nothing.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        calib_n = min(calib_rows, Xd.shape[0])
        t0 = time.monotonic()
        sv = explainer.shap_values(Xd[:calib_n], check_additivity=False)
        calib_dt = time.monotonic() - t0
        per_row = calib_dt / max(calib_n, 1)

        if calib_n < Xd.shape[0] and per_row > 0:
            extra_rows = max(0, int((time_budget_s - calib_dt) / per_row))
            n_explain = min(Xd.shape[0], calib_n + extra_rows)
        else:
            n_explain = Xd.shape[0]
        conn.send({
            "stage": "calibrated", "calib_n": calib_n, "calib_dt": calib_dt,
            "per_row": per_row, "n_explain": n_explain,
        })

        if n_explain > calib_n:
            sv = explainer.shap_values(Xd[:n_explain], check_additivity=False)
        conn.send({"stage": "done", "sv": sv, "n_explain": n_explain})
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not raised here
        conn.send({"stage": "error", "error": str(exc)})
    finally:
        conn.close()


def _model_agnostic_child(
    detector, Xd, calib_rows, time_budget_s, bg_rows, random_state, conn,
) -> None:
    """Same isolation/kill rationale as `_tree_explainer_child`, for the
    model-agnostic `shap.Explainer` fallback."""
    try:
        import shap

        bg_n = min(bg_rows, Xd.shape[0])
        bg_idx = np.random.default_rng(random_state).choice(
            Xd.shape[0], size=bg_n, replace=False
        )
        background = Xd[bg_idx]

        def _score_fn(data):
            return np.asarray(detector.score_samples(data), dtype=np.float64)

        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(_score_fn, masker)

        calib_n = min(calib_rows, Xd.shape[0])
        t0 = time.monotonic()
        explanation = explainer(Xd[:calib_n])
        calib_dt = time.monotonic() - t0
        per_row = calib_dt / max(calib_n, 1)

        if per_row > 0:
            extra_rows = max(0, int((time_budget_s - calib_dt) / per_row))
        else:
            extra_rows = Xd.shape[0] - calib_n
        n_explain = min(Xd.shape[0], calib_n + extra_rows)
        conn.send({
            "stage": "calibrated", "calib_n": calib_n, "calib_dt": calib_dt,
            "per_row": per_row, "n_explain": n_explain,
        })

        if n_explain > calib_n:
            explanation = explainer(Xd[:n_explain])
        values = np.asarray(explanation.values, dtype=np.float64)
        conn.send({"stage": "done", "values": values, "n_explain": n_explain})
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not raised here
        conn.send({"stage": "error", "error": str(exc)})
    finally:
        conn.close()


def _run_with_hard_kill(target, args, hard_timeout_s, on_progress=None):
    """Run ``target(*args, conn)`` in a fresh child process; force-kill it if
    it has not sent a final (``"done"``/``"error"``) message within
    ``hard_timeout_s``.

    ``on_progress(msg)`` runs in the parent for every non-final message the
    child sends (its calibration measurement) -- this is what keeps this
    project's fine-grained checkpoints (`_checkpoint`, above) working even
    when the child is eventually killed for taking too long overall: whatever
    it reported before being killed is not lost.

    Returns the child's final message dict, or ``{"stage": "hard_killed"}``
    if the ceiling was reached with no final message received.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(*args, child_conn))
    proc.start()
    child_conn.close()

    deadline = time.monotonic() + hard_timeout_s
    final = None
    try:
        while final is None and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if parent_conn.poll(timeout=min(_HARD_KILL_POLL_S, remaining)):
                try:
                    msg = parent_conn.recv()
                except (EOFError, OSError):
                    break
                if msg.get("stage") in ("done", "error"):
                    final = msg
                elif on_progress is not None:
                    on_progress(msg)
            elif not proc.is_alive():
                break  # child exited without a final message (e.g. a crash)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(_HARD_KILL_GRACE_S)
            if proc.is_alive():
                proc.kill()
                proc.join()
        parent_conn.close()

    return final if final is not None else {"stage": "hard_killed"}


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


def _permutation_importance(
    detector,
    X,
    random_state=42,
    n_repeats=3,
    max_samples=_PERM_IMPORTANCE_MAX_SAMPLES,
    call_budget=_PERM_IMPORTANCE_CALL_BUDGET,
):
    """Manual permutation importance on ``score_samples``.

    For each feature, its column is randomly permuted and the mean absolute
    change in the (higher=more-anomalous) anomaly score is recorded, averaged
    over ``n_repeats`` shuffles. This is label-free and only reads how much each
    feature drives the anomaly score.

    Cost is ``d * n_repeats`` calls to ``score_samples`` over ``n`` rows, which
    scales with the feature count -- at ~150-200 features and hundreds of
    trees this can reach minutes even though it looked instant on a
    dozen-feature panel. Bounded two ways: rows are subsampled to
    ``max_samples`` (this fallback needs far fewer rows than the SHAP paths
    for a stable ranking), and ``n_repeats`` is reduced so total calls never
    exceed ``call_budget``.
    """
    from tqdm.auto import tqdm

    log = setup_logging()
    X, _ = _subsample(X, max_samples, random_state)
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    n_repeats_eff = max(1, min(n_repeats, call_budget // max(d, 1)))
    if n_repeats_eff < n_repeats:
        log.info(
            "permutation_importance: %d features x %d requested repeat(s) would "
            "cost %d score_samples() calls over %d row(s); capping to %d "
            "repeat(s) (%d calls total) to keep this bounded.",
            d, n_repeats, d * n_repeats, n, n_repeats_eff, d * n_repeats_eff,
        )
    _checkpoint(
        "permutation_importance_repeats_set", n_features=int(d), n_rows=int(n),
        requested_repeats=int(n_repeats), effective_repeats=int(n_repeats_eff),
        total_calls=int(d * n_repeats_eff), capped=bool(n_repeats_eff < n_repeats),
    )

    base = detector.score_samples(X)
    imp = np.zeros(d, dtype=np.float64)
    # A checkpoint roughly every quarter of the features (not per-feature --
    # that would be d health-check writes for what tqdm already shows live in
    # the console) so a stall shows up in run_events.jsonl within ~1/4 of the
    # loop's total budgeted time instead of only at the very end.
    checkpoint_every = max(1, d // 4)
    for j in tqdm(range(d), desc="permutation_importance", unit="feature"):
        acc = 0.0
        col = X[:, j].copy()
        for _ in range(n_repeats_eff):
            perm = rng.permutation(n)
            X[:, j] = col[perm]
            shuffled = detector.score_samples(X)
            acc += float(np.mean(np.abs(shuffled - base)))
        X[:, j] = col  # restore
        imp[j] = acc / max(n_repeats_eff, 1)
        if (j + 1) % checkpoint_every == 0 or j + 1 == d:
            _checkpoint("permutation_importance_progress", features_done=int(j + 1), features_total=int(d))
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

    _checkpoint("started", n_rows=int(Xd.shape[0]), n_features=int(Xd.shape[1]))

    with log_phase("interpretability.shap_iforest", log):
        model = getattr(detector, "model_", None)
        if model is None:
            raise RuntimeError(
                "shap_summary_iforest requires a fitted detector (.model_ is None)."
            )

        shap_values = None
        used_path = None

        # -- path 1: native TreeExplainer, isolated + hard-kill guarded ----- #
        try:
            import shap  # import check now; the real work runs in a child process

            calib_n = min(_TREE_EXPLAINER_CALIBRATION_ROWS, Xd.shape[0])
            log.info(
                "Computing SHAP values via shap.TreeExplainer in an isolated "
                "process -- calibrating on %d of %d row(s) x %d feature(s) "
                "first to bound this to a ~%ds soft budget, with a hard %ds "
                "kill ceiling if even that does not return.",
                calib_n, Xd.shape[0], Xd.shape[1], _TREE_EXPLAINER_TIME_BUDGET_S,
                _TREE_EXPLAINER_HARD_KILL_S,
            )
            # If the process is killed with THIS checkpoint as the last one
            # logged (no matching "tree_explainer_calibrated" after it), the
            # calibration call itself is what hung inside the child -- the
            # one sub-step no soft budget can preempt, which is exactly why
            # the hard kill below exists as a backstop.
            _checkpoint("tree_explainer_calibration_started", calib_rows=calib_n)

            n_total_rows = Xd.shape[0]

            def _on_progress(msg, _n_total=n_total_rows):
                _checkpoint(
                    "tree_explainer_calibrated", calib_rows=msg["calib_n"],
                    calib_seconds=round(msg["calib_dt"], 2),
                    seconds_per_row=round(msg["per_row"], 2),
                    planned_rows=int(msg["n_explain"]),
                    bounded=bool(msg["n_explain"] < _n_total),
                )
                log.info(
                    "shap.TreeExplainer calibration: %.2fs for %d row(s) "
                    "(~%.2fs/row) at %d feature(s); explaining %d of %d row(s).",
                    msg["calib_dt"], msg["calib_n"], msg["per_row"], Xd.shape[1],
                    msg["n_explain"], _n_total,
                )
                if msg["n_explain"] > msg["calib_n"]:
                    _checkpoint(
                        "tree_explainer_explain_started",
                        planned_rows=int(msg["n_explain"]),
                    )
                elif _n_total > msg["calib_n"]:
                    log.warning(
                        "shap.TreeExplainer is too slow at %d feature(s) "
                        "(~%.2fs/row); keeping only the %d calibration "
                        "row(s) explained instead of running unbounded.",
                        Xd.shape[1], msg["per_row"], msg["calib_n"],
                    )

            result = _run_with_hard_kill(
                _tree_explainer_child,
                (model, Xd, _TREE_EXPLAINER_CALIBRATION_ROWS, _TREE_EXPLAINER_TIME_BUDGET_S),
                _TREE_EXPLAINER_HARD_KILL_S,
                on_progress=_on_progress,
            )

            if result["stage"] == "hard_killed":
                _checkpoint(
                    "tree_explainer_hard_killed",
                    hard_timeout_s=_TREE_EXPLAINER_HARD_KILL_S,
                )
                raise RuntimeError(
                    f"shap.TreeExplainer did not finish within the hard "
                    f"{_TREE_EXPLAINER_HARD_KILL_S}s ceiling and was force-killed."
                )
            if result["stage"] == "error":
                raise RuntimeError(result["error"])

            sv = result["sv"]
            n_explain = result["n_explain"]
            log.info("shap.TreeExplainer finished.")
            _checkpoint("tree_explainer_done", rows_explained=int(min(n_explain, Xd.shape[0])))
            if isinstance(sv, list):  # some versions wrap in a list
                sv = sv[0]
            shap_values = np.asarray(sv, dtype=np.float64)
            if shap_values.ndim == 3:  # (n, d, outputs)
                shap_values = shap_values[..., 0]
            if shap_values.shape[0] != Xd.shape[0]:
                # The budget above may have explained fewer rows than Xd
                # holds; keep the beeswarm plot's feature matrix row-aligned
                # with the values actually computed.
                Xd = Xd[:shap_values.shape[0]]
            used_path = "shap.TreeExplainer"
        except Exception as exc:  # noqa: BLE001 - version/support guard
            log.warning(
                "shap.TreeExplainer unavailable for IsolationForest (%s); "
                "falling back to model-agnostic Explainer.", exc,
            )
            _checkpoint("tree_explainer_failed", error=str(exc))

        # -- path 2: model-agnostic Explainer, isolated + hard-kill guarded - #
        if shap_values is None:
            try:
                import shap  # import check now; the real work runs in a child process

                # This path calls `score_samples` (which re-scores the whole
                # forest) roughly `2 * n_features + 1` times *per explained
                # row*, so its cost scales with the feature count, not just
                # the row count -- measured at ~4.8s/row for 180 features,
                # i.e. ~2.7 hours to explain 2000 rows unbounded. That is the
                # "the pipeline hangs/crashes during interpretability" failure
                # on a feature-rich real dataset; it never showed up on this
                # project's ~20-70 feature synthetic panel.
                calib_n = min(_MODEL_AGNOSTIC_CALIBRATION_ROWS, Xd.shape[0])
                log.info(
                    "shap.TreeExplainer unavailable -- falling back to the "
                    "model-agnostic Explainer (isolated process) over up to "
                    "%d row(s) x %d feature(s); calibrating on %d row(s) "
                    "first to bound this to a ~%ds soft budget, with a hard "
                    "%ds kill ceiling if even that does not return.",
                    Xd.shape[0], Xd.shape[1], calib_n,
                    _MODEL_AGNOSTIC_TIME_BUDGET_S, _MODEL_AGNOSTIC_HARD_KILL_S,
                )
                _checkpoint("model_agnostic_calibration_started", calib_rows=calib_n)

                n_total_rows = Xd.shape[0]

                def _on_progress(msg, _n_total=n_total_rows):
                    _checkpoint(
                        "model_agnostic_calibrated", calib_rows=msg["calib_n"],
                        calib_seconds=round(msg["calib_dt"], 2),
                        seconds_per_row=round(msg["per_row"], 2),
                        planned_rows=int(msg["n_explain"]),
                        bounded=bool(msg["n_explain"] < _n_total),
                    )
                    log.info(
                        "Model-agnostic SHAP calibration: %.2fs for %d row(s) "
                        "(~%.2fs/row) at %d feature(s); explaining %d of %d row(s).",
                        msg["calib_dt"], msg["calib_n"], msg["per_row"],
                        Xd.shape[1], msg["n_explain"], _n_total,
                    )
                    if msg["n_explain"] > msg["calib_n"]:
                        _checkpoint(
                            "model_agnostic_explain_started",
                            planned_rows=int(msg["n_explain"]),
                        )
                    elif _n_total > msg["calib_n"]:
                        log.warning(
                            "Model-agnostic SHAP is too slow at %d feature(s) "
                            "(~%.2fs/row); keeping only the %d calibration "
                            "row(s) explained instead of running unbounded.",
                            Xd.shape[1], msg["per_row"], msg["calib_n"],
                        )

                result = _run_with_hard_kill(
                    _model_agnostic_child,
                    (detector, Xd, _MODEL_AGNOSTIC_CALIBRATION_ROWS,
                     _MODEL_AGNOSTIC_TIME_BUDGET_S, _MODEL_AGNOSTIC_BACKGROUND_ROWS,
                     random_state),
                    _MODEL_AGNOSTIC_HARD_KILL_S,
                    on_progress=_on_progress,
                )

                if result["stage"] == "hard_killed":
                    _checkpoint(
                        "model_agnostic_hard_killed",
                        hard_timeout_s=_MODEL_AGNOSTIC_HARD_KILL_S,
                    )
                    raise RuntimeError(
                        f"model-agnostic shap.Explainer did not finish within "
                        f"the hard {_MODEL_AGNOSTIC_HARD_KILL_S}s ceiling and "
                        f"was force-killed."
                    )
                if result["stage"] == "error":
                    raise RuntimeError(result["error"])

                n_explain = result["n_explain"]
                log.info("Model-agnostic shap.Explainer finished.")
                _checkpoint("model_agnostic_done", rows_explained=int(min(n_explain, Xd.shape[0])))
                shap_values = np.asarray(result["values"], dtype=np.float64)
                if shap_values.ndim == 3:
                    shap_values = shap_values[..., 0]
                if shap_values.shape[0] != Xd.shape[0]:
                    # The budget above may have explained fewer rows than
                    # Xd holds; keep the beeswarm plot's feature matrix
                    # row-aligned with the values actually computed.
                    Xd = Xd[:shap_values.shape[0]]
                used_path = "shap.Explainer(score_samples)"
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "shap.Explainer fallback failed (%s); using permutation "
                    "importance on score_samples.", exc,
                )
                _checkpoint("model_agnostic_failed", error=str(exc))

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
                _checkpoint(
                    "beeswarm_render_started", rows=int(Xd.shape[0]),
                    features=int(Xd.shape[1]),
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
                _checkpoint("beeswarm_render_done")
            except Exception as exc:  # noqa: BLE001 - beeswarm may be unavailable
                log.warning(
                    "SHAP beeswarm plot failed (%s); writing mean|SHAP| bar "
                    "chart instead.", exc,
                )
                _checkpoint("beeswarm_render_failed", error=str(exc))
                _save_bar_plot(
                    names, mean_abs, out_path,
                    title="Isolation Forest -- mean|SHAP| feature importance",
                    xlabel="mean |SHAP value|",
                )
            log.info(
                "SHAP summary (%s) saved to %s; top feature=%s.",
                used_path, out_path, next(iter(importance)) if importance else "n/a",
            )
            _checkpoint("completed", used_path=used_path)
            return importance

        # -- path 3: permutation importance fallback ------------------------ #
        used_path = "permutation_importance(score_samples)"
        _checkpoint("permutation_importance_started")
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
        _checkpoint("completed", used_path=used_path)
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
        _checkpoint("path_length_started", n_rows=int(Xd.shape[0]))
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
        _checkpoint("path_length_completed", n_rows=int(scores.size))
        return summary


def explain_rows_iforest(
    detector,
    X,
    feature_names=None,
    top_k: int = 5,
    time_budget_s: int = _TREE_EXPLAINER_TIME_BUDGET_S,
    hard_kill_s: int = _TREE_EXPLAINER_HARD_KILL_S,
):
    """Per-row explanation: the ``top_k`` features (by ``|SHAP value|``)
    driving *each individual row's* anomaly score.

    This is the complement of :func:`shap_summary_iforest`: that function
    explains a representative *subsample* to build one aggregate,
    population-level ranking ("which features matter most overall"). This
    function explains the exact rows it is given -- meant for a small,
    specific set such as an OOT alert queue -- and returns one answer *per
    row* ("why is *this* individual flagged"), suitable for a spreadsheet
    column a reviewer reads next to each case.

    Reuses `shap.TreeExplainer` via the same isolated-child-process,
    hard-kill-guarded path as `shap_summary_iforest` (`_run_with_hard_kill`,
    `_tree_explainer_child`) -- the same tree-depth-driven slowness that
    motivated that safety net applies here too, just against a (usually much
    smaller) fixed row set instead of a subsample size.

    Args:
        detector: A fitted :class:`IsolationForestDetector`.
        X: Preprocessed feature matrix for exactly the rows to explain (dense
            ndarray or scipy sparse) -- not subsampled internally, unlike
            `shap_summary_iforest`. Keep this to the rows that actually need
            an answer (e.g. the alert queue), not the whole panel.
        feature_names: Optional names aligned to ``X``'s columns.
        top_k: How many feature names to report per row.
        time_budget_s / hard_kill_s: Same soft-budget/hard-kill-ceiling
            meaning as `shap_summary_iforest`'s path 1. If the ceiling fires
            or SHAP is unavailable, every row gets ``None`` rather than
            raising -- this must never block the business deliverable it
            feeds.

    Returns:
        A list of length ``len(X)``, same row order: each entry is a
        comma-joined string of the ``top_k`` feature names for that row
        (descending by ``|SHAP value|``), or ``None`` for a row that could
        not be explained (SHAP unavailable, or cut off by the hard-kill
        ceiling on an unusually large/slow request).
    """
    log = setup_logging()
    n_rows_requested = int(np.asarray(X).shape[0]) if hasattr(X, "shape") else len(X)
    if n_rows_requested == 0:
        return []

    _checkpoint("explain_rows_started", n_rows=n_rows_requested)

    model = getattr(detector, "model_", None)
    if model is None:
        log.warning("explain_rows_iforest: detector has no fitted model_; skipping.")
        _checkpoint("explain_rows_completed", n_explained=0, n_requested=n_rows_requested)
        return [None] * n_rows_requested

    Xd = _densify(X)
    names = _resolve_feature_names(feature_names, Xd.shape[1])
    calib_rows = min(_TREE_EXPLAINER_CALIBRATION_ROWS, Xd.shape[0])

    try:
        import shap  # noqa: F401 - import check before spending a subprocess

        # Same checkpoint names/shape as `shap_summary_iforest`'s path 1 (this
        # reuses that exact isolated, hard-kill-guarded child), so a stall here
        # is diagnosed the same way: whichever `explain_rows_*` name is last in
        # the log is the sub-step that was in flight when progress stopped.
        _checkpoint("explain_rows_calibration_started", calib_rows=calib_rows)

        def _on_progress(msg, _n_total=Xd.shape[0]):
            _checkpoint(
                "explain_rows_calibrated", calib_rows=msg["calib_n"],
                calib_seconds=round(msg["calib_dt"], 2),
                seconds_per_row=round(msg["per_row"], 2),
                planned_rows=int(msg["n_explain"]),
                bounded=bool(msg["n_explain"] < _n_total),
            )
            if msg["n_explain"] > msg["calib_n"]:
                _checkpoint(
                    "explain_rows_explain_started", planned_rows=int(msg["n_explain"]),
                )

        result = _run_with_hard_kill(
            _tree_explainer_child, (model, Xd, calib_rows, time_budget_s), hard_kill_s,
            on_progress=_on_progress,
        )
        if result.get("stage") == "hard_killed":
            _checkpoint("explain_rows_hard_killed", hard_timeout_s=hard_kill_s)
        elif result.get("stage") != "done":
            _checkpoint("explain_rows_failed", error=str(result.get("error", result.get("stage"))))
        if result.get("stage") != "done":
            log.warning(
                "explain_rows_iforest: TreeExplainer unavailable or too slow "
                "for %d row(s) (%s); no per-row explanations.",
                n_rows_requested, result.get("error", result.get("stage")),
            )
            _checkpoint("explain_rows_completed", n_explained=0, n_requested=n_rows_requested)
            return [None] * n_rows_requested

        _checkpoint("explain_rows_done", rows_explained=int(result["n_explain"]))
        sv = result["sv"]
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv, dtype=np.float64)
        if sv.ndim == 3:
            sv = sv[..., 0]
    except Exception as exc:  # noqa: BLE001 - never block the deliverable this feeds
        log.warning("explain_rows_iforest failed (%s); no per-row explanations.", exc)
        _checkpoint("explain_rows_failed", error=str(exc))
        _checkpoint("explain_rows_completed", n_explained=0, n_requested=n_rows_requested)
        return [None] * n_rows_requested

    n_explained = sv.shape[0]
    if n_explained < n_rows_requested:
        log.warning(
            "explain_rows_iforest: only %d of %d requested row(s) were explained "
            "within the %ds soft budget; the rest get no explanation.",
            n_explained, n_rows_requested, time_budget_s,
        )
    k = max(1, min(top_k, len(names)))
    explanations = []
    for i in range(n_explained):
        order = np.argsort(-np.abs(sv[i]))[:k]
        explanations.append(", ".join(names[j] for j in order))
    explanations.extend([None] * (n_rows_requested - n_explained))
    _checkpoint("explain_rows_completed", n_explained=n_explained, n_requested=n_rows_requested)
    return explanations
