"""End-to-end orchestrator for the banking-panel anomaly-detection framework.

Wires the project's existing public APIs into a single pipeline:

    data -> preprocessing (+ statistical justification) -> OOT split ->
    Isolation Forest + VAE (tune/fit) -> evaluation -> OOT top-decile Excel ->
    interpretability -> HTML/MD report + technical documentation.

Every phase is wrapped in ``log_phase`` and all runtime output lands in
``logs/execution.log`` (with console echo). Optional steps (SHAP, UMAP) are
guarded so the pipeline always emits the OOT Excel and the report even if an
optional artifact fails.

Run ``python main.py --help`` for the CLI. ``python main.py`` performs a quick
CPU run; ``--full`` triggers the spec-scale run deliberately.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.utils import console_ui, observability, paths
from src.utils.logging_config import log_phase, setup_logging

# --------------------------------------------------------------------------- #
# Project-relative artifact locations (never hardcode absolute paths).        #
#                                                                             #
# Everything the pipeline *writes* lives under a single `artifacts/` tree, so #
# the repository root holds only source (main.py, src/, tests/, docs/) and    #
# the whole generated state can be inspected -- or deleted -- in one place.   #
# `src.utils.paths` is the single source of truth; these names are kept as    #
# module-level aliases so existing references keep working.                   #
# --------------------------------------------------------------------------- #
DATA_PATH = paths.DATA_PATH
TUNING_DIR = paths.TUNING_DIR
MODELS_DIR = paths.MODELS_DIR
REPORTS_DIR = paths.REPORTS_DIR
FIGURES_DIR = paths.FIGURES_DIR
LOGS_DIR = paths.LOGS_DIR

IFOREST_MODEL = paths.IFOREST_MODEL
VAE_MODEL = paths.VAE_MODEL
IFOREST_BEST_PARAMS = paths.IFOREST_BEST_PARAMS
VAE_BEST_PARAMS = paths.VAE_BEST_PARAMS


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Effective configuration for a single pipeline run."""

    n_individuals: int = 2000
    # 15 periods: enough for a 10/2/3 chronological split and for the 6-period
    # contrast horizon to exist.
    n_periods: int = 15
    seed: int = 42
    # Strategy default: unsupervised. Ground-truth labels are always loaded
    # (Phase 3) when a ground-truth file exists -- ground_truth.parquet is
    # still read for diagnostics either way -- but they only feed the tuning
    # objective and the metrics computed as "supervised" (PR-AUC/ROC-AUC
    # against true labels) when this is explicitly turned on. Auto-detecting
    # "supervised" from label availability (the previous behavior) meant a
    # run's strategy silently depended on whether a ground-truth file
    # happened to be present, not on an explicit choice. Pass --supervised
    # to opt in; still requires n_pos > 0 (see Phase 3), so an explicit
    # request against a label-free dataset degrades to unsupervised with a
    # logged warning rather than raising.
    supervised: bool = False
    # Open a local-only live progress view (127.0.0.1, a background HTTP
    # server, no external network exposure) in the default browser at the
    # start of the run. Never published as a claude.ai Artifact or otherwise
    # -- purely local. See src/reporting/flow_visualization.py::start_live_view.
    live_view: bool = True
    # Live terminal dashboard (progress bar, per-phase timing, run stats,
    # log tail) in place of scrolling log lines. Auto-disables when `rich` is
    # missing or stdout is not a TTY (piped/redirected/CI), so it never
    # corrupts a captured log. See src/utils/console_ui.py.
    console_ui: bool = True
    numeric_transform: str = "yeo-johnson"
    categorical_encoding: str = "onehot"
    # Within-entity lag/diff/ratio/own-z + seasonality features. Off by default:
    # this pipeline's own real-data usage computes those features in a separate
    # upstream flow, so generating them here would duplicate/conflict with that.
    # Still available for the synthetic-data workflow via --panel-features.
    panel_features: bool = False
    tune: bool = True
    iforest_trials: int = 15
    vae_trials: int = 10
    vae_epochs: int = 15
    # Chronological split: trailing test months, and the validation months
    # immediately before them (used for tuning AND threshold calibration).
    n_val_periods: int = 2
    n_test_periods: int = 3
    # Headline deliverable: a fixed-size review queue. `top_n` wins; set it to
    # None to fall back to `top_fraction`.
    top_n: Optional[int] = 50
    top_fraction: float = 0.10
    threshold_method: str = "pot"
    threshold_percentile: float = 99.0
    threshold_target_far: float = 1e-3
    # IF -> VAE stacking: append the Isolation Forest's anomaly score to the
    # matrix the VAE trains on. When on, the VAE subsumes the forest's signal
    # and becomes the single deliverable (see `deliverable_models`).
    stack_iforest_into_vae: bool = True
    data_path: str = DATA_PATH
    # P95 checkpoint gate (Phase 6c, between the Isolation Forest fit and the
    # VAE layer): percentile of the in-time score distribution above which a
    # row is exported. See CONTEXT.md "Panel features default OFF..." / P95
    # section for why the threshold is fitted on in-time rows only.
    p95_percentile: float = 95.0
    # Isolation Forest params. `contamination` here is the single source of
    # truth for BOTH paths: the tuned path passes it explicitly to
    # `tune_iforest` and the untuned fallback constructs the detector with it
    # directly -- before this centralization, `tune_iforest` silently used
    # its own internal default (0.10) whenever `--tune` was on, while the
    # fallback path used this dict's 0.02, so the effective contamination
    # changed depending on --tune/--no-tune with no warning. `max_samples`/
    # `max_features`/`bootstrap` apply to the untuned fallback fit only (the
    # tuned path searches its own values for these, see docs/models_isolation_forest.md §2b).
    iforest_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 200, "contamination": 0.02,
            "max_samples": "auto", "max_features": 1.0, "bootstrap": False,
        }
    )
    # Held-out fraction of each Optuna trial's fit set, used by `tune_iforest`
    # for its rank-agreement objective (see docs/models_isolation_forest.md §3).
    iforest_holdout_frac: float = 0.3
    # Trial-level early stopping for both Optuna studies (src/models/_tuning_stop.py,
    # distinct from the VAE's own *per-epoch* stopping in `vae_params` below).
    # `patience=None` disables it; defaults mirror tune_iforest/tune_vae's own
    # function defaults, so leaving these alone changes nothing.
    iforest_tuning_early_stopping: dict = field(
        default_factory=lambda: {"patience": 10, "min_delta": 0.005, "min_trials": 10}
    )
    vae_tuning_early_stopping: dict = field(
        default_factory=lambda: {"patience": 10, "min_delta": 0.005, "min_trials": 10}
    )
    # VAE architecture/training params for the untuned fallback fit (the tuned
    # path searches its own values for most of these -- see
    # docs/models_vae.md §2b). Defaults mirror VAEDetector's own class
    # defaults (src/models/vae.py:315) exactly, so leaving these alone
    # changes nothing; edit here, not in vae.py, to change a pipeline run.
    vae_params: dict = field(
        default_factory=lambda: {
            "latent_dim": 8, "hidden_dim": 64, "n_layers": 2, "dropout": 0.0,
            "beta": 1.0, "lr": 1e-3, "optimizer": "adam", "batch_size": 256,
            "weight_decay": 0.0, "activation": "relu", "kl_anneal_epochs": 10,
            "early_stopping_patience": 10,
        }
    )

    @property
    def deliverable_models(self) -> tuple:
        """Detectors that get a risk-ranked Excel.

        Stacked: only the VAE, since it already carries the forest's score.
        Parallel: both, because they rank genuinely different people.
        """
        return ("vae",) if self.stack_iforest_into_vae else ("iforest", "vae")


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _ensure_dirs() -> None:
    for d in (
        os.path.dirname(DATA_PATH),
        TUNING_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ):
        if d:
            os.makedirs(d, exist_ok=True)


def _read_best_params(path: str) -> dict:
    """Read the ``best_params`` block from a tuning YAML (empty dict on failure)."""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("best_params", {}) or {}
    except Exception:
        return {}


def _add_fig(figs: list, title: str, path: Optional[str]) -> None:
    """Append a figure entry if its file actually exists on disk."""
    if path and os.path.isfile(path):
        figs.append({"title": title, "path": path})


def _model_metrics(supervised, labels, scores, oot_mask, X, label_types=None) -> dict:
    """Assemble supervised (OOT + overall) and unsupervised metrics into one dict.

    When ``label_types`` is given, also attaches ``by_type`` / ``oot_by_type``
    recall breakdowns: an aggregate PR-AUC cannot say *which* anomaly geometry
    a detector is blind to, and the four injected types call for different
    remedies.
    """
    from src.evaluation import (
        metrics_by_anomaly_type,
        supervised_metrics,
        unsupervised_metrics,
    )

    metrics: dict = {}
    if supervised:
        oot = supervised_metrics(labels[oot_mask], scores[oot_mask])
        overall = supervised_metrics(labels, scores)
        for k in (
            "roc_auc", "pr_auc", "best_f1", "mcc",
            "precision_at_10pct", "recall_at_10pct", "lift_at_10pct",
        ):
            metrics[f"oot_{k}"] = oot.get(k)
        metrics["overall_pr_auc"] = overall.get("pr_auc")
        metrics["overall_roc_auc"] = overall.get("roc_auc")
        if label_types is not None:
            metrics["by_type"] = metrics_by_anomaly_type(labels, label_types, scores)
            metrics["oot_by_type"] = metrics_by_anomaly_type(
                labels[oot_mask], np.asarray(label_types)[oot_mask], scores[oot_mask]
            )
    unsup = unsupervised_metrics(X, scores)
    for k in ("silhouette", "calinski_harabasz", "rank_stability", "n_flagged"):
        metrics[f"unsup_{k}"] = unsup.get(k)
    return metrics


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig) -> dict:
    """Run the full anomaly-detection pipeline end to end.

    Returns a dict of the key output artifact paths (OOT Excels, report files,
    model checkpoints, best-param YAMLs, log file).
    """
    # -- Phase 1: environment / logging ------------------------------------- #
    logger = setup_logging()
    _ensure_dirs()
    logger.info("=" * 72)
    logger.info("Anomaly-detection pipeline starting")
    logger.info("Effective config: %s", asdict(config))
    ctx = observability.start_run(config=asdict(config), seed=config.seed)
    logger.info("Run ID: %s (config_hash=%s) -- structured events: %s",
                ctx.run_id, ctx.config_hash, ctx.events_path)
    observability.check(
        name="reproducibility.seed_recorded", category="reproducibility",
        definition="A fixed random seed is set for this run.",
        expected="seed is not None", severity="info",
        passed=config.seed is not None, observed=config.seed,
        evidence="PipelineConfig.seed",
    )

    # Live local view: a background HTTP server on 127.0.0.1 only (never
    # published anywhere -- no claude.ai Artifact, no network exposure
    # beyond this machine), opened in the default browser immediately so
    # progress is visible from the very first phase, not only after the run
    # ends. Best-effort: a browser/server failure must never break the run
    # it's only supposed to be showing.
    live_url = ""
    if config.live_view:
        try:
            from src.reporting import start_live_view

            live_url = start_live_view(events_path=ctx.events_path, run_id=ctx.run_id)
            logger.info("Live flow view: %s", live_url)
            import webbrowser

            webbrowser.open(live_url)
        except Exception as exc:
            logger.warning("Live flow view failed to start (%s); continuing without it.", exc)

    # Terminal dashboard. Started after the live view so it can show that URL,
    # and torn down in `main`'s finally-block so the terminal is always
    # restored -- including on Ctrl-C or an exception mid-phase.
    console_ui.start_dashboard(
        enabled=config.console_ui, run_id=ctx.run_id, live_url=live_url
    )
    console_ui.set_stat("Semilla", config.seed)
    console_ui.set_stat("Modo", "supervisado" if config.supervised else "no supervisado")

    figures: list = []
    oot_excels: dict = {}
    model_specs: dict = {}
    chart_data: dict = {"models": {}}
    # Numeric payloads behind the charts that used to be embedded PNGs. The
    # report renders 100% Plotly, so it needs the arrays rather than an image;
    # the PNGs are still written to artifacts/ as run evidence.
    chart_static: dict = {}
    # Posterior-collapse diagnostics, per model. Keyed by model so the report
    # renders them beside that model's other numbers.
    model_latent_diagnostics: dict = {}
    generated_at = datetime.now().isoformat(timespec="seconds")

    # Artifact destinations for THIS run. Locals, not the module constants:
    # a synthetic run redirects them into `_dev/` (see Phase 2) so it can never
    # overwrite the official model or tuned parameters. Bound here, before any
    # read, because assigning them later in the function would make every
    # earlier read an UnboundLocalError.
    IFOREST_MODEL = paths.IFOREST_MODEL
    VAE_MODEL = paths.VAE_MODEL
    IFOREST_BEST_PARAMS = paths.IFOREST_BEST_PARAMS
    VAE_BEST_PARAMS = paths.VAE_BEST_PARAMS
    IFOREST_STUDY_DB = paths.IFOREST_STUDY_DB
    VAE_STUDY_DB = paths.VAE_STUDY_DB

    # -- Phase 2: data ------------------------------------------------------ #
    with log_phase("Phase 2: data load/generate"):
        from src.data import load_or_generate_panel

        df, schema = load_or_generate_panel(
            data_path=config.data_path,
            n_individuals=config.n_individuals,
            n_periods=config.n_periods,
            seed=config.seed,
        )
        entity_col = schema.entity_col or "entity_id"
        time_col = schema.time_col or "period"
        n_entities = int(df[entity_col].nunique()) if entity_col in df.columns else -1
        n_periods = int(df[time_col].nunique()) if time_col in df.columns else -1
        logger.info(
            "Panel: %d rows x %d cols; %d entities x %d periods; ground_truth=%s",
            df.shape[0], df.shape[1], n_entities, n_periods,
            schema.ground_truth_path,
        )
        console_ui.set_stat("Filas", f"{df.shape[0]:,}")
        console_ui.set_stat("Entidades", f"{n_entities:,}")
        console_ui.set_stat("Períodos", f"{n_periods:,}")

        # -- official vs development run ------------------------------------ #
        # A run on generated data is a rehearsal, not the real thing. Its tuned
        # parameters describe invented structure, so letting it write
        # `best_params_*.yaml` (or the model checkpoint, or the Optuna study)
        # would put development output exactly where the deployed artifacts
        # live -- and the next reader has no way to tell which is which.
        # Everything writable is redirected under `_dev/` instead; the run
        # still works end to end, it just cannot contaminate the official tree.
        official_run = not schema.is_synthetic
        if not official_run:
            IFOREST_MODEL = paths.dev_variant(IFOREST_MODEL)
            VAE_MODEL = paths.dev_variant(VAE_MODEL)
            IFOREST_BEST_PARAMS = paths.dev_variant(IFOREST_BEST_PARAMS)
            VAE_BEST_PARAMS = paths.dev_variant(VAE_BEST_PARAMS)
            IFOREST_STUDY_DB = paths.dev_variant(IFOREST_STUDY_DB)
            VAE_STUDY_DB = paths.dev_variant(VAE_STUDY_DB)
            for p in (IFOREST_MODEL, IFOREST_BEST_PARAMS, IFOREST_STUDY_DB):
                os.makedirs(os.path.dirname(p), exist_ok=True)
            logger.warning(
                "CORRIDA DE DESARROLLO (datos sintéticos): los parámetros "
                "ajustados, los modelos y los estudios de Optuna se escriben "
                "en '%s/', no en los artefactos oficiales.", paths.DEV_SEGMENT,
            )
        else:
            logger.info("Corrida OFICIAL (datos reales): los artefactos "
                        "ajustados se persisten en su ubicación definitiva.")
        console_ui.set_stat("Tipo de corrida",
                            "oficial" if official_run else "desarrollo (sintética)")
        observability.check(
            name="run.official", category="reproducibility",
            definition="Whether this run used real data, and therefore whether "
                       "its tuned parameters and model checkpoints are the "
                       "official ones.",
            expected="informational; synthetic runs write under _dev/",
            severity="info", passed=True,
            observed={"official": official_run,
                      "is_synthetic": schema.is_synthetic,
                      "data_path": config.data_path},
            evidence=config.data_path,
        )

        ctx.set_dataset(observability.DatasetFingerprint.from_path(config.data_path, df=df))
        observability.check(
            name="data.non_empty", category="data",
            definition="The loaded panel has at least one row and one column.",
            expected="rows > 0 and cols > 0", severity="critical",
            passed=bool(df.shape[0] > 0 and df.shape[1] > 0),
            observed={"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            failure_action="Stop before preprocessing -- every downstream phase assumes a non-empty panel.",
            evidence=config.data_path,
        )

    # -- Phase 3b: assumption validation ------------------------------------ #
    # Blocking: checks `schema.entity_col`/`schema.time_col` as *inferred*
    # (not the "entity_id"/"period" display fallback above, which would mask
    # a real inference failure), duplicate panel keys, unparseable periods,
    # infinite values. Non-blocking: null rates, constant features, full-row
    # duplicates -- logged as warnings, execution continues. Must run before
    # Phase 4 (preprocessing is undefined over a panel with duplicate keys or
    # unparseable periods) and after Phase 2 (needs the loaded frame).
    with log_phase("Phase 3b: assumption validation"):
        from src.utils import assumptions

        assumptions.validate_panel(df, schema.entity_col, schema.time_col)

    # -- Phase 3a: chronological split (computed on the RAW panel) ---------- #
    # The split must be known *before* preprocessing so the estimated part of
    # the pipeline can be fitted on train rows only. It is derivable from the
    # raw frame because `keys` is just df[[entity_col, time_col]] in row order,
    # so the masks are identical either way (re-verified in Phase 5).
    #
    # Three blocks, three distinct jobs: train fits preprocessing + models,
    # validation selects hyperparameters and calibrates the alert threshold,
    # test is read exactly once at the end.
    with log_phase("Phase 3a: chronological split"):
        from src.evaluation import chronological_split

        split = chronological_split(
            df, time_col=time_col,
            n_val_periods=config.n_val_periods,
            n_test_periods=config.n_test_periods,
        )
        train_mask, val_mask, test_mask = split.train_mask, split.val_mask, split.test_mask
        # Preprocessing may only learn from train; tuning may see train+val.
        in_mask = train_mask | val_mask     # everything the models may touch
        oot_mask = test_mask                # the reported, never-tuned-on block
        oot_period_str = ", ".join(str(v)[:10] for v in np.atleast_1d(split.test_periods))
        person_overlap = assumptions.measure_person_overlap(
            df, entity_col, train_mask, val_mask, test_mask
        )
        logger.info(
            "Person overlap (diagnostic, not pass/fail -- see CONTEXT.md Finding #4): "
            "train/test=%.1f%%, val/test=%.1f%%, test entities never in train/val=%d",
            100.0 * (person_overlap["train_test_person_overlap"] or 0),
            100.0 * (person_overlap["val_test_person_overlap"] or 0),
            person_overlap["entities_in_test_never_seen_in_train_or_val"],
        )

    # -- Phase 4: preprocessing (diagnostics + transform) ------------------- #
    # (Preprocessing runs before ground-truth labels because the labels are
    # joined onto the keys that preprocessing produces.)
    with log_phase("Phase 4: preprocessing + statistical justification"):
        from src.preprocessing import (
            compute_transform_diagnostics,
            fit_transform_panel,
            plot_transform_diagnostics,
            recommend_transform,
        )

        try:
            # Diagnostics are computed on the in-time rows only, for the same
            # reason the transforms are fitted there: a recommendation informed
            # by the OOT month is a recommendation informed by the future.
            diagnostics = compute_transform_diagnostics(
                df[train_mask], schema, random_state=config.seed
            )
            rec = recommend_transform(diagnostics)
            logger.info(
                "Transform recommendation (per feature):\n%s",
                rec.to_string(index=False) if len(rec) else "(none)",
            )
            diag_paths = plot_transform_diagnostics(df, schema, random_state=config.seed)
            for p in diag_paths:
                _add_fig(figures, f"Preprocessing diagnostics: {os.path.basename(p)}", p)
            # NOTE: deliberately not collected into chart_static for the HTML
            # report -- one interactive chart per raw column does not scale to
            # a real feature mart (50+ columns), and would saturate the page.
            # The static PNGs above are still written and are linked (not
            # embedded) from model_documentation.md.
        except Exception as exc:  # diagnostics are evidence, not load-bearing
            logger.warning("Transform diagnostics/plots failed (%s); continuing.", exc)

        # `fit_mask` fits the estimated stage (imputers, scalers, encoders, the
        # "auto" per-column choice) on in-time rows only, while the causal panel
        # features are still computed over the whole panel -- see the
        # `fit_transform_panel` docstring for why the naive alternative would
        # zero out every lag on exactly the OOT rows.
        X, keys, feature_names = fit_transform_panel(
            df,
            schema,
            fit_mask=train_mask,
            numeric_transform=config.numeric_transform,
            categorical_encoding=config.categorical_encoding,
            add_panel_features=config.panel_features,
            random_state=config.seed,
        )
        n_features = len(feature_names)
        console_ui.set_stat("Features", f"{n_features:,}")
        logger.info(
            "Preprocessing: %d input cols -> %d features (%d rows)",
            df.shape[1], n_features, X.shape[0],
        )

        # Per-model feature views. Categorical columns are detected purely by
        # dtype upstream (object/category -> the ColumnTransformer's `cat__`
        # branch), so a new string column in the source data is handled with
        # no code change here. The Isolation Forest is fitted on the numeric
        # view only; the VAE keeps the full matrix -- see
        # `split_matrix_for_model` for the reasoning behind the asymmetry.
        from src.preprocessing import categorical_feature_mask, split_matrix_for_model

        X_if, names_if = split_matrix_for_model(X, feature_names, "iforest")
        n_cat = int(categorical_feature_mask(feature_names).sum())
        logger.info(
            "Feature routing by dtype: %d categorical-derived column(s) -> "
            "Isolation Forest sees %d numeric feature(s), VAE sees all %d.",
            n_cat, len(names_if), n_features,
        )
        observability.check(
            name="data.categorical_routing", category="data",
            definition="Categorical-derived features are withheld from the Isolation "
                       "Forest and kept for the VAE, identified by source dtype.",
            expected="iforest feature count == total - categorical count",
            severity="info", passed=(len(names_if) == n_features - n_cat),
            observed={"total_features": n_features, "categorical_features": n_cat,
                      "iforest_features": len(names_if), "vae_features": n_features},
            evidence="src.preprocessing.split_matrix_for_model",
        )

    # -- Phase 3: ground-truth labels --------------------------------------- #
    with log_phase("Phase 3: ground-truth labels"):
        from src.evaluation import load_ground_truth_labels, load_ground_truth_types

        labels = load_ground_truth_labels(schema, keys)
        label_types = load_ground_truth_types(schema, keys)
        n_pos = int(labels.sum())
        if config.supervised and n_pos == 0:
            logger.warning(
                "config.supervised=True but ground truth has 0 positive labels; "
                "falling back to unsupervised evaluation -- there is nothing to supervise with."
            )
        supervised = bool(config.supervised) and n_pos > 0
        anomaly_rate = float(labels.mean()) if len(labels) else float("nan")
        logger.info(
            "Labels: %d positives / %d rows (rate=%.4f%%) -> %s evaluation (strategy=%s)",
            n_pos, len(labels), 100.0 * anomaly_rate,
            "SUPERVISED" if supervised else "UNSUPERVISED",
            "explicitly requested via --supervised" if config.supervised else "default",
        )
        console_ui.set_stat("Modo", "supervisado" if supervised else "no supervisado")
        if n_pos:
            console_ui.set_stat("Anomalías (verdad base)",
                                f"{n_pos:,} · {anomaly_rate:.2%}")

    # -- Phase 5: split verification + model-facing slices ------------------ #
    with log_phase("Phase 5: split verification"):
        # Recompute from `keys` and assert it matches the mask preprocessing was
        # actually fitted with. If preprocessing ever stops preserving row
        # order, this fails loudly instead of silently misaligning the split.
        keys_split = chronological_split(
            keys, time_col=time_col,
            n_val_periods=config.n_val_periods,
            n_test_periods=config.n_test_periods,
        )
        if not np.array_equal(keys_split.train_mask, train_mask):
            n_disagree = int(np.sum(keys_split.train_mask != train_mask))
            raise assumptions.LeakageAssumptionError(
                f"Split recomputed from `keys` disagrees with the mask preprocessing "
                f"was actually fitted with on {n_disagree} row(s) -- preprocessing did "
                f"not preserve row order, so `fit_mask` may have fitted estimators on "
                f"rows the split intends as validation/test.",
                check="leakage.split_row_order_preserved",
                observed={"n_disagreeing_rows": n_disagree, "total_rows": len(train_mask)},
            )
        # Models see train+val; `valid_local` marks the validation rows *within*
        # that slice, which is what tune_* uses as its held-out block.
        # `X_in` is the Isolation Forest's slice: numeric-only view, in-time rows.
        # The VAE's slice is taken from the full matrix in Phase 7.
        X_in = X_if[in_mask]
        labels_in = labels[in_mask]
        valid_local = val_mask[in_mask]
        entities_in = (
            keys[entity_col].to_numpy()[in_mask] if entity_col in keys.columns else None
        )
        logger.info(
            "Model input: %d rows (train %d + val %d) | test/OOT rows=%d | test period(s)=%s",
            int(in_mask.sum()), int(train_mask.sum()), int(val_mask.sum()),
            int(oot_mask.sum()), oot_period_str,
        )

    # -- Phase 6: Isolation Forest ------------------------------------------ #
    with log_phase("Phase 6: Isolation Forest"):
        from src.models import (
            IsolationForestDetector,
            plot_score_distribution,
            tune_iforest,
        )

        # Blocking gate: validates the *fallback* config (used verbatim when
        # --no-tune, and as the seed for the tuning search space either way);
        # a bad contamination value here would otherwise surface only much
        # later as a silently-wrong threshold.
        assumptions.validate_iforest_config(
            contamination=config.iforest_params.get("contamination", 0.02),
            n_estimators=config.iforest_params.get("n_estimators", 200),
            max_samples=config.iforest_params.get("max_samples", "auto"),
        )
        assumptions.validate_matrix_for_fit(X_in, "iforest")

        if_detector = None
        if config.tune:
            try:
                tune_iforest(
                    X_in,
                    n_trials=config.iforest_trials,
                    y=(labels_in if supervised else None),
                    random_state=config.seed,
                    # Passed explicitly, not left to the module defaults: on a
                    # synthetic run these point under `_dev/`.
                    best_params_path=IFOREST_BEST_PARAMS,
                    model_out=IFOREST_MODEL,
                    storage="sqlite:///" + IFOREST_STUDY_DB.replace("\\", "/"),
                    # Temporal holdout: trials are scored on the validation
                    # MONTHS, which is what deployment looks like. `groups` is
                    # kept as the fallback for callers with no time split.
                    valid_mask=valid_local,
                    groups=entities_in,
                    feature_names=names_if,
                    # Centralized here (PipelineConfig.iforest_params) so the
                    # tuned and untuned paths never silently disagree on the
                    # operating-point contamination -- see the field's
                    # docstring above for the inconsistency this closes.
                    contamination=config.iforest_params.get("contamination", 0.02),
                    holdout_frac=config.iforest_holdout_frac,
                    early_stopping_patience=config.iforest_tuning_early_stopping["patience"],
                    early_stopping_min_delta=config.iforest_tuning_early_stopping["min_delta"],
                    early_stopping_min_trials=config.iforest_tuning_early_stopping["min_trials"],
                )
                if_detector = IsolationForestDetector.load(IFOREST_MODEL)
            except Exception as exc:
                logger.warning(
                    "iForest tuning/load failed (%s); fitting default detector.", exc
                )
        if if_detector is None:
            if_detector = IsolationForestDetector(
                random_state=config.seed, **config.iforest_params
            )
            if_detector.fit(X_in)

        if_scores = if_detector.score_samples(X_if)
        if_best_params = _read_best_params(IFOREST_BEST_PARAMS) or dict(config.iforest_params)
        _add_fig(
            figures,
            "Isolation Forest score distribution",
            plot_score_distribution(
                if_scores, y=(labels if supervised else None), filename="iforest_scores.png"
            ),
        )

    # -- Phase 6c: IF P95 checkpoint export (gate) --------------------------- #
    # Not wrapped in a lenient try/except on purpose: `export_p95_checkpoint`
    # raises `ArtifactGenerationError` (unblocked, propagates) on any
    # validation failure, and by not catching it here the VAE genuinely does
    # not start when this gate fails -- the non-negotiable constraint this
    # phase exists to enforce.
    with log_phase("Phase 6c: IF P95 checkpoint export"):
        from src.evaluation import export_p95_checkpoint

        p95_path, p95_table, p95_threshold = export_p95_checkpoint(
            df, if_scores,
            in_mask=in_mask, schema=schema,
            split_masks={"train": train_mask, "val": val_mask, "test": test_mask},
            percentile=config.p95_percentile, model_name="iforest",
        )
        logger.info(
            "IF P95 checkpoint: %d/%d rows >= threshold %.6f -> %s",
            len(p95_table), len(df), p95_threshold, p95_path,
        )

    # -- Phase 6b: IF -> VAE stacking --------------------------------------- #
    # The forest's score becomes an extra column of the matrix the VAE trains
    # on, so the VAE models the normal manifold *including* how isolated the
    # forest finds each row.
    X_vae, vae_feature_names = X, feature_names
    stack_info = None
    if config.stack_iforest_into_vae:
        with log_phase("Phase 6b: IF -> VAE stacking"):
            from src.models import build_stacked_matrix, score_shift_report

            # The stacked feature comes from a forest fitted on TRAIN ONLY.
            # `if_detector` above was refit on train+val (correct for its own
            # ranking), but a column that is in-sample over train+val would let
            # the VAE's threshold — calibrated on val — see a distribution the
            # forest had already memorised.
            stack_detector = IsolationForestDetector(
                random_state=config.seed, n_jobs=-1,
                **{k: v for k, v in (if_best_params or {}).items()
                   if k in {"n_estimators", "max_samples", "max_features",
                            "contamination", "bootstrap"}}
            )
            # Same numeric-only view the main forest uses -- a stacked score
            # produced from a different feature set than the forest being
            # reported would not be the same quantity.
            stack_detector.fit(X_if[train_mask])
            stack_scores = stack_detector.score_samples(X_if)

            stack_info = score_shift_report(
                stack_scores, train_mask,
                {"validation": val_mask, "test": test_mask},
            )
            stacked = build_stacked_matrix(
                X, stack_scores, fit_mask=train_mask, feature_names=feature_names,
            )
            X_vae, vae_feature_names = stacked.X, stacked.feature_names
            logger.info(
                "VAE input: %d -> %d features (last column = %r)",
                stacked.n_original, X_vae.shape[1], stacked.score_name,
            )

    # -- Phase 7: VAE ------------------------------------------------------- #
    with log_phase("Phase 7: VAE"):
        from src.models import (
            VAEDetector,
            plot_latent_space,
            plot_reconstruction_error,
            tune_vae,
        )

        X_vae_in = X_vae[in_mask]
        assumptions.validate_matrix_for_fit(X_vae_in, "vae")
        vae_detector = None
        if config.tune:
            try:
                tune_vae(
                    X_vae_in,
                    n_trials=config.vae_trials,
                    y=(labels_in if supervised else None),
                    max_epochs=config.vae_epochs,
                    random_state=config.seed,
                    # Explicit, so a synthetic run writes under `_dev/`.
                    best_params_path=VAE_BEST_PARAMS,
                    model_out=VAE_MODEL,
                    storage="sqlite:///" + VAE_STUDY_DB.replace("\\", "/"),
                    # Same temporal holdout as the iForest: trials, early
                    # stopping and best-epoch selection all judged on the
                    # validation months.
                    valid_mask=valid_local,
                    early_stopping_patience=config.vae_tuning_early_stopping["patience"],
                    early_stopping_min_delta=config.vae_tuning_early_stopping["min_delta"],
                    early_stopping_min_trials=config.vae_tuning_early_stopping["min_trials"],
                )
                vae_detector = VAEDetector.load(VAE_MODEL)
            except Exception as exc:
                logger.warning(
                    "VAE tuning/load failed (%s); fitting default detector.", exc
                )
        if vae_detector is None:
            vae_detector = VAEDetector(
                random_state=config.seed, epochs=config.vae_epochs, **config.vae_params
            )
            # `valid_mask=valid_local` is NOT optional here. Without it `fit`
            # falls back to a shuffled 10% split, which in a panel draws its
            # validation rows from every period -- including ones later than
            # the rows it trains on. Early stopping and best-epoch selection
            # would then be judged on the future. The tuned path above already
            # passes it; omitting it here made the two paths disagree on
            # something load-bearing, and left `--no-tune` (a documented mode)
            # silently leaking.
            vae_detector.fit(X_vae_in, valid_mask=valid_local)

        vae_scores = vae_detector.score_samples(X_vae)
        vae_best_params = _read_best_params(VAE_BEST_PARAMS)
        _add_fig(
            figures,
            "VAE reconstruction-error distribution",
            plot_reconstruction_error(
                vae_scores, y=(labels if supervised else None), filename="vae_recon.png"
            ),
        )
        try:
            _add_fig(
                figures,
                "VAE latent space",
                plot_latent_space(
                    vae_detector, X_vae, y=(labels if supervised else None)
                ),
            )
        except Exception as exc:
            logger.warning("plot_latent_space failed (%s); continuing.", exc)

        # -- Posterior-collapse gate ---------------------------------------- #
        # The one VAE failure this pipeline could not previously see. The
        # anomaly score IS the reconstruction error, so a decoder that has
        # learned to ignore the latent code still emits finite scores, still
        # ranks rows, and still writes a populated best_params.yaml -- the run
        # looks healthy end to end while the score has stopped meaning
        # anything. Measured here (right after the fit, on the matrix the model
        # was actually fitted on) rather than in Phase 10, so a collapsed model
        # is flagged before its scores are used for the Excel deliverable.
        try:
            from src.models import collapse_verdict

            latent_diag = vae_detector.latent_diagnostics(X_vae)
            verdict = collapse_verdict(latent_diag)
            model_latent_diagnostics["vae"] = {**latent_diag, **verdict}
            logger.info(
                "VAE latent health: %d/%d active units (delta=%.3g), mean KL=%.4f -- %s",
                latent_diag["active_units"], latent_diag["latent_dim"],
                latent_diag["delta"], latent_diag["mean_kl"], verdict["reason"],
            )
            console_ui.set_stat(
                "vae dims. latentes activas",
                f"{latent_diag['active_units']}/{latent_diag['latent_dim']}",
            )
            observability.check(
                name="vae.posterior_collapse", category="training",
                definition="The VAE's latent space has enough active dimensions "
                           "(A_j = Var_x(E_q[z_j|x]) > delta, Burda et al. 2016) "
                           "for the reconstruction error to remain a meaningful "
                           "anomaly score.",
                expected=f"active_fraction >= 1/3 and not degenerate "
                         f"(delta={latent_diag['delta']})",
                severity="critical" if verdict["severity"] == "critical" else "warning",
                passed=not verdict["collapsed"],
                observed={k: latent_diag[k] for k in
                          ("active_units", "inactive_units", "latent_dim",
                           "active_fraction", "mean_kl", "delta")},
                failure_action="The VAE score is no longer discriminative -- retune "
                               "with a lower beta or a smaller latent_dim before "
                               "trusting its OOT queue.",
                evidence="src/models/vae.py::latent_diagnostics",
            )
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not break the run
            logger.warning("VAE latent diagnostics failed (%s); continuing.", exc)

    # -- Phases 8-10 per model: evaluation, OOT export, interpretability ---- #
    # Each detector carries the matrix it was actually fitted on: under
    # stacking the VAE lives in a wider space than the forest, and its
    # unsupervised metrics / embeddings / interpretability must use that one.
    models = {
        "iforest": (if_detector, if_scores, if_best_params, X_if, names_if),
        "vae": (vae_detector, vae_scores, vae_best_params, X_vae, vae_feature_names),
    }

    for name, (detector, scores, best_params, X_model, names_model) in models.items():
        # -- Phase 8: evaluation -------------------------------------------- #
        with log_phase(f"Phase 8: evaluation [{name}]"):
            from src.evaluation import plot_embedding, plot_roc_pr

            metrics = _model_metrics(
                supervised, labels, scores, oot_mask, X_model, label_types=label_types
            )
            model_specs[name] = {"best_params": best_params, "metrics": metrics}
            if name in model_latent_diagnostics:
                model_specs[name]["latent_health"] = model_latent_diagnostics[name]
            # Headline KPIs for the console dashboard, published the moment
            # each model's evaluation finishes rather than only at the end.
            if supervised and metrics.get("oot_roc_auc") is not None:
                console_ui.set_stat(
                    f"{name} ROC-AUC / PR-AUC (OOT)",
                    f"{metrics['oot_roc_auc']:.3f} / {metrics.get('oot_pr_auc', float('nan')):.3f}",
                )
            else:
                sil, ch = metrics.get("unsup_silhouette"), metrics.get("unsup_calinski_harabasz")
                if sil is not None:
                    console_ui.set_stat(f"{name} Silhouette / CH", f"{sil:.3f} / {ch:.1f}")
            # Chart inputs. Labels are attached ONLY when the run is actually
            # supervised: this pipeline's default strategy is unsupervised, and
            # a report that quietly showed ROC/PR against ground truth would be
            # claiming an evaluation the run did not perform. In the default
            # mode the report gets scores + threshold and nothing label-derived.
            chart_data["models"][name] = {
                "oot_scores": [float(v) for v in scores[oot_mask]],
                "oot_labels": ([int(v) for v in labels[oot_mask]] if supervised else None),
                "metrics": metrics,
                "supervised": bool(supervised),
            }

            oot_by_type = metrics.get("oot_by_type") or {}
            for type_name in sorted(k for k in oot_by_type if k != "__overall__"):
                block = oot_by_type[type_name]
                logger.info(
                    "[%s] OOT recall by type -- %-11s n_pos=%3d  recall@1%%=%.3f  "
                    "recall@5%%=%.3f  recall@10%%=%.3f  mean_pctile=%.3f",
                    name, type_name, int(block["n_positive"]),
                    block["recall_at_1pct"], block["recall_at_5pct"],
                    block["recall_at_10pct"], block["mean_score_percentile"],
                )

            try:
                _add_fig(
                    figures,
                    f"{name} PCA embedding",
                    plot_embedding(
                        X_model, scores, method="pca",
                        y=(labels if supervised else None),
                        filename=f"embedding_{name}_pca.png",
                    ),
                )
            except Exception as exc:
                logger.warning("plot_embedding(pca) failed for %s (%s).", name, exc)
            try:  # UMAP is optional; falls back internally but guard the import too
                emb_path, emb_data = plot_embedding(
                    X_model, scores, method="umap",
                    y=(labels if supervised else None),
                    filename=f"embedding_{name}_umap.png",
                    return_data=True,
                )
                _add_fig(figures, f"{name} UMAP embedding", emb_path)
                # UMAP (not PCA) feeds the interactive chart: it preserves local
                # neighbourhood structure, which is what the plot is read for.
                chart_static[f"embedding_{name}"] = emb_data
            except Exception as exc:
                logger.warning("plot_embedding(umap) failed for %s (%s).", name, exc)
            if supervised:
                try:
                    _add_fig(
                        figures,
                        f"{name} ROC/PR (OOT)",
                        plot_roc_pr(
                            labels[oot_mask], scores[oot_mask],
                            filename=f"roc_pr_{name}.png",
                        ),
                    )
                except Exception as exc:
                    logger.warning("plot_roc_pr failed for %s (%s).", name, exc)

        # -- Phase 8b: threshold calibration (on VALIDATION only) ----------- #
        with log_phase(f"Phase 8b: threshold calibration [{name}]"):
            from src.evaluation import calibrate_threshold

            # The cut-off is fitted on the validation months and only ever
            # applied to test. Calibrating on test would be reporting the best
            # threshold in hindsight -- a leak no deployed system can reproduce.
            cal = calibrate_threshold(
                scores[val_mask],
                method=config.threshold_method,
                percentile=config.threshold_percentile,
                target_far=config.threshold_target_far,
            )
            model_specs[name]["threshold"] = cal
            if name in chart_data["models"]:
                chart_data["models"][name]["threshold"] = cal["threshold"]
            test_scores = scores[oot_mask]
            n_alerts = int((test_scores >= cal["threshold"]).sum()) if np.isfinite(cal["threshold"]) else 0
            model_specs[name]["metrics"]["threshold_value"] = cal["threshold"]
            model_specs[name]["metrics"]["threshold_method"] = cal["method"]
            model_specs[name]["metrics"]["test_alert_count"] = float(n_alerts)
            model_specs[name]["metrics"]["test_alert_rate"] = (
                float(n_alerts) / len(test_scores) if len(test_scores) else float("nan")
            )
            logger.info(
                "[%s] threshold=%.6f (%s, calibrated on %d validation rows) -> "
                "%d/%d test rows alert (%.2f%%)",
                name, cal["threshold"], cal["method"], int(val_mask.sum()),
                n_alerts, len(test_scores),
                100.0 * n_alerts / max(len(test_scores), 1),
            )
            console_ui.set_stat(
                f"Alertas OOT [{name}]",
                f"{n_alerts:,} · {100.0 * n_alerts / max(len(test_scores), 1):.2f}%",
            )

        # -- Phase 9: top-N risk-ranked Excel deliverable ------------------- #
        # Under stacking only the VAE ships a queue: its ranking already carries
        # the forest's score as an input feature. The forest still gets metrics
        # and interpretability below, because those are how you tell whether the
        # stacked feature is doing any work.
        if name in config.deliverable_models:
            with log_phase(f"Phase 9: top-N Excel deliverable [{name}]"):
                from src.evaluation import build_scored_frame, export_oot_top_anomalies

                scored_df = build_scored_frame(df, keys, scores, schema)
                out_path, _table = export_oot_top_anomalies(
                    scored_df,
                    schema,
                    top_n=config.top_n,
                    top_fraction=config.top_fraction,
                    model_name=name,
                    n_oot_periods=config.n_test_periods,
                    threshold=cal["threshold"],
                )
                oot_excels[name] = out_path
                logger.info("Top-N deliverable [%s] -> %s", name, out_path)
                _exported_ok = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
                observability.check(
                    name=f"artifact.oot_excel_written[{name}]", category="artifact",
                    definition="The top-N risk-ranked Excel deliverable exists and is non-empty.",
                    expected="file exists and size_bytes > 0", severity="critical",
                    passed=_exported_ok,
                    observed={
                        "path": out_path,
                        "size_bytes": os.path.getsize(out_path) if _exported_ok else 0,
                        "rows_expected": int(len(_table)),
                    },
                    failure_action="The headline deliverable for this model is missing or empty -- treat the run as failed.",
                    evidence=out_path,
                )
        else:
            logger.info(
                "[%s] no Excel deliverable: the VAE trains on the stacked matrix, "
                "so its queue already reflects this detector's score.", name,
            )

    # -- Phase 10: interpretability, AFTER every Excel deliverable ---------- #
    # Deliberately outside the per-model loop above. Interpretability is the
    # slowest stage in the pipeline (SHAP over the forest, UMAP's one-time
    # numba compilation) and produces no deliverable of its own, so running it
    # per model inside the loop delayed the VAE's Excel queue behind the
    # forest's SHAP computation. Hoisted here, every Excel export is on disk
    # and reviewable before any interpretability work starts.
    for name, (detector, scores, best_params, X_model, names_model) in models.items():
        with log_phase(f"Phase 10: interpretability [{name}]"):
            if name == "iforest":
                from src.interpretability import (
                    path_length_analysis,
                    shap_summary_iforest,
                )

                try:
                    imp = shap_summary_iforest(detector, X_model, feature_names=names_model)
                    _add_fig(
                        figures, "iForest SHAP summary",
                        os.path.join(FIGURES_DIR, "iforest_shap_summary.png"),
                    )
                    # Also feeds the report's interactive version of this chart.
                    chart_static["shap_importance"] = imp
                except Exception as exc:
                    logger.warning("shap_summary_iforest failed (%s); continuing.", exc)
                try:
                    summary = path_length_analysis(detector, X_model)
                    _add_fig(figures, "iForest path-length analysis", summary.get("figure_path"))
                    chart_static["path_length"] = {
                        "scores": summary.get("plot_scores"),
                        "path_lengths": summary.get("plot_path_lengths"),
                        "corr": summary.get("score_pathlen_corr"),
                    }
                except Exception as exc:
                    logger.warning("path_length_analysis failed (%s); continuing.", exc)
            else:
                from src.interpretability import (
                    latent_space_plot,
                    reconstruction_error_by_feature,
                )

                try:
                    p, latent_data = latent_space_plot(
                        detector, X_model, y=(labels if supervised else None),
                        return_data=True,
                    )
                    _add_fig(figures, "VAE latent space (interpretability)", p)
                    chart_static["latent_vae"] = latent_data
                except Exception as exc:
                    logger.warning("latent_space_plot failed (%s); continuing.", exc)
                try:
                    recon = reconstruction_error_by_feature(
                        detector, X_model, feature_names=names_model
                    )
                    _add_fig(
                        figures, "VAE per-feature reconstruction error",
                        os.path.join(FIGURES_DIR, "vae_recon_by_feature.png"),
                    )
                    chart_static["recon_by_feature"] = recon
                except Exception as exc:
                    logger.warning("reconstruction_error_by_feature failed (%s); continuing.", exc)

    # -- Phase 11: report --------------------------------------------------- #
    report_paths: dict = {"html": None, "md": None, "model_doc": None}
    with log_phase("Phase 11: report"):
        from src.reporting import build_report

        oot_note = "; ".join(f"{k}: {v}" for k, v in oot_excels.items())
        context = {
            "title": "Reporte de Detección de Anomalías del Panel Bancario",
            "generated_at": generated_at,
            "dataset": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "entities": n_entities,
                "periods": n_periods,
                "features": n_features,
                "oot_period": oot_period_str,
                "anomaly_rate": anomaly_rate,
                "n_anomalies": n_pos,
                "evaluation_mode": "supervised" if supervised else "unsupervised",
                "in_time_rows": int(in_mask.sum()),
                "oot_rows": int(oot_mask.sum()),
                "train_rows": int(train_mask.sum()),
                "val_rows": int(val_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "split": split.describe(),
            },
            "models": model_specs,
            "figures": figures,
            "chart_data": {**chart_data, "anomaly_rate": anomaly_rate,
                            "static": chart_static},
            "oot_excel": oot_excels,
            "preprocessing": {
                "numeric_transform": config.numeric_transform,
                "categorical_encoding": config.categorical_encoding,
                "panel_features": config.panel_features,
                "tuning": config.tune,
                "threshold_method": config.threshold_method,
                "threshold_percentile": config.threshold_percentile,
                "threshold_target_far": config.threshold_target_far,
            },
            "notes": (
                f"Entregables Excel top-{config.top_n if config.top_n is not None else f'{config.top_fraction:.0%}'} "
                f"clasificados por riesgo -> {oot_note}. "
                f"División cronológica: {split.describe()}. "
                f"Umbral calibrado en validación vía {config.threshold_method}. "
                f"Transformación numérica={config.numeric_transform}, "
                f"codificación categórica={config.categorical_encoding}, "
                f"features de panel={'activado' if config.panel_features else 'desactivado'}, "
                f"tuning={'activado' if config.tune else 'desactivado'}."
            ),
        }
        try:
            report_paths = build_report(
                context, out_dir=REPORTS_DIR, basename="anomaly_report",
                formats=("html", "md", "model_doc"),
            )
            observability.check(
                name="artifact.report_written", category="artifact",
                definition="At least one report format (html/md/model_doc) was written.",
                expected="report_paths has >=1 non-None entry", severity="warning",
                passed=any(v for v in report_paths.values()),
                observed=report_paths,
                failure_action="Report is best-effort by design (see module docstring) -- "
                               "OOT Excel deliverables are still authoritative; investigate build_report.",
                evidence=REPORTS_DIR,
            )
        except Exception as exc:
            logger.warning("Report build failed (%s); OOT deliverables still emitted.", exc)
            observability.check(
                name="artifact.report_written", category="artifact",
                definition="At least one report format (html/md/model_doc) was written.",
                expected="report_paths has >=1 non-None entry", severity="warning",
                passed=False, observed={"exception": str(exc)},
                failure_action="Report is best-effort by design -- OOT Excel deliverables are still authoritative.",
                evidence=REPORTS_DIR,
            )

    # -- Phase 12: final summary -------------------------------------------- #
    # Tear the dashboard down *before* printing: the summary is the one thing
    # that must survive in the scrollback, and a Live display owns the bottom
    # of the terminal until it is stopped.
    console_ui.stop_dashboard()

    artifacts = {
        "oot_excels": oot_excels,
        "p95_checkpoint": p95_path,
        "reports": report_paths,
        "figures_dir": os.path.abspath(FIGURES_DIR),
        "log_file": os.path.abspath(os.path.join(LOGS_DIR, "execution.log")),
        "best_params": {"iforest": IFOREST_BEST_PARAMS, "vae": VAE_BEST_PARAMS},
        "models": {"iforest": IFOREST_MODEL, "vae": VAE_MODEL},
    }
    summary_lines = [
        "=" * 72,
        "PIPELINE COMPLETE - artifact locations:",
        f"  IF P95 checkpoint: {p95_path}",
        f"  OOT Excel(s)   : {', '.join(oot_excels.values()) or '(none)'}",
        f"  Report (html)  : {report_paths.get('html')}",
        f"  Report (md)    : {report_paths.get('md')}",
        f"  Model docs     : {report_paths.get('model_doc')}",
        f"  Figures        : {os.path.abspath(FIGURES_DIR)}",
        f"  Best params    : {IFOREST_BEST_PARAMS}, {VAE_BEST_PARAMS}",
        f"  Model ckpts    : {IFOREST_MODEL}, {VAE_MODEL}",
        f"  Log file       : {artifacts['log_file']}",
        "=" * 72,
    ]
    summary = "\n".join(summary_lines)
    logger.info("Final summary:\n%s", summary)
    print(summary)
    health_summary = observability.end_run(ctx, status="success")
    if health_summary["failed_checks"]:
        logger.warning(
            "Run %s completed with %d failed health check(s): %s (see %s)",
            ctx.run_id, len(health_summary["failed_checks"]),
            health_summary["failed_checks"], ctx.events_path,
        )
    artifacts["run_id"] = ctx.run_id
    artifacts["events_log"] = ctx.events_path

    # -- Phase 12b: flow visualization --------------------------------------- #
    # Best-effort by design (like the report above): a visualization bug must
    # never mask a successful pipeline run. Built from `ctx.events_path`,
    # which by this point already contains the `run_ended` event this run
    # just wrote, so the diagram's own summary panel reflects final status.
    try:
        from src.reporting import build_flow_visualization

        flow_path = build_flow_visualization(events_path=ctx.events_path, run_id=ctx.run_id)
        artifacts["flow_visualization"] = flow_path
        if flow_path:
            logger.info("Flow visualization -> %s", flow_path)
    except Exception as exc:
        logger.warning("Flow visualization failed (%s); continuing.", exc)

    return artifacts


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
# Preset knobs affected by --quick / --full (resolved against explicit flags).
_BASE = {
    "n_individuals": 2000, "n_periods": 15,
    "iforest_trials": 15, "vae_trials": 10, "vae_epochs": 15, "tune": True,
}
# --quick still needs >= 3 periods for the chronological split; 8 keeps a
# 4/2/2 train/val/test and lets the h=3 contrast horizon exist (h=6 is dropped
# automatically by PanelFeatureEngineer).
# --quick keeps 12 periods so the 7-month training block can actually support
# the h=6 contrast horizon; with fewer, PanelFeatureEngineer drops it and the
# smoke run stops exercising the feature it is meant to smoke-test.
_QUICK = {
    "n_individuals": 500, "n_periods": 12,
    "iforest_trials": 5, "vae_trials": 5, "vae_epochs": 5,
}
_FULL = {
    "n_individuals": 100_000, "n_periods": 15,
    "iforest_trials": 50, "vae_trials": 30, "vae_epochs": 30,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "End-to-end banking-panel anomaly-detection pipeline: data -> "
            "preprocessing -> Isolation Forest + VAE (tune/fit) -> evaluation "
            "-> OOT top-decile Excel -> interpretability -> HTML/MD report. "
            "Defaults perform a quick CPU run; use --full for the spec-scale run."
        ),
    )
    # Preset-affected args default to None so an explicit value wins over a preset.
    parser.add_argument("--n-individuals", type=int, default=None,
                        help="Number of individuals in the synthetic panel (default 2000).")
    parser.add_argument("--n-periods", type=int, default=None,
                        help="Number of monthly periods (default 8).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for reproducibility (default 42).")
    parser.add_argument("--numeric-transform", default="yeo-johnson",
                        help="Numeric transform for preprocessing (default 'yeo-johnson').")
    parser.add_argument("--categorical-encoding", default="onehot",
                        help="Categorical encoding for preprocessing (default 'onehot').")
    parser.add_argument("--live-view", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Open a local-only live progress view (127.0.0.1, no external "
                             "network exposure) in the default browser at the start of the "
                             "run (default on). --no-live-view disables it, e.g. for a "
                             "headless/CI environment.")
    parser.add_argument("--console-ui", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Live terminal dashboard -- progress bar, per-phase timing, "
                             "run stats and a log tail -- instead of scrolling log lines "
                             "(default on). Auto-disables when stdout is not a terminal "
                             "(piped/redirected/CI) or 'rich' is missing, so it never "
                             "corrupts a captured log; --no-console-ui forces it off.")
    parser.add_argument("--supervised", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Use ground-truth labels for the tuning objective and for "
                             "supervised metrics (PR-AUC/ROC-AUC against true labels) "
                             "(default off -- unsupervised strategy unless explicitly "
                             "requested here). Falls back to unsupervised with a warning "
                             "if the ground truth has zero positive labels.")
    parser.add_argument("--panel-features", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Generate within-entity lag/diff/ratio/own-z + seasonality "
                             "features (default off -- this pipeline's real-data usage "
                             "computes those upstream in a separate flow; pass "
                             "--panel-features to turn them back on, e.g. for the "
                             "synthetic-data workflow).")
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=None,
                        help="Optuna tuning of both detectors (default on; --no-tune to disable).")
    parser.add_argument("--iforest-trials", type=int, default=None,
                        help="Isolation Forest Optuna trials (default 15).")
    parser.add_argument("--vae-trials", type=int, default=None,
                        help="VAE Optuna trials (default 10).")
    parser.add_argument("--vae-epochs", type=int, default=None,
                        help="Max VAE epochs per trial / for the fit (default 15).")
    parser.add_argument("--stack-iforest-into-vae", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Feed the Isolation Forest score to the VAE as an extra "
                             "feature; the VAE then ships the only Excel queue. "
                             "--no-stack-iforest-into-vae runs them in parallel with "
                             "one queue each (default: stacked).")
    parser.add_argument("--contamination", type=float, default=None,
                        help="Isolation Forest operating-point contamination, used by both "
                             "the tuned and untuned paths (default 0.02; must be in (0, 0.5]). "
                             "Does not affect score_samples()/ranking, only predict()/"
                             "decision_function() and the P95 checkpoint export's downstream "
                             "consumers -- see docs/models_isolation_forest.md §2b.")
    parser.add_argument("--p95-percentile", type=float, default=None,
                        help="Percentile of in-time IF scores above which a row is exported "
                             "in the Phase 6c checkpoint gate before the VAE runs (default 95).")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Individuals exported in the risk-ranked Excel (default 50). "
                             "Pass 0 to use --top-fraction instead.")
    parser.add_argument("--n-val-periods", type=int, default=2,
                        help="Validation months for tuning + threshold calibration (default 2).")
    parser.add_argument("--n-test-periods", type=int, default=3,
                        help="Trailing test months, reported but never tuned on (default 3).")
    parser.add_argument("--threshold-method", default="pot", choices=["pot", "percentile"],
                        help="Threshold calibration on validation scores (default 'pot').")
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                        help="Percentile for --threshold-method percentile (default 99).")
    parser.add_argument("--threshold-target-far", type=float, default=1e-3,
                        help="Target false-alarm rate for POT calibration (default 1e-3).")
    parser.add_argument("--top-fraction", type=float, default=0.10,
                        help="OOT top fraction exported to Excel (default 0.10 = top decile).")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny fast smoke run (800 individuals, 6 periods, 5 trials, 5 epochs).")
    parser.add_argument("--full", action="store_true",
                        help="Spec-scale run (100000 individuals, 10 periods, more trials).")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if args.quick and args.full:
        raise SystemExit("--quick and --full are mutually exclusive.")
    preset = _QUICK if args.quick else (_FULL if args.full else {})

    def resolve(name):
        explicit = getattr(args, name)
        if explicit is not None:
            return explicit
        if name in preset:
            return preset[name]
        return _BASE[name]

    config = PipelineConfig(
        n_individuals=resolve("n_individuals"),
        n_periods=resolve("n_periods"),
        seed=args.seed,
        numeric_transform=args.numeric_transform,
        categorical_encoding=args.categorical_encoding,
        supervised=args.supervised,
        live_view=args.live_view,
        console_ui=args.console_ui,
        panel_features=args.panel_features,
        tune=resolve("tune"),
        iforest_trials=resolve("iforest_trials"),
        vae_trials=resolve("vae_trials"),
        vae_epochs=resolve("vae_epochs"),
        # --top-n 0 is the explicit "use the percentage instead" escape hatch.
        top_n=(None if args.top_n is not None and args.top_n <= 0 else args.top_n),
        top_fraction=args.top_fraction,
        n_val_periods=args.n_val_periods,
        n_test_periods=args.n_test_periods,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_target_far=args.threshold_target_far,
        stack_iforest_into_vae=args.stack_iforest_into_vae,
    )
    # Both apply after construction so they override the dataclass's own
    # `default_factory` dict rather than requiring the CLI to rebuild it.
    if args.contamination is not None:
        config.iforest_params["contamination"] = args.contamination
    if args.p95_percentile is not None:
        config.p95_percentile = args.p95_percentile
    return config


def _close_run_as(status: str, error: Optional[str], live_view: bool) -> None:
    """Close the active run's event stream and refresh the flow diagram.

    Shared by the failure and cancellation paths so both leave a complete,
    parseable `run_events.jsonl` (ending in a real `run_ended` event) plus an
    up-to-date static diagram, instead of a stream that simply stops.
    """
    ctx = observability.current_run()
    if ctx is None:
        return
    observability.end_run(ctx, status=status, error=error)
    try:
        from src.reporting import build_flow_visualization

        # Runs on the abnormal paths especially: seeing exactly which phase
        # stopped is the whole point of the diagram here.
        build_flow_visualization(events_path=ctx.events_path, run_id=ctx.run_id)
    except Exception:
        pass
    if live_view:
        # The live view's HTTP server dies with this process. Hold briefly so
        # its 1s poll can read the final `run_ended` state and render
        # "cancelled"/"failed" before the connection drops -- otherwise the
        # page's last successful poll is a stale "running" snapshot. The page
        # also detects a dropped connection on its own (see
        # `flow_visualization._LIVE_HTML`), so this is a nicety, not the
        # mechanism it depends on.
        time.sleep(1.5)


def _install_sigbreak_handler() -> None:
    """Route Ctrl+Break (Windows `SIGBREAK`) through the same clean shutdown
    as Ctrl+C (`SIGINT`/`KeyboardInterrupt`).

    Verified empirically 2026-08-19, not assumed: `signal.getsignal(signal.
    SIGBREAK)` reads `0` (`SIG_DFL`) by default on this interpreter -- unlike
    `SIGINT`, which CPython wires to `default_int_handler` (i.e. `raise
    KeyboardInterrupt`) out of the box. Left at its default, a Ctrl+Break
    hard-kills the process at the OS level (`STATUS_CONTROL_C_EXIT`) before
    any Python `except` clause ever runs -- confirmed by reproducing it on a
    bare `time.sleep()` script with no application code involved. SIGBREAK is
    Windows-only, so this is a no-op (the `hasattr` guard) everywhere else,
    where Ctrl+C alone already raises `KeyboardInterrupt` correctly.
    """
    if not hasattr(signal, "SIGBREAK"):
        return

    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGBREAK, _raise_keyboard_interrupt)


def main(argv: Optional[list] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    _install_sigbreak_handler()
    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        # NOTE: KeyboardInterrupt inherits from BaseException, *not* Exception,
        # so the `except Exception` below never saw a Ctrl+C. The run's event
        # stream was left without a terminating `run_ended` event, and the live
        # view kept showing the interrupted phase as "running" forever.
        #
        # The dashboard is stopped first on every abnormal path: a Live display
        # owns the bottom of the terminal until it is torn down, so anything
        # logged or printed before that gets interleaved with its repaints.
        console_ui.stop_dashboard()
        logger = setup_logging()
        logger.warning("Run cancelled by user (KeyboardInterrupt).")
        _close_run_as("cancelled", "KeyboardInterrupt: cancelled by user", config.live_view)
        raise SystemExit(130)  # 128 + SIGINT, the conventional shell exit code
    except Exception as exc:
        # `run_pipeline` has no top-level try/except of its own (it is a long,
        # already-tested linear sequence of phases; wrapping it would mean
        # re-indenting the whole function for no behavioral gain). Catching
        # here instead closes the run's structured event stream with
        # status="failed" before re-raising.
        console_ui.stop_dashboard()
        _close_run_as("failed", f"{type(exc).__name__}: {exc}", config.live_view)
        raise
    finally:
        # Belt and braces: SystemExit/GeneratorExit and any path that skipped
        # the handlers above still leave the terminal usable. Idempotent.
        console_ui.stop_dashboard()


if __name__ == "__main__":
    main()
