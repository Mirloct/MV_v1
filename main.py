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

Data sources / inputs: configured panel data, reviewed-label files, model
artifacts, and ``configs/pipeline.yaml``; writes run models, diagnostics, and
reporting context under the project artifact tree.
Created: 2026-08-22
Last modified: 2026-09-26
Changelog:
- 2026-09-26: Enforced the fixed IF/VAE training contract at dataclass, CLI,
  YAML and full-run entry points.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

# Operator preference: keep Python/library warnings out of console and report
# artifacts. Runtime failures still use structured logging at ERROR/CRITICAL;
# diagnostic non-execution remains explicit in the §9 status matrix.
warnings.filterwarnings("ignore")

from src.utils import console_ui, observability, paths
from src.utils.dependency_check import check_dependencies
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

_VAE_MAX_EPOCHS = 15
_IFOREST_CONTAMINATION = 0.005


def _enforce_model_training_contract(config: "PipelineConfig") -> "PipelineConfig":
    """Normalize fixed training controls after every configuration entry point."""
    config.vae_epochs = max(1, min(int(config.vae_epochs), _VAE_MAX_EPOCHS))
    config.diagnostic_experiment_epoch_cap = max(
        1, min(int(config.diagnostic_experiment_epoch_cap), _VAE_MAX_EPOCHS)
    )
    config.iforest_params["n_estimators"] = 300
    config.iforest_params["contamination"] = _IFOREST_CONTAMINATION
    config.vae_params["kl_anneal_epochs"] = 3
    config.vae_params["early_stopping_patience"] = 3
    return config


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Effective configuration for a single pipeline run."""

    n_individuals: int = 2000
    # 16 periods: enough for a 10/2/3/1 chronological split (train/val/test/OOT)
    # and for the 6-period contrast horizon to exist.
    n_periods: int = 16
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
    # Startup check (src/utils/dependency_check.py): requirements.txt packages
    # below their minimum get installed automatically via pip. `--no-auto-install-deps`
    # switches to check-only; `skip_dependency_check` disables the check.
    skip_dependency_check: bool = False
    auto_install_deps: bool = True
    numeric_transform: str = "yeo-johnson"
    categorical_encoding: str = "onehot"
    # Categories below this fraction of rows collapse into one "infrequent"
    # bucket before encoding (`fit_transform_panel`'s own default, previously
    # only reachable by calling the preprocessing module directly). Raising
    # this shrinks how many one-hot columns a high-cardinality categorical
    # expands into -- without abandoning one-hot's category-identity signal
    # for the VAE the way switching to "frequency"/"ordinal" encoding would
    # -- when the granularity is inflating the VAE's per-feature attribution
    # or score contribution; see CONTEXT.md "VAE feature attribution:
    # categorical granularity".
    rare_min_frequency: float = 0.001
    # Numeric NaN fill. "zero" (default) is statistic-free: nothing is
    # estimated from the data, so it cannot leak across the train/test
    # boundary or shift when the fit window changes. Paired with
    # `add_missing_indicators` below, which keeps "this value was absent"
    # recoverable -- a filled 0 is otherwise indistinguishable from a real 0
    # in columns where zero means something. `--no-zero-impute` switches to
    # "median".
    impute_numeric: str = "zero"
    # One 0/1 feature per numeric column that had any missing value in the
    # training fit, flagging "this value was absent" (named `missing__<col>`).
    # ON by default: with `impute_numeric="zero"`, a filled 0 is otherwise
    # indistinguishable from a real 0. Only worth it when missingness itself
    # is informative in your data; `--no-add-missing-indicators` turns it off
    # if it is not, or if seeing `missing__*` names in the output is unwanted
    # regardless. See CONTEXT.md "Missing-indicator features".
    add_missing_indicators: bool = True
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
    # Trailing periods reserved AFTER test, exclusively for the OOT business
    # deliverable (Phase 9's Excel export). Kept strictly separate from
    # `test_mask` -- test is the once-touched block model metrics are reported
    # on; OOT is data neither training, tuning, nor test-set reporting has
    # seen, so the "OOT Excel" never describes the same rows as "test".
    # `0` would collapse OOT back onto test (the historical, since-removed
    # behaviour) -- see `src.evaluation.splits.chronological_split`.
    # Explicit business requirement (2026-08-30): the OOT window is the last
    # 3 months, not 1 -- wide enough to see whether an individual is a
    # one-off or a recurring case across the window, which the last-3-months
    # analyst dashboard depends on. `--n-oot-periods` overrides.
    n_oot_periods: int = 3
    # Fields shown below the entity ID in the analyst profile, one row each
    # (`dashboard.identity_columns`). Not model features: folded into
    # `identification_columns` below automatically. A missing column renders
    # "No disponible"; a value that varies across periods shows the latest
    # non-empty one (`analyst_dashboard._identities_for`).
    analyst_identity_columns: tuple = ("puesto",)
    # Columns that only identify/describe a record: never a model feature, never
    # checked by the zero-row filter or sensitivity study. Superset of
    # `analyst_identity_columns` (folded in automatically) -- a column named only
    # here is excluded from modelling but not shown on the dashboard card.
    # See CONTEXT.md "Identification columns".
    identification_columns: tuple = ()
    # Headline deliverable: everyone at or above this percentile of the OOT
    # score distribution, each row graded p90/p95/p99 so the queue can be
    # triaged. A percentile rather than a fixed headcount because the cut then
    # scales with the portfolio and keeps its distributional meaning.
    # `--top-n N` overrides it with a fixed-size queue instead.
    oot_min_percentile: Optional[float] = 90.0
    top_n: Optional[int] = None
    top_fraction: float = 0.10
    threshold_method: str = "pot"
    threshold_percentile: float = 99.0
    threshold_target_far: float = 1e-3
    # IF -> VAE stacking: append the Isolation Forest's anomaly score to the
    # matrix the VAE trains on. When on, the VAE subsumes the forest's signal
    # and becomes the single deliverable (see `deliverable_models`).
    stack_iforest_into_vae: bool = True
    # Cross-validate this run's IF+VAE against the vendored, optional
    # IF-VAE Diagnostic Suite (`tools/if_vae_diagnostic_suite`, `pip install
    # -e` required -- not a core dependency) and add the "Diagnóstico
    # cruzado IF-VAE" + "Interpretación y recomendaciones" chapters to the
    # report. ON by default as of 2026-09-05 (explicit request: this is now
    # part of the official run) -- see CONTEXT.md "IF-VAE Diagnostic Suite
    # integration". TRADE-OFF: this requires the vendored package installed
    # and adds real runtime -- most of it from `diagnostic_stability_refits`
    # below, which refits both detectors several times. Pass
    # --no-run-diagnostic-suite to skip it (e.g. on a machine without the
    # package installed).
    run_diagnostic_suite: bool = True
    # Threshold grid the diagnostic chapter's sensitivity section sweeps.
    # Defaults to this project's own P90/P95/P99 operating points (the same
    # bands `PERCENTILE_BANDS`/the OOT Excel already use), not an arbitrary
    # choice -- so the section runs by default instead of sitting at
    # NOT_REQUESTED. Values are percentile thresholds in (0, 1).
    diagnostic_sensitivity_grid: tuple = (0.90, 0.95, 0.99)
    # Add the entity-aggregated view to the chapter's scope section. ON by
    # default: the aggregation rule (max score per entity across the OOT
    # window) is the same one `true_oot_entity_scores` already applies
    # elsewhere in this pipeline, so showing it here is not a new rule.
    diagnostic_entity_view: bool = True
    # Seed refits for IF/VAE alert-set stability (top-K Jaccard). Each VAE
    # unit is a full retrain -- real cost on large real-data runs. Minimum 2,
    # 0 disables (reports UNAVAILABLE).
    diagnostic_stability_refits: int = 3
    # Column in the raw panel (`df`) used for the diagnostic chapter's §8
    # per-segment breakdown (temporal/segmentación). Default `"segment"`
    # matches this project's own synthetic panel; point it at any other
    # categorical column the real panel carries (e.g. `"region"`,
    # `"product_type"`, `"channel"`) via `--diagnostic-segment-column NAME`,
    # or pass `--diagnostic-segment-column ""` to disable the breakdown
    # even if a column named "segment" happens to exist. A configured name
    # that is not one of `df`'s columns logs a warning and falls back to
    # NOT_APPLICABLE for that run -- never a silent no-op.
    diagnostic_segment_column: Optional[str] = "segment"
    # The single user-editable file (`configs/pipeline.yaml`, or `--config PATH`) that
    # can set the segment column, the analyst identity columns and the diagnostic
    # grids in ONE place. `cli_explicit` = fields given on the command line (never
    # overridden by the file); `config_sources` records where each file-managed value
    # came from (cli / code / file / default) and lands in the run's resolved config.
    # See `src/utils/config_file.py` for the precedence rules.
    config_file: Optional[str] = None
    cli_explicit: tuple = ()
    config_sources: dict = field(default_factory=dict)
    # Auto-install the vendored IF-VAE Diagnostic Suite
    # (`tools/if_vae_diagnostic_suite`) into this environment on first use if
    # it is not already importable, instead of requiring a manual
    # `pip install -e` beforehand. Always installs from THIS repo's own
    # vendored copy, never from an index. `--no-auto-install-suite` disables
    # it (the run then fails Phase 9c with a clear "not installed" reason
    # instead of installing anything).
    diagnostic_auto_install_suite: bool = True
    # §9 "Experimentos diagnósticos" grids. Contamination is cheap (a forest
    # refit is fast) so it runs by default; capacity/beta are VAE refits --
    # a FULL retrain per grid point -- so they default to an empty grid
    # (NOT_REQUESTED) and only run when explicitly configured.
    diagnostic_experiment_contamination_grid: tuple = (0.02, 0.01, 0.005)
    # Section 9 families that used to be NOT_REQUESTED are ACTIVE BY DEFAULT: latent
    # capacity, beta + KL schedule, loss by feature type, feature-family ablation,
    # temporal backtests and window stability (src/evaluation/ifvae_experiments.py).
    # Grids: `None` = automatic points derived from the production detector; `()` =
    # that sweep is switched off; values = explicit points. All of these live in the
    # `experiments:` block of configs/pipeline.yaml. The VAE retrains share ONE
    # budget (`..._vae_fit_budget`, 0 = no retrains), epochs are capped and very large
    # fits subsampled so a default run stays affordable; what was capped is written
    # into each row of the report.
    diagnostic_experiment_capacity_grid: Optional[tuple] = None
    diagnostic_experiment_beta_grid: Optional[tuple] = None
    diagnostic_experiment_kl_grid: Optional[tuple] = None
    diagnostic_experiment_families: tuple = (
        "capacity", "beta_kl", "loss_by_type", "ablation", "backtest", "window_stability")
    # 1 noise control + 5 capacity + 4 beta/KL + 3 ablation + 2 backtest + 1 one-hot control (embedding mode)
    diagnostic_experiment_vae_fit_budget: int = 16
    diagnostic_experiment_epoch_cap: int = 15
    diagnostic_experiment_max_fit_rows: int = 300_000
    diagnostic_backtest_origins: int = 6
    diagnostic_backtest_vae_origins: int = 2
    diagnostic_backtest_min_fit_periods: int = 4
    # Post-training robustness analysis. It reuses the already fitted
    # preprocessor and detectors (never refits) and introduces labels only for
    # post-hoc performance comparison when they exist.
    run_sensitivity_analysis: bool = True
    sensitivity_max_test_rows: int = 5000
    sensitivity_combination_top_k: int = 6
    sensitivity_random_subsets_per_level: int = 5
    sensitivity_missing_levels: tuple = (0.10, 0.25, 0.50, 0.75, 0.90)
    sensitivity_high_zero_cutoff: float = 0.90
    # Hard, pre-split exclusion (Phase 2): rows where at least this share of
    # input columns are an exact 0 are dropped from `df` itself, before any
    # split, fit, or export -- see the "exact-zero row filter" block above.
    # Distinct from `sensitivity_high_zero_cutoff`, which only studies such
    # rows post-hoc without removing them.
    exact_zero_row_cutoff: float = 0.90
    # -- Reviewed event labels (Phase 8c) ------------------------------------ #
    # data.csv has NO target column. A separate CSV (default folder
    # `data/reviewed_labels/`, or an explicit file via `--labels-path`) gives
    # (entity_id, codmes) rows a reviewed target; it is read only AFTER the
    # detectors are fitted, to evaluate IF/VAE, decide gate 4.5 (how many
    # mature independent positive episodes exist) and, if the gate allows it,
    # run small supervised challengers. No file, an unusable file or a red gate
    # all fall back to the unsupervised IF/VAE run. See
    # `src/evaluation/event_supervision.py`.
    run_event_supervision: bool = True
    labels_path: Optional[str] = None
    labels_dir: str = paths.REVIEWED_LABELS_DIR
    labels_entity_col: Optional[str] = None
    labels_period_col: Optional[str] = None
    labels_target_col: Optional[str] = None
    labels_status_col: Optional[str] = None
    labels_usable_statuses: tuple = ("confirmed", "adjudicated")
    # Rows the file does not mention are UNKNOWN by default ("unreviewed" is never
    # a negative). Turn this on only for a file that is an exhaustive base, and
    # attest the audit, or the gate vetoes it.
    labels_unlisted_as_negative: bool = False
    labels_audit_attested: bool = False
    labels_formal_calc_ok: bool = False
    label_horizon_months: int = 1
    label_confirm_delay_months: int = 0
    label_maturity_buffer_months: int = 0
    label_washout_months: int = 1
    labels_as_of: Optional[str] = None
    review_capacity_k: Optional[int] = None
    hazard_horizon_months: int = 3
    event_challengers: str = "auto"     # off | auto (gate decides) | force
    event_bootstrap_reps: int = 100
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
    # tuned path searches `max_samples`/`max_features` on its own, see
    # docs/models_isolation_forest.md §2b). `n_estimators` is FIXED for both paths
    # (tuning it only inflated the old objective). `max_samples` is an ABSOLUTE
    # row count (rows per tree), never a fraction: the untuned default 4096 is the
    # value the lab found robust on the synthetic panel, but the best size is
    # regime-dependent -- run `--tune` on real data (the paper's 256 is the
    # tuner's reference and is capped to the rows available).
    iforest_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300, "contamination": 0.005,
            "max_samples": 4096, "max_features": 1.0, "bootstrap": False,
        }
    )
    # Absolute `max_samples` range and `max_features` range of the IF tuner.
    iforest_max_samples_range: tuple = (1024, 32768)
    iforest_max_features_range: tuple = (0.5, 1.0)
    # -- Post-tuning selection (src/models/iforest.py::tune_iforest, "Selection rule") --
    # What it guarantees: the tuner's single-seed "winner" can just be a lucky random
    # forest draw, not a genuinely better configuration. Before deploying it, this step
    # re-evaluates the untuned paper default (max_samples=256) AND the top
    # `iforest_selection_top_k` trials on `iforest_noise_seeds` different seeds each,
    # and only deploys the tuned pick if (a) it is not statistically distinguishable
    # from a cheaper near-tied trial (picks the cheaper one instead) and (b) it beats
    # the untuned default by more than 1 noise standard error -- otherwise the untuned
    # default ships. Without it, a "win" that is really just noise could get deployed.
    # Cost: (min(iforest_selection_top_k, iforest_trials) + 1) * iforest_noise_seeds
    # full Isolation Forest fit+score cycles -- bounded independent of `iforest_trials`,
    # but each cycle still scores the FULL validation set (unlike `max_samples`, which
    # is capped absolute), so it is not free on a large real panel. Lower
    # `iforest_selection_top_k` (down to 1, which keeps the default-margin check but
    # drops the "prefer the cheaper near-tied trial" refinement) or `iforest_noise_seeds`
    # (minimum 2) to bound it further; `--iforest-selection-top-k 0` skips the whole
    # step (deploys the tuner's single-seed winner directly, with none of the guarantees
    # above -- only do this if the extra wall-clock time genuinely cannot be spared).
    iforest_selection_top_k: int = 5
    iforest_noise_seeds: int = 3
    # Model selection against reviewed labels. `auto`: when gate 4.5 authorises labels
    # and the validation months hold >= `tune_min_positive_rows` mature positive rows,
    # the IF tuner maximises average precision against them (the only real signal);
    # otherwise it stays label-free (`tail_separation`). `off` never uses them.
    # An explicit `--supervised` (ground-truth column) has priority over both.
    tune_with_labels: str = "auto"
    tune_min_positive_rows: int = 10
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
    # -- How the VAE sees categorical variables (src/models/mixed_vae.py) ---------------
    # `onehot`: the original MLP VAE over the one-hot matrix (kept as the migration path and
    # as the control of the A/B comparison). `embedding`: continuous + binary + ONE integer index
    # per categorical variable (explicit MISSING / UNKNOWN tokens), a learned embedding per variable
    # and one softmax head per variable, so each original variable yields exactly one
    # reconstruction contribution. The IF is unaffected. Edit these in `vae:` of configs/pipeline.yaml.
    vae_categorical_representation: str = "onehot"
    vae_embedding_dimension_strategy: str = "auto"
    vae_embedding_dimension: Optional[int] = None
    vae_embedding_min_dimension: int = 2
    vae_embedding_max_dimension: int = 32
    vae_numeric_loss: str = "huber"
    vae_boolean_loss: str = "binary_cross_entropy"
    vae_categorical_loss: str = "cross_entropy"
    vae_aggregate_by_original_feature: bool = True
    vae_weight_numeric: float = 1.0
    vae_weight_boolean: float = 1.0
    vae_weight_categorical: float = 1.0
    vae_unknown_category_policy: str = "explicit_token"
    vae_missing_category_policy: str = "explicit_token"
    # VAE architecture/training params for the untuned fallback fit (the tuned
    # path searches its own values for most of these -- see
    # docs/models_vae.md §2b). Defaults mirror VAEDetector's own class
    # defaults (src/models/vae.py:315) exactly, so leaving these alone
    # changes nothing; edit here, not in vae.py, to change a pipeline run.
    vae_params: dict = field(
        default_factory=lambda: {
            "latent_dim": 8, "hidden_dim": 64, "n_layers": 2, "dropout": 0.0,
            "beta": 1.0, "lr": 1e-3, "optimizer": "adam", "batch_size": 256,
            "weight_decay": 0.0, "activation": "relu", "kl_anneal_epochs": 3,
            "early_stopping_patience": 3,
        }
    )

    def __post_init__(self) -> None:
        _enforce_model_training_contract(self)

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


def _event_supervision_config(config: "PipelineConfig"):
    """Build the Phase 5b / 8c :class:`EventSupervisionConfig` from the pipeline config."""
    from src.evaluation.event_labels import LabelSpec
    from src.evaluation.event_supervision import EventSupervisionConfig
    from src.models.event_challenger import ChallengerConfig

    return EventSupervisionConfig(
        labels_path=config.labels_path,
        labels_dir=config.labels_dir,
        spec=LabelSpec(
            entity_col=config.labels_entity_col,
            period_col=config.labels_period_col,
            target_col=config.labels_target_col,
            status_col=config.labels_status_col,
            usable_statuses=tuple(s.lower() for s in config.labels_usable_statuses),
            unlisted_as_negative=config.labels_unlisted_as_negative,
            horizon_months=config.label_horizon_months,
            confirm_delay_months=config.label_confirm_delay_months,
            maturity_buffer_months=config.label_maturity_buffer_months,
            washout_months=config.label_washout_months,
            as_of=config.labels_as_of,
        ),
        challenger=ChallengerConfig(
            hazard_horizon_months=config.hazard_horizon_months,
            review_capacity_k=config.review_capacity_k,
            n_boot=config.event_bootstrap_reps, seed=config.seed,
        ),
        challenger_mode=config.event_challengers,
        formal_calculation_ok=config.labels_formal_calc_ok,
        unlisted_attested=config.labels_audit_attested,
        out_dir=REPORTS_DIR,
    )


#: Real-data alternate names for the §8 grouping column, tried (in order, exact
#: case-insensitive match -- never a substring: a segment name is specific enough that
#: substring matching would risk a false positive) only when `diagnostic_segment_column`
#: is still at its built-in default and that default is not a column of this panel. An
#: *explicit* request (file/CLI/code) never falls back here -- it is validated as given.
_SEGMENT_NAME_FALLBACKS: tuple = ("despuestocolaboradoragrupado",)


def _validate_segment_column(config: "PipelineConfig", df, logger) -> None:
    """Fail EARLY (before any fitting) when a segment the user asked for is missing.

    A segment given by the user (config file / CLI / code) that is not a column of
    the panel used to degrade to a one-line warning and an unexplained NOT_APPLICABLE
    section at the very end of the run. Now an *explicit* request that cannot be met
    stops the run up front, naming the column asked for, the closest match ignoring
    case/whitespace and every available column; only the built-in default degrades
    (warning + observability incident) -- but first tries `_SEGMENT_NAME_FALLBACKS`,
    this project's own real-data name(s) for the same grouping.
    """
    import difflib

    name = (config.diagnostic_segment_column or "").strip()
    if not name or name in df.columns:
        return
    source = config.config_sources.get("diagnostic_segment_column", "default")
    if source == "default":
        norm_cols = {str(c).strip().lower(): c for c in df.columns}
        for candidate in _SEGMENT_NAME_FALLBACKS:
            hit = norm_cols.get(candidate.strip().lower())
            if hit:
                logger.info(
                    "diagnostic.segment_column default %r is not in this panel; using "
                    "%r instead (known alternate name for the same §8 grouping).",
                    name, hit,
                )
                observability.check(
                    name="config.segment_column_present", category="data",
                    definition="The configured segment column (or a known alternate "
                               "name for it) exists in the panel.",
                    expected=f"{name!r} or an alternate name in panel columns",
                    severity="warning", passed=True,
                    observed={"requested": name, "resolved": hit, "source": source},
                    evidence="configs/pipeline.yaml",
                )
                config.diagnostic_segment_column = hit
                return
    norm = {str(c).strip().lower(): c for c in df.columns}
    close = norm.get(name.lower()) or next(
        iter(difflib.get_close_matches(name, [str(c) for c in df.columns], 1, 0.6)), None)
    msg = (f"La columna de segmento {name!r} no existe en el panel. "
           + (f"¿Quisiste decir {close!r}? " if close else "")
           + f"Columnas disponibles: {', '.join(map(str, df.columns))}. "
           "Corrígela en configs/pipeline.yaml (diagnostic.segment_column) o usa '' para desactivar.")
    source = config.config_sources.get("diagnostic_segment_column", "default")
    explicit = source != "default"
    observability.check(
        name="config.segment_column_present", category="data",
        definition="The configured segment column exists in the panel.",
        expected=f"{name!r} in panel columns", severity="critical" if explicit else "warning",
        passed=False, observed={"requested": name, "closest": close, "source": source},
        evidence="configs/pipeline.yaml",
    )
    if explicit:
        raise ValueError(msg)
    logger.warning("%s La sección 8 por segmento quedará NOT_APPLICABLE.", msg)


def _resolve_identification_columns(config: "PipelineConfig", df, logger) -> tuple:
    """The columns no modelling phase may see: ``identification_columns`` plus every
    field in ``analyst_identity_columns`` (folded in automatically -- an identity
    column is by definition "for identification, not for the model", so none of them
    needs to be listed twice). A configured name absent from this panel only warns:
    unlike the segment column, nothing downstream breaks if it is a no-op here, and
    the same config file is meant to run against panels that do not all carry the
    same optional columns.
    """
    requested = tuple(dict.fromkeys(  # de-dup, keep order
        [c for c in config.identification_columns if c]
        + [c for c in config.analyst_identity_columns if c]
    ))
    missing = [c for c in requested if c not in df.columns]
    if missing:
        logger.warning(
            "identification_columns/dashboard.identity_columns %s not in this panel; "
            "ignored (no effect on feature building). Available columns: %s.",
            missing, ", ".join(map(str, df.columns)),
        )
    resolved = tuple(c for c in requested if c in df.columns)
    observability.check(
        name="config.identification_columns_present", category="data",
        definition="Every configured identification column exists in the panel and is "
                   "excluded from every modelling phase (features, VAE categorical "
                   "sources, exact-zero-row filter, transform diagnostics, sensitivity).",
        expected="requested ⊆ panel columns", severity="warning", passed=not missing,
        observed={"requested": list(requested), "resolved": list(resolved), "missing": missing},
        evidence="configs/pipeline.yaml (data.identification_columns, dashboard.identity_columns)",
    )
    return resolved


def _existing_identity_columns(identity_columns: tuple, resolved_identification_columns: tuple) -> tuple:
    """Dashboard display fields trimmed to those that survived
    ``_resolve_identification_columns`` (i.e. actually exist in this panel), order
    preserved. A configured field absent from the real panel must never reach
    ``build_analyst_dashboard`` -- the warning for it was already logged by
    ``_resolve_identification_columns`` above, so this only drops it, it does not
    warn a second time.
    """
    return tuple(c for c in identity_columns if c in resolved_identification_columns)


def _mixed_vae_config(config: "PipelineConfig"):
    """The embedding / loss / token settings of the mixed-type VAE (validated on construction)."""
    from src.models.mixed_vae import MixedVAEConfig

    return MixedVAEConfig(
        embedding_dimension_strategy=config.vae_embedding_dimension_strategy,
        embedding_dimension=config.vae_embedding_dimension,
        embedding_min_dimension=config.vae_embedding_min_dimension,
        embedding_max_dimension=config.vae_embedding_max_dimension,
        numeric_loss=config.vae_numeric_loss, boolean_loss=config.vae_boolean_loss,
        categorical_loss=config.vae_categorical_loss,
        aggregate_by_original_feature=config.vae_aggregate_by_original_feature,
        weight_numeric=config.vae_weight_numeric, weight_boolean=config.vae_weight_boolean,
        weight_categorical=config.vae_weight_categorical,
        unknown_category_policy=config.vae_unknown_category_policy,
        missing_category_policy=config.vae_missing_category_policy,
    )


def _metric_matrix(detector, X_model):
    """Distance-based views (silhouette, PCA/UMAP) must not read category *indices* as magnitudes:
    for the mixed VAE they use the continuous / binary / missing-flag block only."""
    layout = getattr(detector, "layout", None)
    if layout is None:
        return X_model
    keep = np.concatenate([layout.positions("num"), layout.positions("bool"), layout.positions("flag")])
    return np.asarray(X_model)[:, np.sort(keep)]


def _add_fig(figs: list, title: str, path: Optional[str]) -> None:
    """Append a figure entry if its file actually exists on disk."""
    if path and os.path.isfile(path):
        figs.append({"title": title, "path": path})


def _flagged_rows_for_split_count(
    oot_mask: np.ndarray, scores: np.ndarray, threshold: float,
    *, min_rows: int = 10, fallback_n: int = 50, logger=None,
) -> np.ndarray:
    """Which OOT rows count as "flagged" for `split_count_analysis`: normally the
    calibrated-threshold alert set, so the chart answers "how are THIS run's real
    alerts isolated" -- but a strict business threshold (this project's default POT
    calibration targets a 0.1% false-alarm rate) can legitimately flag zero rows in a
    small OOT window, which is real, not a bug, and would leave nothing to analyse.
    Falls back to the top OOT rows by score instead of skipping the chart outright;
    logged, never silent about which one happened.
    """
    flagged_mask = (
        oot_mask & (scores >= threshold) if np.isfinite(threshold) else np.zeros_like(oot_mask)
    )
    if int(flagged_mask.sum()) >= min_rows or not oot_mask.any():
        return flagged_mask
    oot_idx = np.flatnonzero(oot_mask)
    n_fallback = min(oot_idx.size, max(min_rows, fallback_n))
    top_idx = oot_idx[np.argsort(-scores[oot_idx])[:n_fallback]]
    if logger is not None:
        logger.info(
            "split_count_analysis: only %d OOT row(s) above the calibrated threshold; "
            "using the top %d OOT row(s) by score instead so the per-feature reading has "
            "enough rows to be meaningful.", int(flagged_mask.sum()), n_fallback,
        )
    result = np.zeros_like(oot_mask)
    result[top_idx] = True
    return result


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
        with log_phase("evaluation.supervised_metrics"):
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
            with log_phase("evaluation.metrics_by_anomaly_type"):
                metrics["by_type"] = metrics_by_anomaly_type(labels, label_types, scores)
                metrics["oot_by_type"] = metrics_by_anomaly_type(
                    labels[oot_mask], np.asarray(label_types)[oot_mask], scores[oot_mask]
                )
    with log_phase("evaluation.unsupervised_metrics"):
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
    # Before any phase does real work: fixes a missing/outdated package here
    # rather than an opaque ImportError later (most third-party imports in
    # this codebase are deferred, not at module top, so a fix still applies
    # to the rest of this run).
    if not config.skip_dependency_check:
        requirements_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "requirements.txt"
        )
        if not check_dependencies(
            requirements_path, logger=logger, auto_install=config.auto_install_deps,
        ):
            raise SystemExit(
                "No se pudieron asegurar las dependencias del pipeline -- revisa "
                "execution.log para el detalle (pip falló o --no-auto-install-deps "
                "está activo) y el comando `pip install` exacto para resolverlo a mano."
            )
    # Mirrors ERROR/CRITICAL records for the report's "Qué no se ejecutó o
    # falló" section. Warnings stay out of the report by operator request;
    # `execution.log` remains the complete runtime record. It is detached
    # before the final summary so a second in-process run never inherits
    # the first run's incidents.
    from src.utils.logging_config import IncidentCollector

    # Idempotent: drop any collector left attached by a PRIOR failed
    # `run_pipeline` call in this same process (e.g. repeated calls in a
    # test session) before adding this run's own, so incidents never leak
    # across runs sharing the process-global logger.
    for _h in list(logger.handlers):
        if isinstance(_h, IncidentCollector):
            logger.removeHandler(_h)
    incident_collector = IncidentCollector()
    logger.addHandler(incident_collector)
    from src.utils.config_file import apply_config_file, find_config_file

    _cfg_report = apply_config_file(
        config, find_config_file(config.config_file), frozenset(config.cli_explicit)
    )
    _enforce_model_training_contract(config)
    config.config_sources = dict(_cfg_report["sources"])
    logger.info("=" * 72)
    logger.info("Anomaly-detection pipeline starting")
    logger.info("Configuration file: %s | sources: %s",
                _cfg_report["path"] or "(none found)", config.config_sources)
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
    deliverable_tables: dict = {}
    # One representative, explained OOT row per entity and detector.  Unlike
    # ``deliverable_tables`` these are not P90/P95 filtered: the dashboard's
    # P95 union can contain a VAE-only entity below IF's export threshold (or
    # vice versa), and its profile must still show both detectors' variables.
    dashboard_explanation_tables: dict = {}
    analyst_dashboard_path: Optional[str] = None
    diagnostic_suite_result: Optional[dict] = None
    sensitivity_result: Optional[dict] = None
    model_specs: dict = {}
    chart_data: dict = {"models": {}}
    # Numeric payloads behind the charts that used to be embedded PNGs. The
    # report renders 100% Plotly, so it needs the arrays rather than an image;
    # the PNGs are still written to artifacts/ as run evidence.
    chart_static: dict = {}
    # Posterior-collapse diagnostics, per model. Keyed by model so the report
    # renders them beside that model's other numbers.
    model_latent_diagnostics: dict = {}
    # Full per-feature attribution, per model -- the uncropped counterpart to
    # the top-20 SHAP/reconstruction-error figures. Exported as a workbook
    # after the interpretability phase; see attribution_export.py.
    model_attributions: dict = {}
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
        from src.data import drop_exact_zero_rows, load_or_generate_panel

        df, schema = load_or_generate_panel(
            data_path=config.data_path,
            n_individuals=config.n_individuals,
            n_periods=config.n_periods,
            seed=config.seed,
        )
        # Set once, here, before anything reads `schema`: every phase that decides what
        # the model sees (feature building, VAE categorical sources, the exact-zero-row
        # filter below, transform diagnostics, post-training sensitivity) calls
        # `src.data.loader.key_columns(schema)` instead of keeping its own column list,
        # so this is the only place identification columns need to be resolved.
        schema.identification_columns = _resolve_identification_columns(config, df, logger)
        # Trim the dashboard display list to fields that actually exist in this
        # panel, before `build_analyst_dashboard` sees it -- an absent field is
        # dropped silently, never a permanent "No disponible" row.
        config.analyst_identity_columns = _existing_identity_columns(
            config.analyst_identity_columns, schema.identification_columns
        )
        # -- exact-zero row filter ------------------------------------------- #
        # Hard exclusion, applied to `df` itself before the chronological
        # split, any fit, and every export: a row whose input columns are
        # (almost) all a literal 0 is nearly empty, so both detectors would
        # score it as extreme (or trivially normal) for that reason alone.
        # Unlike the post-training `sensitivity_high_zero_cutoff` study in
        # Phase 9d -- which only *measures* such rows' effect on fitted
        # models -- this removes them, so nothing downstream ever sees them.
        # The helper logs rows before / dropped / after to execution.log.
        _validate_segment_column(config, df, logger)
        df, zero_filter_stats = drop_exact_zero_rows(
            df, schema, logger, cutoff=config.exact_zero_row_cutoff,
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
        console_ui.set_stat(
            "Filas excluidas (cero exacto)", f"{zero_filter_stats['n_rows_dropped']:,}"
        )

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
    # Four blocks, four distinct jobs: train fits preprocessing + models,
    # validation selects hyperparameters and calibrates the alert threshold,
    # test is read exactly once at the end (model metrics/ROC-PR/threshold
    # diagnostics), and OOT -- strictly later than test, never touched by any
    # of the above -- is reserved for the Phase 9 business deliverable Excel.
    # `eval_mask` (== test_mask) and `oot_mask` are deliberately DIFFERENT
    # blocks: aliasing them (the historical behaviour) made the "OOT Excel"
    # describe the exact same rows as the reported test metrics, which is the
    # confusion `n_oot_periods` exists to remove.
    with log_phase("Phase 3a: chronological split"):
        from src.evaluation import chronological_split

        split = chronological_split(
            df, time_col=time_col,
            n_val_periods=config.n_val_periods,
            n_test_periods=config.n_test_periods,
            n_oot_periods=config.n_oot_periods,
        )
        train_mask, val_mask, test_mask = split.train_mask, split.val_mask, split.test_mask
        oot_mask, oot_periods = split.oot_mask, split.oot_periods
        # Preprocessing may only learn from train; tuning may see train+val.
        in_mask = train_mask | val_mask     # everything the models may touch
        # `eval_mask` feeds every "OOT ROC-AUC"/"OOT PR-AUC"/threshold-alert
        # metric below -- historically named for this block ("test is read
        # exactly once, at the end") and kept on `test_mask` so those numbers
        # are unaffected by adding the separate `oot_mask` block above.
        eval_mask = test_mask
        eval_period_str = ", ".join(str(v)[:10] for v in np.atleast_1d(split.test_periods))
        oot_period_str = (
            ", ".join(str(v)[:10] for v in np.atleast_1d(oot_periods))
            if oot_periods.size else "(none)"
        )
        assert not np.any(test_mask & oot_mask), (
            "test_mask and oot_mask overlap -- chronological_split must guarantee "
            "OOT periods are strictly later than test periods."
        )
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
        X, keys, feature_names, fitted_preprocessor = fit_transform_panel(
            df,
            schema,
            fit_mask=train_mask,
            numeric_transform=config.numeric_transform,
            categorical_encoding=config.categorical_encoding,
            rare_min_frequency=config.rare_min_frequency,
            impute_numeric=config.impute_numeric,
            add_panel_features=config.panel_features,
            add_missing_indicators=config.add_missing_indicators,
            random_state=config.seed,
            return_pipeline=True,
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
            n_oot_periods=config.n_oot_periods,
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
        if not np.array_equal(keys_split.oot_mask, oot_mask):
            n_disagree = int(np.sum(keys_split.oot_mask != oot_mask))
            raise assumptions.LeakageAssumptionError(
                f"OOT split recomputed from `keys` disagrees with the mask scoring will "
                f"use on {n_disagree} row(s) -- the OOT business deliverable would no "
                f"longer be guaranteed disjoint from test.",
                check="leakage.oot_split_row_order_preserved",
                observed={"n_disagreeing_rows": n_disagree, "total_rows": len(oot_mask)},
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
            "Model input: %d rows (train %d + val %d) | test rows=%d (period(s)=%s) | "
            "OOT deliverable rows=%d (period(s)=%s)",
            int(in_mask.sum()), int(train_mask.sum()), int(val_mask.sum()),
            int(eval_mask.sum()), eval_period_str,
            int(oot_mask.sum()), oot_period_str,
        )

    # -- Phase 5b: reviewed labels + gate 4.5, BEFORE tuning ----------------- #
    # The labels file is read here (not only in Phase 8c) because gate 4.5 decides
    # whether the IF tuner may select hyper-parameters against reviewed labels --
    # the only real signal available. Labels reach the tuner as a held-out target
    # for the validation months only (never as training data). Any failure or a
    # red gate leaves tuning label-free; Phase 8c reuses this same load.
    event_prepared = None
    tuning_labels = {"y": None, "used": False, "reason": "not evaluated"}
    if config.run_event_supervision and config.tune and config.tune_with_labels != "off" \
            and not supervised:
        with log_phase("Phase 5b: reviewed labels for tuning (gate 4.5)"):
            try:
                from src.evaluation.event_labels import LabelSpec
                from src.evaluation.event_supervision import (
                    labels_for_tuning,
                    prepare_event_labels,
                )

                event_prepared = prepare_event_labels(
                    cfg=_event_supervision_config(config), schema=schema, keys=keys,
                    masks={"train": train_mask, "val": val_mask,
                           "test": test_mask, "oot": oot_mask},
                )
                tuning_labels = labels_for_tuning(
                    event_prepared, in_mask=in_mask, val_mask=val_mask,
                    min_positive_rows=config.tune_min_positive_rows,
                )
            except Exception as exc:  # noqa: BLE001 - labels are optional, never block tuning
                event_prepared = None
                tuning_labels = {"y": None, "used": False, "reason": f"labels step failed: {exc}"}
                logger.warning("Reviewed labels for tuning failed (%s); tuning stays label-free.", exc)
            logger.info(
                "Tuning objective: %s (%s)",
                "reviewed labels (AP on validation months)" if tuning_labels["used"] else "label-free (tail_separation)",
                tuning_labels["reason"],
            )
            observability.check(
                name="tuning.objective_source", category="training",
                definition="Which signal selects the IF hyper-parameters: reviewed labels "
                           "(only when gate 4.5 authorises them) or the label-free proxy.",
                expected="labels used only when authorised by gate 4.5 and enough validation positives",
                severity="info", passed=True,
                observed={k: v for k, v in tuning_labels.items() if k != "y"},
                evidence="src.evaluation.event_supervision.labels_for_tuning",
            )

    # -- Phase 6: Isolation Forest ------------------------------------------ #
    with log_phase("Phase 6: Isolation Forest"):
        from src.models import (
            IsolationForestDetector,
            plot_score_distribution,
            tune_iforest,
        )

        # Blocking gate: validates the *fallback* config (used verbatim when
        # --no-tune); a bad contamination value here would otherwise surface
        # only much later as a silently-wrong threshold.
        assumptions.validate_iforest_config(
            contamination=config.iforest_params.get("contamination", 0.005),
            n_estimators=config.iforest_params.get("n_estimators", 300),
            max_samples=config.iforest_params.get("max_samples", 4096),
        )
        assumptions.validate_matrix_for_fit(X_in, "iforest")

        # Absolute `max_samples` never exceeds the rows the forest is fitted on.
        fallback_if_params = dict(config.iforest_params)
        if isinstance(fallback_if_params.get("max_samples"), int):
            fallback_if_params["max_samples"] = min(fallback_if_params["max_samples"], X_in.shape[0])

        if_detector = None
        if_tuning_ok = False
        if config.tune:
            try:
                # Priority: explicit ground truth (--supervised) > reviewed labels
                # authorised by gate 4.5 > label-free tail_separation.
                if supervised:
                    tune_y, tune_label_source = labels_in, "ground_truth"
                elif tuning_labels["used"]:
                    tune_y, tune_label_source = tuning_labels["y"], "reviewed_labels"
                else:
                    tune_y, tune_label_source = None, "labels"
                tune_iforest(
                    X_in,
                    n_trials=config.iforest_trials,
                    y=tune_y,
                    label_source=tune_label_source,
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
                    # Only the deployed detector's operating point; it does not
                    # take part in model selection.
                    contamination=config.iforest_params.get("contamination", 0.005),
                    n_estimators=config.iforest_params.get("n_estimators", 300),
                    max_samples_range=tuple(config.iforest_max_samples_range),
                    max_features_range=tuple(config.iforest_max_features_range),
                    bootstrap=bool(config.iforest_params.get("bootstrap", False)),
                    # See the comment on `iforest_selection_top_k` above for exactly
                    # what this step guarantees and what it costs.
                    selection_top_k=config.iforest_selection_top_k,
                    noise_seeds=config.iforest_noise_seeds,
                    holdout_frac=config.iforest_holdout_frac,
                    early_stopping_patience=config.iforest_tuning_early_stopping["patience"],
                    early_stopping_min_delta=config.iforest_tuning_early_stopping["min_delta"],
                    early_stopping_min_trials=config.iforest_tuning_early_stopping["min_trials"],
                )
                if_detector = IsolationForestDetector.load(IFOREST_MODEL)
                if_tuning_ok = True
            except Exception as exc:
                logger.error(
                    "iForest tuning/load failed (%s); fitting the default detector %s.",
                    exc, {k: fallback_if_params[k] for k in ("n_estimators", "max_samples", "max_features")},
                )
            observability.check(
                name="tuning.iforest_completed", category="training",
                definition="The IF tuning study finished and its refit model was loaded.",
                expected="tuning completes; otherwise the run falls back to the default forest",
                severity="warning", passed=if_tuning_ok,
                observed={"tuned": if_tuning_ok, "fallback_params": fallback_if_params},
                failure_action="The default forest was used; its parameters are recorded, and any "
                               "stale best-params YAML from an earlier run is ignored.",
                evidence=IFOREST_BEST_PARAMS,
            )
        if if_detector is None:
            if_detector = IsolationForestDetector(
                random_state=config.seed, **fallback_if_params
            )
            if_detector.fit(X_in)

        if_scores = if_detector.score_samples(X_if)
        # A YAML left by an earlier run must not describe this run's model when
        # tuning failed or was off: use the params of the detector actually fitted.
        if_best_params = (
            (_read_best_params(IFOREST_BEST_PARAMS) if if_tuning_ok else {})
            or {k: getattr(if_detector, k) for k in
                ("n_estimators", "max_samples", "max_features", "contamination", "bootstrap")}
        )
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
            split_masks={"train": train_mask, "val": val_mask, "test": test_mask, "oot": oot_mask},
            percentile=config.p95_percentile, model_name="iforest",
        )
        logger.info(
            "IF P95 checkpoint: %d/%d rows >= threshold %.6f -> %s",
            len(p95_table), len(df), p95_threshold, p95_path,
        )

    # -- Phase 6d: IF OOT export (validation, stacked mode only) ------------ #
    # Answers a concrete risk of stacking: once the forest's score becomes
    # just one input feature to the VAE, the VAE's own ranking could in
    # principle "lose" OOT individuals the forest alone would have flagged.
    # Exported here -- right after the forest is fit, before its score is
    # folded into the VAE's matrix below -- so it reflects the forest alone,
    # not the stacked model. Only meaningful when stacking would otherwise
    # hide IF's own queue entirely (`"iforest" not in
    # config.deliverable_models`); in parallel mode IF already ships this
    # exact export via Phase 9's per-model loop, so doing it again here would
    # just recompute and overwrite an identical file.
    if config.stack_iforest_into_vae:
        with log_phase("Phase 6d: IF OOT export (validation)"):
            try:
                from src.evaluation import (
                    build_scored_frame,
                    calibrate_threshold,
                    export_oot_top_anomalies,
                )

                # Calibrated on validation only, exactly like Phase 8b does
                # per model later -- this is the forest's own threshold, not
                # borrowed from the VAE.
                if_cal = calibrate_threshold(
                    if_scores[val_mask],
                    method=config.threshold_method,
                    percentile=config.threshold_percentile,
                    target_far=config.threshold_target_far,
                )
                if_scored_df = build_scored_frame(df, keys, if_scores, schema)
                try:
                    if oot_mask.any():
                        from src.interpretability import explain_rows_iforest

                        top_vars = explain_rows_iforest(
                            if_detector, X_if[oot_mask], feature_names=names_if,
                        )
                        if_scored_df.loc[oot_mask, "top_5_variables"] = top_vars
                        dashboard_explanation_tables["iforest"] = (
                            if_scored_df.loc[oot_mask]
                            .sort_values("anomaly_score", ascending=False)
                            .drop_duplicates(schema.entity_col or "entity_id", keep="first")
                            .reset_index(drop=True)
                        )
                except Exception as exc:  # noqa: BLE001 - never block this export
                    logger.warning(
                        "[iforest] per-row explanation for the OOT validation "
                        "export failed (%s); continuing without it.", exc,
                    )
                if_oot_path, if_oot_table = export_oot_top_anomalies(
                    if_scored_df, schema,
                    min_percentile=(None if config.top_n is not None
                                    else config.oot_min_percentile),
                    top_n=config.top_n, top_fraction=config.top_fraction,
                    model_name="iforest", n_oot_periods=config.n_oot_periods,
                    threshold=if_cal["threshold"],
                )
                oot_excels["iforest"] = if_oot_path
                deliverable_tables["iforest"] = if_oot_table
                logger.info(
                    "IF OOT validation export (pre-stacking, threshold=%.6f) -> %s "
                    "(%d rows). Compare against the VAE's stacked queue to check "
                    "whether stacking dropped individuals the forest alone flags.",
                    if_cal["threshold"], if_oot_path, len(if_oot_table),
                )
                _if_oot_ok = (os.path.isfile(if_oot_path)
                             and os.path.getsize(if_oot_path) > 0)
                observability.check(
                    name="artifact.oot_excel_written[iforest_validation]",
                    category="artifact",
                    definition="The Isolation Forest's own pre-stacking OOT "
                               "risk-ranked Excel exists and is non-empty, so it "
                               "can be compared against the VAE's stacked queue.",
                    expected="file exists and size_bytes > 0", severity="warning",
                    passed=_if_oot_ok,
                    observed={"path": if_oot_path,
                              "size_bytes": os.path.getsize(if_oot_path) if _if_oot_ok else 0,
                              "rows": int(len(if_oot_table))},
                    failure_action="Best-effort validation artifact; the P95 "
                                   "checkpoint above and the VAE's own OOT "
                                   "deliverable below are unaffected.",
                    evidence=if_oot_path,
                )
            except Exception as exc:  # noqa: BLE001 - never block stacking/VAE below
                logger.warning(
                    "IF OOT validation export failed (%s); the P95 checkpoint "
                    "above and the VAE deliverable below are unaffected.", exc,
                )

    # -- Phase 6b: IF -> VAE stacking --------------------------------------- #
    # The forest's score becomes an extra column of the matrix the VAE trains
    # on, so the VAE models the normal manifold *including* how isolated the
    # forest finds each row.
    X_vae, vae_feature_names = X, feature_names
    stack_info = None
    # -- VAE view. IF keeps its own matrix (`X_if`, one-hot withheld). `onehot`: the VAE gets the
    # full matrix as before. `embedding`: continuous + binary + missing-flag columns taken from the
    # same causal preprocessing, plus ONE integer index per categorical variable (MISSING / UNKNOWN
    # tokens; vocabularies learned on the train rows only) -- no one-hot column enters the VAE.
    vae_layout = vae_builder = vae_mixed_cfg = None
    _stack_scaler = None
    if config.vae_categorical_representation == "embedding":
        with log_phase("Phase 6a: VAE mixed-type view (embeddings)"):
            from src.preprocessing.mixed_view import (
                MixedViewBuilder,
                MixedViewConfig,
                assert_no_onehot,
                categorical_sources,
            )

            vae_mixed_cfg = _mixed_vae_config(config)
            vae_builder = MixedViewBuilder(MixedViewConfig(
                min_frequency=config.rare_min_frequency,
                unknown_category_policy=config.vae_unknown_category_policy,
                missing_category_policy=config.vae_missing_category_policy,
            ))
            X_vae = vae_builder.fit_transform(df, X, feature_names, train_mask, categorical_sources(df, schema))
            vae_layout = vae_builder.layout
            vae_feature_names = list(vae_layout.columns)
            assert_no_onehot(vae_layout)
            logger.info(
                "VAE view: %d columns (%d numeric, %d binary, %d missing-flag, %d categorical index) "
                "instead of %d one-hot-expanded columns; cardinalities %s; layout %s",
                vae_layout.n_columns, len(vae_layout.names("num")), len(vae_layout.names("bool")),
                len(vae_layout.names("flag")), len(vae_layout.names("cat")), len(feature_names),
                {s.name: s.cardinality for s in vae_layout.cat_specs()}, vae_layout.fingerprint(),
            )
            observability.check(
                name="vae.no_onehot_input", category="data",
                definition="No one-hot dummy column enters the VAE: every categorical variable is one index column.",
                expected="no cat__ column with a role other than 'cat'", severity="critical", passed=True,
                observed={"columns": vae_layout.n_columns, "categorical_variables": len(vae_layout.names("cat")),
                          "onehot_columns_replaced": int(sum(n.startswith("cat__") for n in feature_names))},
                evidence="src/preprocessing/mixed_view.py",
            )
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
                {"validation": val_mask, "test": test_mask, "oot": oot_mask},
            )
            if vae_layout is None:
                stacked = build_stacked_matrix(
                    X, stack_scores, fit_mask=train_mask, feature_names=feature_names,
                )
                X_vae, vae_feature_names = stacked.X, stacked.feature_names
                _stack_score_name = stacked.score_name
                _stack_scaler = stacked.scaler
            else:
                # Mixed matrix: only the appended score is standardised (train rows only); the
                # category indices and binary flags must stay as they are.
                from sklearn.preprocessing import StandardScaler
                from src.models.stacking import DEFAULT_SCORE_FEATURE

                _stack_scaler = StandardScaler().fit(np.asarray(stack_scores, float)[train_mask].reshape(-1, 1))
                col = _stack_scaler.transform(np.asarray(stack_scores, float).reshape(-1, 1)).astype(np.float32)
                X_vae = np.hstack([X_vae, col])
                _stack_score_name = DEFAULT_SCORE_FEATURE
                vae_layout = vae_layout.with_extra_numeric(_stack_score_name)
                vae_feature_names = list(vae_layout.columns)
            logger.info(
                "VAE input: %d columns (last column = %r)", X_vae.shape[1], _stack_score_name,
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
        vae_tuning_ok = False
        # Architecture the run is configured for: a checkpoint of the other one is REJECTED on load.
        _vae_arch = "mixed_v1" if vae_layout is not None else "onehot_v1"
        _vae_fp = None if vae_layout is None else vae_mixed_cfg.fingerprint(vae_layout)
        _vae_kwargs = {} if vae_layout is None else {"layout": vae_layout, "mixed_config": vae_mixed_cfg}
        if config.tune:
            try:
                tune_vae(
                    X_vae_in,
                    n_trials=config.vae_trials,
                    # Ground truth (--supervised) > reviewed labels authorised by gate 4.5 (partial: NaN =
                    # unknown, validation months only) > label-free ELBO.
                    y=(labels_in if supervised else (tuning_labels["y"] if tuning_labels["used"] else None)),
                    label_source=("ground_truth" if supervised else "reviewed_labels"),
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
                    feature_names=vae_feature_names,
                    **_vae_kwargs,
                )
                vae_detector = VAEDetector.load(VAE_MODEL, expect_architecture=_vae_arch,
                                                expect_fingerprint=_vae_fp)
                vae_tuning_ok = True
            except Exception as exc:
                logger.error(
                    "VAE tuning/load failed (%s); fitting the default detector.", exc
                )
            observability.check(
                name="tuning.vae_completed", category="training",
                definition="The VAE tuning study finished and its refit model was loaded.",
                expected="tuning completes; otherwise the run falls back to the default VAE",
                severity="warning", passed=vae_tuning_ok,
                observed={"tuned": vae_tuning_ok},
                failure_action="The default VAE was used; a best-params YAML left by an "
                               "earlier run is ignored.",
                evidence=VAE_BEST_PARAMS,
            )
        if vae_detector is None:
            vae_detector = VAEDetector(
                random_state=config.seed, epochs=config.vae_epochs, **config.vae_params, **_vae_kwargs
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
        vae_best_params = _read_best_params(VAE_BEST_PARAMS) if vae_tuning_ok else {}
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
    # Headline "flagged for review" count for the report's hero figure --
    # deliberately NOT the ground-truth positive count (`n_pos` above): that
    # answers "how many are truly anomalous," which is unknowable on real,
    # unlabeled data and is exactly why this number came back 0 on a real
    # run with no ground-truth file. Set from the deliverable model's own
    # OOT export below (last deliverable processed wins, i.e. the VAE's
    # under stacking or in parallel mode, since it always runs last in this
    # loop) and left None if no deliverable ran, so Phase 11 can fall back
    # to the ground-truth count rather than showing nothing.
    oot_review_count: Optional[int] = None
    oot_review_rate: Optional[float] = None

    for name, (detector, scores, best_params, X_model, names_model) in models.items():
        # -- Phase 8: evaluation -------------------------------------------- #
        with log_phase(f"Phase 8: evaluation [{name}]"):
            from src.evaluation import plot_embedding, plot_roc_pr

            X_metric = _metric_matrix(detector, X_model)
            metrics = _model_metrics(
                supervised, labels, scores, eval_mask, X_metric, label_types=label_types
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
            #
            # `true_oot_entity_scores` is deliberately NOT `scores[eval_mask]`:
            # `eval_mask` is the test block (see the Phase 3a comment on why
            # `eval_mask`/`oot_mask` are kept distinct), and it is not
            # deduplicated by entity -- an entity can appear more than once
            # when `n_oot_periods > 1`. This dict is keyed by `entity_id` and
            # holds each entity's max score across the genuine OOT window,
            # the same dedup Phase 9's `export_oot_top_anomalies` applies --
            # so the model-agreement chart below describes the exact same
            # population as the alert queue: one row per real individual,
            # drawn only from held-out OOT rows.
            oot_ids = keys[entity_col].to_numpy()[oot_mask]
            oot_entity_scores = scores[oot_mask]
            true_oot_entity_scores: dict = {}
            for eid, sc in zip(oot_ids, oot_entity_scores):
                k = str(eid)
                prev = true_oot_entity_scores.get(k)
                if prev is None or sc > prev:
                    true_oot_entity_scores[k] = float(sc)

            chart_data["models"][name] = {
                "oot_scores": [float(v) for v in scores[eval_mask]],
                "oot_labels": ([int(v) for v in labels[eval_mask]] if supervised else None),
                "true_oot_entity_scores": true_oot_entity_scores,
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
                        X_metric, scores, method="pca",
                        y=(labels if supervised else None),
                        filename=f"embedding_{name}_pca.png",
                    ),
                )
            except Exception as exc:
                logger.warning("plot_embedding(pca) failed for %s (%s).", name, exc)
            try:  # UMAP is optional; falls back internally but guard the import too
                emb_path, emb_data = plot_embedding(
                    X_metric, scores, method="umap",
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
                            labels[eval_mask], scores[eval_mask],
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
            test_scores = scores[eval_mask]
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
                from src.evaluation import (
                    BAND_COL,
                    build_scored_frame,
                    export_oot_top_anomalies,
                    oot_period,
                )

                scored_df = build_scored_frame(df, keys, scores, schema)

                # Per-row "why is this individual flagged" explanation, for
                # exactly the OOT rows this deliverable can select from (not
                # the whole panel -- an alert queue is small, a real panel
                # is not). `build_scored_frame` returns one row per `keys`
                # row in `keys` order, so `oot_mask` computed against `keys`
                # indexes `scored_df`/`X_model` positionally correctly.
                try:
                    time_col = schema.time_col or "period"
                    oot_periods = oot_period(
                        keys, time_col=time_col, n_oot_periods=config.n_oot_periods
                    )
                    oot_row_mask = keys[time_col].isin(oot_periods).to_numpy()
                    if oot_row_mask.any():
                        if name == "iforest":
                            from src.interpretability import explain_rows_iforest
                            with log_phase("interpretability.explain_rows_iforest"):
                                top_vars = explain_rows_iforest(
                                    detector, X_model[oot_row_mask], feature_names=names_model,
                                )
                        else:
                            from src.interpretability import explain_rows_vae
                            categorical_columns = df.select_dtypes(
                                include=["object", "category"]
                            ).columns.tolist()
                            with log_phase("interpretability.explain_rows_vae"):
                                top_vars = explain_rows_vae(
                                    detector, X_model[oot_row_mask], feature_names=names_model,
                                    categorical_columns=categorical_columns,
                                )
                        scored_df.loc[oot_row_mask, "top_5_variables"] = top_vars
                        dashboard_explanation_tables[name] = (
                            scored_df.loc[oot_row_mask]
                            .sort_values("anomaly_score", ascending=False)
                            .drop_duplicates(schema.entity_col or "entity_id", keep="first")
                            .reset_index(drop=True)
                        )
                        logger.info(
                            "[%s] per-row top-5 explanation computed for %d OOT row(s).",
                            name, int(oot_row_mask.sum()),
                        )
                except Exception as exc:  # noqa: BLE001 - never block the deliverable
                    logger.warning(
                        "[%s] per-row explanation failed (%s); the Excel export "
                        "will not carry a top_5_variables column.", name, exc,
                    )

                # `n_oot_periods=config.n_oot_periods` (NOT `n_test_periods`):
                # `export_oot_top_anomalies` derives its own OOT population as
                # the trailing `n_oot_periods` distinct periods of `scored_df`
                # (the whole panel). Passing the true OOT period count here is
                # what makes this deliverable draw from `oot_mask`'s rows --
                # strictly later than, and disjoint from, `test_mask`/`eval_mask`.
                out_path, _table = export_oot_top_anomalies(
                    scored_df,
                    schema,
                    # An explicit --top-n switches off the percentile cut, so
                    # the two selection modes can never both apply.
                    min_percentile=(None if config.top_n is not None
                                    else config.oot_min_percentile),
                    top_n=config.top_n,
                    top_fraction=config.top_fraction,
                    model_name=name,
                    n_oot_periods=config.n_oot_periods,
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

                # Headline "flagged for review" count for the report -- read
                # directly off the exported table's own `percentil` column
                # rather than recomputed independently, so the number on the
                # report always matches what a reviewer would get by opening
                # this exact file and counting P95+ rows themselves. `p95`
                # and `p99` are both >= the P95 cut-off (bands are cumulative
                # -- see `_percentile_band_labels`), so summing both bands is
                # the full P95+ count. Deliberately NOT the calibrated
                # `threshold`/`alert` count above: that can degenerate to
                # zero on real data with no floor, which is the exact bug
                # this replaces. Denominator is this model's own
                # `true_oot_entity_scores` (Phase 8, same OOT dedup rule).
                oot_review_count = int(_table[BAND_COL].isin(("p95", "p99")).sum())
                oot_review_rate = (
                    oot_review_count / len(true_oot_entity_scores)
                    if true_oot_entity_scores else None
                )
                logger.info(
                    "[%s] flagged for review (score >= P95, from %s): %d of %d "
                    "unique OOT individuals.",
                    name, out_path, oot_review_count, len(true_oot_entity_scores),
                )
                # Kept for the single, unified analyst dashboard built after
                # this loop -- see below. Not built per-model: the dashboard
                # shows both detectors' scores for the same individual side
                # by side, so it is one file, not one per deliverable.
                deliverable_tables[name] = _table
        else:
            logger.info(
                "[%s] no Excel deliverable: the VAE trains on the stacked matrix, "
                "so its queue already reflects this detector's score.", name,
            )

    # -- Phase 8c: reviewed event labels, gate 4.5, challengers ------------- #
    # data.csv carries no target; a separate reviewed-labels CSV is read HERE,
    # after both detectors are fitted and scored, so labels can never touch
    # training. Best-effort: no file, an unusable file or a red gate all leave
    # the unsupervised IF/VAE run exactly as it was. See
    # `src/evaluation/event_supervision.py` and CONTEXT.md.
    event_supervision_result = None
    if config.run_event_supervision:
        with log_phase("Phase 8c: event labels, gate 4.5 and challengers"):
            try:
                from src.evaluation.event_supervision import run_event_supervision

                event_supervision_result = run_event_supervision(
                    cfg=_event_supervision_config(config),
                    prepared=event_prepared,
                    schema=schema, keys=keys,
                    masks={"train": train_mask, "val": val_mask,
                           "test": test_mask, "oot": oot_mask},
                    scores={"if_score": if_scores, "vae_score": vae_scores},
                    X=X_if, feature_names=names_if,
                )
                _es = event_supervision_result
                if _es["status"] == "executed":
                    _gate = _es["gate"]
                    console_ui.set_stat("Compuerta de labels 4.5", _gate["level"])
                    console_ui.set_stat(
                        "Episodios positivos maduros",
                        f"{_gate['metrics']['n_mature_positive_episodes']:,} · "
                        f"{_gate['metrics']['n_positive_entities']:,} entidades",
                    )
                    observability.check(
                        name="artifact.event_supervision_written", category="artifact",
                        definition="The gate 4.5 acta, the event evaluation table and the "
                                   "challenger summary exist and are non-empty.",
                        expected="all three artifacts exist and size_bytes > 0",
                        severity="warning",
                        passed=all(os.path.isfile(p) and os.path.getsize(p) > 0
                                   for p in _es["artifacts"].values()),
                        observed={"gate": _gate["level"], "vetoes": [v["code"] for v in _gate["vetoes"]],
                                  "artifacts": _es["artifacts"]},
                        failure_action="Best-effort artifact; IF/VAE results and the OOT "
                                       "deliverables are unaffected.",
                        evidence=REPORTS_DIR,
                    )
                else:
                    logger.info("Phase 8c: %s", _es.get("reason"))
            except Exception as exc:  # noqa: BLE001 - never invalidate fitted models
                logger.warning(
                    "Event-label supervision failed (%s); IF/VAE results and the OOT "
                    "deliverables are unaffected.", exc,
                )

    # -- Analyst dashboard: ONE file, IF + VAE scores together -------------- #
    # Not one per deliverable model: the dashboard shows both detectors'
    # scores for the same individual side by side (an in-memory join on
    # `true_oot_entity_scores`, Phase 8 -- both detectors are always
    # evaluated regardless of `--stack-iforest-into-vae`). The queue is the
    # P95 union, split into only-IF, only-IF+VAE, and intersection tabs.
    with log_phase("Phase 9b: analyst dashboard"):
        try:
            from scipy.stats import rankdata

            from src.evaluation import (
                build_scored_frame,
                months_present_by_entity,
                oot_period,
            )
            from src.reporting import build_analyst_dashboard

            primary_name = config.deliverable_models[-1]
            if primary_name not in deliverable_tables:
                raise RuntimeError(
                    f"no OOT export found for the primary deliverable model {primary_name!r}"
                )

            def _entity_scores(model_name: str) -> dict:
                return chart_data["models"].get(model_name, {}).get(
                    "true_oot_entity_scores"
                ) or {}

            def _percentiles(entity_scores: dict) -> dict:
                if not entity_scores:
                    return {}
                ids = list(entity_scores.keys())
                vals = np.asarray([entity_scores[i] for i in ids], dtype=float)
                pct = 100.0 * rankdata(vals) / len(vals)
                return dict(zip(ids, pct))

            if_scores_by_entity = _entity_scores("iforest")
            vae_scores_by_entity = _entity_scores("vae")
            primary_scores_by_entity = (
                vae_scores_by_entity if primary_name == "vae" else if_scores_by_entity
            )
            if_percentiles = _percentiles(if_scores_by_entity)
            vae_percentiles = _percentiles(vae_scores_by_entity)

            time_col = schema.time_col or "period"
            oot_periods = oot_period(
                keys, time_col=time_col, n_oot_periods=config.n_oot_periods,
            )
            dash_score_col = "anomaly_score"  # build_scored_frame's default
            months_by_model = {}
            for dash_model_name, dash_scores in (
                ("iforest", if_scores_by_entity), ("vae", vae_scores_by_entity),
            ):
                dash_cutoff = (
                    float(np.percentile(list(dash_scores.values()), 95.0))
                    if dash_scores else float("nan")
                )
                dash_scored_df = build_scored_frame(
                    df, keys, models[dash_model_name][1], schema,
                )
                months_by_model[dash_model_name] = months_present_by_entity(
                    dash_scored_df, schema, oot_periods, cutoff=dash_cutoff,
                    score_col=dash_score_col,
                )
            dashboard_path = build_analyst_dashboard(
                deliverable_tables[primary_name], schema, primary_name, oot_periods,
                if_percentiles, vae_percentiles,
                if_scores_by_entity, vae_scores_by_entity, months_by_model[primary_name],
                n_total_oot=len(primary_scores_by_entity), score_col=dash_score_col,
                model_tables={
                    name: dashboard_explanation_tables.get(name, export_table)
                    for name, export_table in deliverable_tables.items()
                },
                months_present_by_model=months_by_model,
                entity_records=df,
                identity_columns=config.analyst_identity_columns,
            )
            analyst_dashboard_path = dashboard_path
            _dash_ok = os.path.isfile(dashboard_path) and os.path.getsize(dashboard_path) > 0
            observability.check(
                name="artifact.analyst_dashboard_written", category="artifact",
                definition="The unified analyst review-queue dashboard HTML exists "
                           "and is non-empty.",
                expected="file exists and size_bytes > 0", severity="warning",
                passed=_dash_ok,
                observed={"path": dashboard_path,
                          "size_bytes": os.path.getsize(dashboard_path) if _dash_ok else 0},
                failure_action="Best-effort artifact; the OOT Excel deliverable(s) above "
                               "are unaffected. Check the log for the dashboard-build error.",
                evidence=dashboard_path,
            )
        except Exception as exc:  # noqa: BLE001 - never block the Excel deliverable(s)
            logger.warning(
                "Analyst dashboard failed (%s); the OOT Excel deliverable(s) above "
                "are unaffected.", exc,
            )

    # -- Phase 9c: IF-VAE diagnostic suite (on by default) ------------------- #
    # Cross-validates this run's own already-fitted IF+VAE against the
    # vendored, optional diagnostic package -- reusing the live detectors and
    # matrices above for scoring (no refit, no CSV round-trip there). The
    # stability sub-check DOES refit both detectors a few times with
    # different seeds -- see `diagnostic_stability_refits`'s docstring above
    # for that cost. Label-free: this project's official runs carry no
    # target, so this is the only mode wired into the pipeline (see
    # CONTEXT.md "IF-VAE Diagnostic Suite integration"). Pass
    # `--no-run-diagnostic-suite` to skip entirely (e.g. package not
    # installed on this machine).
    if config.run_diagnostic_suite:
        with log_phase("Phase 9c: IF-VAE diagnostic suite"):
            try:
                from src.evaluation import run_ifvae_diagnostic_suite

                # Run metadata the chapter's contract reports as-is. Every
                # value is read from this run's own state -- the architecture
                # mode from the config that produced it, the derived-feature
                # list by differencing the VAE's feature space against the
                # pre-stacking one -- so nothing about the detectors is
                # asserted that the run did not actually do.
                _stacked = bool(config.stack_iforest_into_vae)
                _derived = [_stack_score_name] if config.stack_iforest_into_vae else []
                _run_meta = {
                    "run_id": ctx.run_id,
                    "generated_at": generated_at,
                    "architecture_mode": "Apilado" if _stacked else "Paralelo",
                    "detector_dependency": (
                        "El VAE recibe el puntaje del Isolation Forest como feature "
                        "de entrada; los detectores no son independientes."
                        if _stacked else
                        "Los detectores se ajustan por separado sobre la misma "
                        "matriz base; ninguno recibe el puntaje del otro."
                    ),
                    "derived_features": _derived,
                    "entity_aggregation_rule": (
                        "Máximo puntaje de la entidad dentro de la ventana OOT"
                        if config.diagnostic_entity_view else None
                    ),
                    "detectors": {
                        "iforest": {
                            "label": "Isolation Forest",
                            "model_id": os.path.basename(IFOREST_MODEL),
                            "score_origin": "Precalculado por el pipeline "
                                            "(IsolationForestDetector.score_samples)",
                            "score_column": "if_score",
                            "score_direction": "Mayor = más anómalo",
                        },
                        "vae": {
                            "label": "VAE",
                            "model_id": os.path.basename(VAE_MODEL),
                            "score_origin": "Reconstrucción y latentes calculados en "
                                            "proceso desde el detector ajustado",
                            "score_column": "recon__<feature>, mu__<i>, logvar__<i>",
                            "score_direction": "Mayor residual = más anómalo",
                        },
                    },
                }
                _segment_col = (config.diagnostic_segment_column or "").strip()
                if not _segment_col:
                    _segment = None
                elif _segment_col in df.columns:
                    _segment = df[_segment_col].to_numpy()
                else:
                    logger.warning(
                        "diagnostic_segment_column=%r no existe en el panel "
                        "(columnas disponibles: %s); la sección 8 quedará "
                        "NOT_APPLICABLE para esta corrida.",
                        _segment_col, ", ".join(df.columns[:20]),
                    )
                    _segment = None
                # Context for the section-9 experiment families (retrains, ablations,
                # rolling-origin backtests); built by the same function the tests use.
                from src.evaluation.ifvae_experiments import build_experiment_context

                _experiment_ctx = build_experiment_context(
                    df=df, schema=schema, keys=keys,
                    x_if_all=models["iforest"][3], if_feature_names=names_if,
                    x_vae_all=models["vae"][3], vae_feature_names=models["vae"][4],
                    if_detector=models["iforest"][0], vae_detector=models["vae"][0],
                    train_mask=train_mask, in_mask=in_mask, oot_mask=oot_mask,
                    valid_local=valid_local, n_val_periods=config.n_val_periods,
                    derived_features=_derived, fitted_preprocessor=fitted_preprocessor,
                    x_onehot_all=X, onehot_feature_names=feature_names,
                    vae_builder=vae_builder, mixed_config=vae_mixed_cfg,
                    stack_scaler=_stack_scaler if config.stack_iforest_into_vae else None,
                    prep_kwargs={
                        "numeric_transform": config.numeric_transform,
                        "categorical_encoding": config.categorical_encoding,
                        "rare_min_frequency": config.rare_min_frequency,
                        "impute_numeric": config.impute_numeric,
                        "add_panel_features": config.panel_features,
                        "add_missing_indicators": config.add_missing_indicators,
                        "random_state": config.seed,
                    },
                )
                _experiment_settings = {
                    "families": tuple(config.diagnostic_experiment_families),
                    "vae_fit_budget": config.diagnostic_experiment_vae_fit_budget,
                    "epoch_cap": config.diagnostic_experiment_epoch_cap,
                    "max_fit_rows": config.diagnostic_experiment_max_fit_rows,
                    "capacity_grid": config.diagnostic_experiment_capacity_grid,
                    "beta_grid": config.diagnostic_experiment_beta_grid,
                    "kl_anneal_grid": config.diagnostic_experiment_kl_grid,
                    "backtest_origins": config.diagnostic_backtest_origins,
                    "backtest_vae_origins": config.diagnostic_backtest_vae_origins,
                    "backtest_min_fit_periods": config.diagnostic_backtest_min_fit_periods,
                    "seed": config.seed,
                }
                diagnostic_suite_result = run_ifvae_diagnostic_suite(
                    keys, schema,
                    models["iforest"][0], models["iforest"][1], models["iforest"][3],
                    in_mask, valid_local,
                    models["vae"][0], models["vae"][3], models["vae"][4],
                    train_mask, oot_mask,
                    out_dir=os.path.join(REPORTS_DIR, "ifvae_diagnostics"),
                    run_meta=_run_meta,
                    sensitivity_grid=config.diagnostic_sensitivity_grid,
                    entity_view=config.diagnostic_entity_view,
                    stability_refits=config.diagnostic_stability_refits,
                    base_seed=config.seed,
                    segment=_segment,
                    segment_name=_segment_col or None,
                    auto_install_suite=config.diagnostic_auto_install_suite,
                    experiment_contamination_grid=config.diagnostic_experiment_contamination_grid,
                    experiment_capacity_grid=config.diagnostic_experiment_capacity_grid,
                    experiment_beta_grid=config.diagnostic_experiment_beta_grid,
                    experiment_ctx=_experiment_ctx,
                    experiment_settings=_experiment_settings,
                )
                _n_files = sum(
                    1 for name_ in (
                        "report.md", "disagreement.png", "summary.json",
                        "drift.csv", "scored_diagnostics.csv", "warnings.json",
                    )
                    if os.path.isfile(os.path.join(diagnostic_suite_result["report_dir"], name_))
                )
                observability.check(
                    name="artifact.diagnostic_suite_written", category="artifact",
                    definition="The IF-VAE Diagnostic Suite's report and core "
                               "artifact files exist in its output directory.",
                    expected="report.md, disagreement.png and 4 other core files exist",
                    severity="warning", passed=_n_files == 6,
                    observed={"report_dir": diagnostic_suite_result["report_dir"],
                              "files_found": _n_files,
                              "rows_scored": diagnostic_suite_result["rows_scored"]},
                    failure_action="Best-effort artifact; the OOT Excel deliverable(s) "
                                   "and report above are unaffected. Check the log for "
                                   "the diagnostic-suite error.",
                    evidence=diagnostic_suite_result["report_dir"],
                )
                logger.info(
                    "IF-VAE diagnostic suite: %d OOT rows scored against %d reference "
                    "rows, quadrants=%s -> %s",
                    diagnostic_suite_result["rows_scored"],
                    diagnostic_suite_result["rows_reference"],
                    diagnostic_suite_result["quadrants"],
                    diagnostic_suite_result["report_dir"],
                )
                # Per-seed breakdown (§7's aggregate mean/min are computed over exactly
                # this matrix -- see `_seeded_refit_stability`) for the report's seed x
                # seed Jaccard heatmap: visual stability, not only one summary number.
                _stability = diagnostic_suite_result.get("stability") or {}
                chart_static["stability_seeds"] = {
                    key: {
                        "seeds": block.get("seeds"),
                        "pairwise_jaccard": block.get("pairwise_jaccard"),
                        "per_seed_mean_jaccard": block.get("per_seed_mean_jaccard"),
                    }
                    for key, block in _stability.items()
                    if block.get("status") == "EXECUTED" and block.get("pairwise_jaccard")
                }
            except Exception as exc:  # noqa: BLE001 - never block the report above
                logger.warning(
                    "IF-VAE diagnostic suite failed (%s); the report chapter for it "
                    "will be omitted. Review the auto-install result and vendored "
                    "suite path in execution.log.", exc,
                )

    # -- Phase 9d: post-training sensitivity and data-quality stress test --- #
    # Reuses the fitted preprocessor and fitted detectors. Labels are allowed
    # here even for an unsupervised training run because they are consumed only
    # after fitting to assess hit/miss degradation; they never influence a
    # scenario, transformation, threshold or score.
    if config.run_sensitivity_analysis:
        with log_phase("Phase 9d: post-training sensitivity analysis"):
            try:
                from src.evaluation import run_post_training_sensitivity

                sensitivity_result = run_post_training_sensitivity(
                    df=df,
                    schema=schema,
                    preprocessor=fitted_preprocessor,
                    feature_names=feature_names,
                    models={name: spec[0] for name, spec in models.items()},
                    baseline_scores={name: spec[1] for name, spec in models.items()},
                    thresholds={
                        name: model_specs[name]["threshold"]["threshold"]
                        for name in models
                    },
                    train_mask=train_mask,
                    test_mask=test_mask,
                    labels=(labels if n_pos > 0 else None),
                    stack_context=(
                        {
                            **({"detector": stack_detector, "scaler": _stack_scaler}
                               if config.stack_iforest_into_vae else {}),
                            # Mixed VAE view: perturbed frames are re-encoded with the SAME vocabularies
                            # (nulls -> MISSING, unseen levels -> UNKNOWN); only the score column is scaled.
                            **({"vae_view": vae_builder, "score_only_scaler": True}
                               if vae_builder is not None else {}),
                        } or None
                    ),
                    max_test_rows=config.sensitivity_max_test_rows,
                    combination_top_k=config.sensitivity_combination_top_k,
                    random_subsets_per_level=config.sensitivity_random_subsets_per_level,
                    missing_levels=config.sensitivity_missing_levels,
                    high_zero_cutoff=config.sensitivity_high_zero_cutoff,
                    random_state=config.seed,
                    out_dir=REPORTS_DIR,
                )
                _sens_artifacts = sensitivity_result.get("artifacts", {})
                _sens_ok = all(
                    path and os.path.isfile(path) and os.path.getsize(path) > 0
                    for path in _sens_artifacts.values()
                )
                observability.check(
                    name="artifact.sensitivity_analysis_written", category="artifact",
                    definition="Post-training sensitivity workbook, tables, summary and "
                               "interactive report exist and are non-empty.",
                    expected="all sensitivity artifacts exist and size_bytes > 0",
                    severity="warning", passed=_sens_ok,
                    observed={"artifacts": _sens_artifacts,
                              "required_variables": sensitivity_result.get("required_variables")},
                    failure_action="Review Phase 9d logs; fitted models and OOT deliverables "
                                   "remain valid but robustness conclusions are unavailable.",
                    evidence=REPORTS_DIR,
                )
            except Exception as exc:  # noqa: BLE001 - never invalidate fitted models
                logger.warning(
                    "Post-training sensitivity analysis failed (%s); fitted models, "
                    "OOT deliverables and the analyst dashboard remain available.", exc,
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
                    split_count_analysis,
                )

                try:
                    imp = shap_summary_iforest(detector, X_model, feature_names=names_model)
                    _add_fig(
                        figures, "iForest SHAP summary",
                        os.path.join(FIGURES_DIR, "iforest_shap_summary.png"),
                    )
                    # Also feeds the report's interactive version of this chart
                    # (both capped at the top 20 there) and the uncropped
                    # per-variable Excel workbook below.
                    chart_static["shap_importance"] = imp
                    model_attributions["iforest"] = imp
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
                try:
                    # Which features the OOT ALERT QUEUE actually gets isolated by, and
                    # with how much help: the calibrated threshold on the OOT window is
                    # this run's own definition of "flagged", so that -- not every row --
                    # is what "cuántos cortes para detectar outliers" means here.
                    threshold_if = model_specs["iforest"]["threshold"]["threshold"]
                    flagged_mask = _flagged_rows_for_split_count(
                        oot_mask, scores, threshold_if, logger=logger,
                    )
                    split_result = split_count_analysis(
                        detector, X_model, row_mask=flagged_mask, feature_names=names_model,
                    )
                    _add_fig(figures, "iForest split-count analysis",
                            split_result.get("figure_path"))
                    # A feature absent from `per_feature` never helped isolate any
                    # flagged row this run -- the strongest signal this analysis can
                    # give for "safe to consider dropping" (see the report's
                    # "Variables que menos aportan..." section). Only meaningful when
                    # the analysis actually found rows/features to walk: an empty
                    # `per_feature` from zero analysed rows would otherwise look like
                    # every feature is unused, which is a data-availability artefact,
                    # not a signal about the features themselves.
                    per_feature = split_result.get("per_feature") or {}
                    analysis_ran = bool(split_result.get("top_clear") or split_result.get("top_noisy"))
                    chart_static["iforest_splits"] = {
                        "top_clear": split_result.get("top_clear"),
                        "top_noisy": split_result.get("top_noisy"),
                        "n_rows_analyzed": split_result.get("n_rows_analyzed"),
                        "n_trees": split_result.get("n_trees"),
                        "mean_path_length": split_result.get("mean_path_length"),
                        "n_features_total": len(names_model),
                        "unused_features": (
                            sorted(set(names_model) - set(per_feature.keys()))
                            if analysis_ran else []
                        ),
                    }
                except Exception as exc:
                    logger.warning("split_count_analysis failed (%s); continuing.", exc)
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
                    # Original (pre-transform) categorical column names, so
                    # the chart/diagnostic can sum one-hot-derived columns
                    # back under their source variable instead of letting a
                    # high-cardinality categorical dominate the ranking by
                    # column count alone -- see CONTEXT.md "VAE feature
                    # attribution: categorical granularity".
                    categorical_columns = df.select_dtypes(
                        include=["object", "category"]
                    ).columns.tolist()
                    recon = reconstruction_error_by_feature(
                        detector, X_model, feature_names=names_model,
                        categorical_columns=categorical_columns,
                    )
                    _add_fig(
                        figures, "VAE per-feature reconstruction error",
                        os.path.join(FIGURES_DIR, "vae_recon_by_feature.png"),
                    )
                    chart_static["recon_by_feature"] = recon
                    chart_static["recon_by_feature_kind"] = (
                        "contribution" if getattr(detector, "layout", None) is not None else "squared_error")
                    model_attributions["vae"] = recon
                except Exception as exc:
                    logger.warning("reconstruction_error_by_feature failed (%s); continuing.", exc)

    # -- Phase 10b: full per-feature attribution workbook ------------------- #
    # Outside the per-model loop: this needs both models' dicts together (one
    # sheet each), not one at a time.
    attribution_path = None
    if model_attributions:
        with log_phase("Phase 10b: attribution workbook"):
            from src.interpretability import export_attribution_workbook

            try:
                categorical_columns = df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()
                attribution_path = export_attribution_workbook(
                    model_attributions, categorical_columns=categorical_columns,
                )
                observability.check(
                    name="artifact.attribution_workbook_written", category="artifact",
                    definition="The full per-feature SHAP/reconstruction-error "
                               "workbook exists and is non-empty.",
                    expected="attribution_path is not None", severity="warning",
                    passed=attribution_path is not None,
                    observed={"path": attribution_path,
                              "models": sorted(model_attributions)},
                    failure_action="Best-effort artifact; the report and figures "
                                   "are unaffected. Check the log for which "
                                   "model's attribution failed upstream.",
                    evidence=paths.REPORTS_DIR,
                )
            except Exception as exc:
                logger.warning("export_attribution_workbook failed (%s); continuing.", exc)

    # -- Phase 11: report --------------------------------------------------- #
    report_paths: dict = {"html": None, "md": None, "model_doc": None}
    with log_phase("Phase 11: report"):
        from src.reporting import build_report

        oot_note = "; ".join(f"{k}: {v}" for k, v in oot_excels.items())
        # The report's hero figure ("anomalías marcadas para revisión") is a
        # review-queue count, not a ground-truth count: `n_pos`/`anomaly_rate`
        # (Phase 3) answer "how many rows are truly anomalous," which is
        # unknown on real, unlabeled data and is why this hero showed 0 on a
        # real run with no ground-truth file. `oot_review_count` (Phase 9,
        # read directly off the deliverable's own P95/P99 band) is what a
        # reviewer would actually work from, so it takes priority; falling
        # back to the ground-truth count only if no deliverable ran at all.
        hero_n_anomalies = oot_review_count if oot_review_count is not None else n_pos
        hero_anomaly_rate = oot_review_rate if oot_review_count is not None else anomaly_rate
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
                "anomaly_rate": hero_anomaly_rate,
                "n_anomalies": hero_n_anomalies,
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
            "diagnostic_suite": diagnostic_suite_result,
            "sensitivity_analysis": sensitivity_result,
            "event_supervision": event_supervision_result,
            # Quick-glance mirror of ERROR/CRITICAL lines logged so far this
            # run. Routine warnings are intentionally excluded from reports.
            # `execution.log` is always the complete, authoritative record;
            # this is additive, never a replacement.
            "incidents": list(incident_collector.records),
            "row_filter": zero_filter_stats,
            "preprocessing": {
                "numeric_transform": config.numeric_transform,
                "categorical_encoding": config.categorical_encoding,
                "panel_features": config.panel_features,
                "tuning": config.tune,
                "threshold_method": config.threshold_method,
                "threshold_percentile": config.threshold_percentile,
                "threshold_target_far": config.threshold_target_far,
                "post_training_sensitivity": config.run_sensitivity_analysis,
            },
            "notes": (
                f"Entregables Excel "
                f"{'top-' + str(config.top_n) if config.top_n is not None else 'P' + str(int(config.oot_min_percentile))} "
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
    # Detach the incident collector now that the report has already read it
    # (Phase 11, above) -- nothing later needs it, and leaving it attached
    # would let a second in-process `run_pipeline` call inherit this run's
    # records.
    logger.removeHandler(incident_collector)
    # Tear the dashboard down *before* printing: the summary is the one thing
    # that must survive in the scrollback, and a Live display owns the bottom
    # of the terminal until it is stopped.
    console_ui.stop_dashboard()

    artifacts = {
        "oot_excels": oot_excels,
        "analyst_dashboard": analyst_dashboard_path,
        "ifvae_diagnostic_report": (
            diagnostic_suite_result["report_md_path"] if diagnostic_suite_result else None
        ),
        "sensitivity_analysis": (
            sensitivity_result.get("artifacts", {}) if sensitivity_result else {}
        ),
        "event_supervision": (
            event_supervision_result.get("artifacts", {})
            if event_supervision_result and event_supervision_result.get("status") == "executed" else {}
        ),
        "p95_checkpoint": p95_path,
        "attribution_workbook": attribution_path,
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
        f"  Analyst dashboard: {analyst_dashboard_path or '(none)'}",
        f"  IF-VAE diagnostic suite: {artifacts['ifvae_diagnostic_report'] or '(not run)'}",
        f"  Sensitivity analysis: {artifacts['sensitivity_analysis'].get('html', '(not run)')}",
        f"  Event-label gate 4.5: {(artifacts['event_supervision'] or {}).get('label_gate', '(not run / no labels file)')}",
        f"  Feature attribution (xlsx): {attribution_path}",
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
    "n_individuals": 2000, "n_periods": 16,
    "iforest_trials": 15, "vae_trials": 10, "vae_epochs": 15, "tune": True,
}
# --quick still needs `n_periods >= n_oot_periods + 3` for the chronological
# split (train/val/test/OOT); with the default `n_oot_periods=3`, 13 periods
# gives a 5/2/3/3 split (train/val/test/OOT) -- short enough that
# PanelFeatureEngineer automatically drops the h=6 contrast horizon (needs a
# deeper training block), which is expected for a quick smoke test, not a bug.
_QUICK = {
    "n_individuals": 500, "n_periods": 13,
    "iforest_trials": 5, "vae_trials": 5, "vae_epochs": 5,
}
_FULL = {
    "n_individuals": 100_000, "n_periods": 16,
    "iforest_trials": 50, "vae_trials": 30, "vae_epochs": 15,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,   # an abbreviated flag would not count as explicit for the config file
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
                        help="Number of monthly periods (default 16).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for reproducibility (default 42).")
    parser.add_argument("--numeric-transform", default="yeo-johnson",
                        help="Numeric transform for preprocessing (default 'yeo-johnson').")
    parser.add_argument("--categorical-encoding", default="onehot",
                        help="Categorical encoding for preprocessing (default 'onehot').")
    parser.add_argument("--rare-min-frequency", type=float, default=0.001,
                        help="Categories below this fraction of rows collapse into one "
                             "'infrequent' bucket before encoding (default 0.001 = 0.1%%). "
                             "Raise this if a high-cardinality categorical is inflating the "
                             "VAE's per-feature attribution/score via one-hot column count "
                             "-- see the 'vae_by_source' sheet in feature_attribution.xlsx "
                             "to check first.")
    parser.add_argument("--no-zero-impute", action="store_true",
                        help="Fill numeric NaNs with the column MEDIAN instead of 0.0. "
                             "Zero-fill is the default because it estimates nothing "
                             "from the data (so it cannot leak across the train/test "
                             "split); use this when a filled 0 would be confused with "
                             "a real 0 and the missingness indicators are not enough.")
    parser.add_argument("--add-missing-indicators", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Add a 0/1 `missing__<col>` feature for every numeric column "
                             "that had any NaN in the training fit (default ON -- pairs with "
                             "zero-fill so 'was absent' stays distinguishable from a real "
                             "0). Only worth it when missingness itself is informative in "
                             "your data. --no-add-missing-indicators turns it off, so no "
                             "`missing__*` column is ever created or shown.")
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
    parser.add_argument("--run-diagnostic-suite", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cross-validate this run's IF+VAE against the vendored, "
                             "optional IF-VAE Diagnostic Suite (tools/if_vae_diagnostic_suite; "
                             "auto-installed from that local path when needed) and add the "
                             "'Diagnóstico cruzado IF-VAE' and "
                             "'Interpretación y recomendaciones' report chapters (default ON "
                             "as of 2026-09-05). Runs label-free -- see CONTEXT.md "
                             "\"IF-VAE Diagnostic Suite integration\". "
                             "--no-run-diagnostic-suite skips the whole phase.")
    parser.add_argument("--diagnostic-sensitivity-grid", type=float, nargs="*",
                        default=None, metavar="P",
                        help="Percentile thresholds (each in (0,1)) the diagnostic "
                             "chapter's sensitivity section sweeps (default 0.90 0.95 0.99, "
                             "this project's own P90/P95/P99 operating points). Pass with no "
                             "values to disable that section (NOT_REQUESTED).")
    parser.add_argument("--diagnostic-entity-view", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Add the entity-aggregated view to the diagnostic chapter's "
                             "scope section (default ON; the aggregation rule is the same "
                             "max-per-entity rule already used elsewhere in this pipeline).")
    parser.add_argument("--diagnostic-stability-refits", type=int, default=None,
                        help="Independent seed refits used to measure IF/VAE alert-set "
                             "stability (default 3). VAE refits are full training runs -- "
                             "this is the most expensive part of the diagnostic chapter. "
                             "Pass 0 to disable (reports UNAVAILABLE with a stated reason).")
    parser.add_argument("--diagnostic-segment-column", type=str, default=None,
                        help="Column in the raw panel used for the diagnostic chapter's "
                             "per-segment breakdown (default 'segment', falling back to "
                             "'despuestocolaboradoragrupado' if 'segment' is absent -- only "
                             "for that built-in default, see _SEGMENT_NAME_FALLBACKS). "
                             "Point this at any other categorical column your real panel "
                             "carries, e.g. --diagnostic-segment-column region -- an "
                             "explicit value here never falls back, it must exist as given. "
                             "Pass an empty string (--diagnostic-segment-column '') to "
                             "disable the breakdown.")
    parser.add_argument("--auto-install-suite", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Auto-install the vendored IF-VAE Diagnostic Suite "
                             "(tools/if_vae_diagnostic_suite) into this environment if it "
                             "is not already importable (default ON; always installs from "
                             "this repo's own vendored copy, never from an index). "
                             "--no-auto-install-suite requires it pre-installed instead.")
    parser.add_argument("--skip-dependency-check", action="store_true", default=False,
                        help="Skip the startup check that every package in requirements.txt "
                             "is installed at or above its minimum version (default: check "
                             "runs). Use on a pre-vetted environment (a pinned Docker image, "
                             "an air-gapped machine without pip index access) where the check "
                             "itself is unnecessary or cannot succeed.")
    parser.add_argument("--auto-install-deps", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="If the dependency check above finds a missing or outdated "
                             "package, install it automatically (`pip install --upgrade`) "
                             "(default ON -- a missing/outdated dependency is fixed directly "
                             "instead of stopping the run). --no-auto-install-deps switches to "
                             "check-only: log the problem and the exact `pip install` command, "
                             "then stop, without modifying the active Python environment.")
    parser.add_argument("--diagnostic-experiment-contamination-grid", type=float, nargs="*",
                        default=None, metavar="C",
                        help="Isolation Forest operating points (top-c%% of the score) compared "
                             "in the diagnostic chapter's §9 'Sensibilidad del punto de "
                             "operación' experiment (default 0.02 0.01 0.005). Score cut-offs "
                             "only: contamination does not change the forest, so nothing is "
                             "refitted. Pass with no values to disable.")
    parser.add_argument("--diagnostic-experiment-capacity-grid", type=int, nargs="*",
                        default=None, metavar="DIM",
                        help="VAE latent_dim values for §9 'Capacidad y dimensión latente'. "
                             "Default: automatic points around the production width (½x, 2x, "
                             "4x latent_dim and ½x/2x hidden width). Each point is a VAE "
                             "retrain drawn from the shared budget. Pass with no values to "
                             "switch the sweep off.")
    parser.add_argument("--diagnostic-experiment-beta-grid", type=float, nargs="*",
                        default=None, metavar="BETA",
                        help="VAE beta values for §9 'Beta y programación KL'. Default: "
                             "automatic (0.25x and 4x the production beta, plus kl_anneal_epochs "
                             "0 and =epochs). Pass with no values to skip the beta points.")
    parser.add_argument("--diagnostic-experiment-vae-fit-budget", type=int, default=None,
                        metavar="N",
                        help="Total VAE retrains the §9 experiments may use (default 16: 1 noise control + 1 one-hot control + "
                             "5 + 4 + 3 + 2; 0 = none, the VAE-based points become NOT_REQUESTED).")
    parser.add_argument("--diagnostic-experiment-families", type=str, nargs="*", default=None,
                        metavar="FAMILY",
                        help="§9 families to run (default all): capacity beta_kl loss_by_type "
                             "ablation backtest window_stability.")
    parser.add_argument("--run-sensitivity-analysis", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Run the post-training variable/data-quality sensitivity study "
                             "without refitting either detector (default ON).")
    parser.add_argument("--sensitivity-max-test-rows", type=int, default=5000,
                        help="Maximum test records sampled for perturbation scoring "
                             "(default 5000; full entity histories are retained).")
    parser.add_argument("--sensitivity-combination-top-k", type=int, default=6,
                        help="Highest-leverage variables used for pairwise-removal scenarios "
                             "(default 6).")
    parser.add_argument("--sensitivity-random-subsets", type=int, default=5,
                        help="Random variable subsets per missing-information level for the "
                             "fan chart (default 5).")
    parser.add_argument("--sensitivity-missing-levels", type=float, nargs="*", default=None,
                        metavar="P",
                        help="Fractions of inputs removed/zeroed in data-quality stress tests "
                             "(default 0.10 0.25 0.50 0.75 0.90).")
    parser.add_argument("--sensitivity-high-zero-cutoff", type=float, default=0.90,
                        help="Record-level share of zero-or-missing inputs used for the "
                             "exclusion analysis (default 0.90).")
    parser.add_argument("--exact-zero-row-cutoff", type=float, default=0.90,
                        help="Record-level share of input columns that must be an exact 0 "
                             "for the row to be dropped from the panel before the split, "
                             "fitting, and every downstream artifact (default 0.90).")
    parser.add_argument("--run-event-supervision", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Phase 8c: evaluate IF/VAE against reviewed event labels, decide "
                             "gate 4.5 and (if authorised) run supervised challengers. Does "
                             "nothing when no labels file is found (default ON).")
    parser.add_argument("--labels-path", type=str, default=None,
                        help="CSV/parquet with at least entity_id, codmes and target (plus "
                             "optional label_status, episode_id, maturity_date). Default: the "
                             "newest file under data/reviewed_labels/.")
    parser.add_argument("--labels-dir", type=str, default=None,
                        help="Folder searched when --labels-path is not given "
                             "(default data/reviewed_labels/).")
    parser.add_argument("--labels-entity-column", type=str, default=None,
                        help="Entity column in the labels file (default: auto-detect entity_id/...).")
    parser.add_argument("--labels-period-column", type=str, default=None,
                        help="Month column in the labels file (default: auto-detect codmes/period/...).")
    parser.add_argument("--labels-target-column", type=str, default=None,
                        help="Target column in the labels file (default: auto-detect target/label/...).")
    parser.add_argument("--labels-status-column", type=str, default=None,
                        help="Status column (default: auto-detect label_status).")
    parser.add_argument("--labels-usable-statuses", type=str, nargs="+",
                        default=None,
                        help="label_status values that count as a usable decision "
                             "(default: confirmed adjudicated). pending/uncertain/conflict/"
                             "superseded/withdrawn are never negatives.")
    parser.add_argument("--labels-unlisted-as-negative", action="store_true",
                        help="Treat panel rows the labels file does not mention as confirmed "
                             "negatives. Only for an exhaustive base; vetoed by the gate unless "
                             "--labels-audit-attested.")
    parser.add_argument("--labels-audit-attested", action="store_true",
                        help="Attest that unlisted rows were audited as negatives.")
    parser.add_argument("--labels-formal-calc-ok", action="store_true",
                        help="Attest that the formal sample-size calculation (Riley/pmsampsize) "
                             "is satisfied; required for the green gate level.")
    parser.add_argument("--label-horizon-months", type=int, default=1,
                        help="Months a label needs to be mature (default 1).")
    parser.add_argument("--label-confirm-delay-months", type=int, default=0,
                        help="Operational delay to confirm a label, in months (default 0).")
    parser.add_argument("--label-maturity-buffer-months", type=int, default=0,
                        help="Extra maturity buffer, in months (default 0).")
    parser.add_argument("--label-washout-months", type=int, default=1,
                        help="Non-positive months that close an episode (default 1: only "
                             "consecutive positive months collapse into one episode).")
    parser.add_argument("--labels-as-of", type=str, default=None,
                        help="Point-in-time cut-off (YYYYMM or YYYY-MM-DD) for maturity and "
                             "label availability (default: last panel month).")
    parser.add_argument("--review-capacity-k", type=int, default=None,
                        help="Rows the business can review per evaluation window (K for "
                             "precision@K/recall@K). Default: 5%% of eligible rows (an assumption).")
    parser.add_argument("--hazard-horizon-months", type=int, default=3,
                        help="Horizon H of the discrete-hazard challenger: 'a new episode "
                             "starts within H months' (default 3).")
    parser.add_argument("--tune-with-labels", choices=("auto", "off"), default="auto",
                        help="IF tuning objective: 'auto' uses average precision against the "
                             "reviewed labels of the validation months when gate 4.5 authorises "
                             "them (and there are enough positives), otherwise the label-free "
                             "tail_separation; 'off' never uses labels for model selection.")
    parser.add_argument("--tune-min-positive-rows", type=int, default=10,
                        help="Minimum mature positive rows in the validation months for the "
                             "labelled tuning objective (default 10).")
    parser.add_argument("--iforest-max-samples-range", type=int, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Absolute rows-per-tree range searched by the IF tuner "
                             "(default 1024 32768, capped to the rows available).")
    parser.add_argument("--iforest-selection-top-k", type=int, default=None, metavar="K",
                        help="How many of the completed IF tuning trials get replicated "
                             "with --iforest-noise-seeds seeds each before deploying one "
                             "(default 5; bounds the post-tuning selection phase's cost "
                             "independent of --iforest-trials). 1 keeps the "
                             "margin-vs-untuned-default check but drops the "
                             "prefer-the-cheaper-near-tied-trial refinement. 0 skips the "
                             "whole phase and deploys the tuner's single-seed winner "
                             "directly -- no noise-floor or margin check at all. See "
                             "PipelineConfig.iforest_selection_top_k for what each level "
                             "guarantees; lower it on a large real panel where the "
                             "wall-clock cost of this phase matters.")
    parser.add_argument("--iforest-noise-seeds", type=int, default=None, metavar="N",
                        help="Seeds used to replicate the IF tuning selection (default 3, "
                             "minimum 2). Multiplies the selection phase's cost together "
                             "with --iforest-selection-top-k.")
    parser.add_argument("--event-challengers", choices=("off", "auto", "force"), default="auto",
                        help="Supervised challengers: off; auto = only if gate 4.5 authorises "
                             "them (default); force = exploratory run despite a red gate.")
    parser.add_argument("--event-bootstrap-reps", type=int, default=100,
                        help="Entity-cluster bootstrap replicates for the intervals (0 = off).")
    parser.add_argument("--contamination", type=float, default=None,
                        help="Isolation Forest operating-point contamination, used by both "
                             "the tuned and untuned paths (default 0.02; must be in (0, 0.5]). "
                             "Does not affect score_samples()/ranking, only predict()/"
                             "decision_function() and the P95 checkpoint export's downstream "
                             "consumers -- see docs/models_isolation_forest.md §2b.")
    parser.add_argument("--p95-percentile", type=float, default=None,
                        help="Percentile of in-time IF scores above which a row is exported "
                             "in the Phase 6c checkpoint gate before the VAE runs (default 95).")
    parser.add_argument("--oot-min-percentile", type=float, default=90.0,
                        help="Percentile of the OOT score distribution above which an "
                             "individual is exported in the risk-ranked Excel (default "
                             "90). Each row is also graded p90/p95/p99. Overridden by "
                             "--top-n when that is set.")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Export a FIXED headcount instead of the percentile cut "
                             "(default: unset, percentile mode). Pass 0 or a negative "
                             "value to force percentile mode explicitly.")
    parser.add_argument("--n-val-periods", type=int, default=2,
                        help="Validation months for tuning + threshold calibration (default 2).")
    parser.add_argument("--n-test-periods", type=int, default=3,
                        help="Trailing test months, reported but never tuned on (default 3).")
    parser.add_argument("--n-oot-periods", type=int, default=3,
                        help="Trailing months reserved AFTER test, exclusively for the OOT "
                             "Excel deliverable -- never used for fitting, tuning, threshold "
                             "calibration, or test-set metrics (default 3: the last 3 months).")
    parser.add_argument("--analyst-identity-columns", type=str, nargs="*", default=None,
                        metavar="COLUMN",
                        help="Source column(s) shown below the entity ID in the analyst profile, "
                             "one row per field in the order given (default: 'puesto', or "
                             "dashboard.identity_columns of configs/pipeline.yaml). Missing values "
                             "are shown as unavailable. Folded into --identification-columns "
                             "automatically: none of them is ever a model feature either.")
    parser.add_argument("--identification-columns", type=str, nargs="*", default=None,
                        metavar="COLUMN",
                        help="Panel columns that identify/describe a record for a human reader "
                             "(job title, name, area, an internal reference number, a segment/"
                             "grouping column used elsewhere in the report, ...) and must never be "
                             "a model feature, in ANY phase (default: none besides "
                             "--analyst-identity-columns, which is folded in automatically). This "
                             "list is a SUPERSET of --analyst-identity-columns: a column named only "
                             "here is excluded from modelling but NOT shown in the dashboard card. "
                             "Columns named here ARE still shown wherever identification is the "
                             "point regardless: the OOT Excel export, the raw data profile. "
                             "Default: data.identification_columns of configs/pipeline.yaml.")
    parser.add_argument("--vae-categorical-representation", choices=("onehot", "embedding"), default=None,
                        help="How the VAE receives categorical variables: 'embedding' (one index per "
                             "variable, learned embeddings, one contribution per variable) or 'onehot' "
                             "(original MLP over dummy columns; migration/control). Default: vae."
                             "categorical_representation of configs/pipeline.yaml.")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="Single configuration file for the diagnostic knobs (segment "
                             "column, analyst identity columns, experiment grids). Default: "
                             "configs/pipeline.yaml next to main.py if it exists. Precedence: "
                             "CLI flag > value set in code > file > built-in default.")
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


#: argparse dest -> the file-managed PipelineConfig attribute it sets.
_FILE_MANAGED_DESTS = {
    "run_diagnostic_suite": "run_diagnostic_suite",
    "auto_install_suite": "diagnostic_auto_install_suite",
    "diagnostic_segment_column": "diagnostic_segment_column",
    "diagnostic_entity_view": "diagnostic_entity_view",
    "diagnostic_stability_refits": "diagnostic_stability_refits",
    "diagnostic_sensitivity_grid": "diagnostic_sensitivity_grid",
    "analyst_identity_columns": "analyst_identity_columns",
    "identification_columns": "identification_columns",
    "diagnostic_experiment_contamination_grid": "diagnostic_experiment_contamination_grid",
    "diagnostic_experiment_capacity_grid": "diagnostic_experiment_capacity_grid",
    "diagnostic_experiment_beta_grid": "diagnostic_experiment_beta_grid",
    "diagnostic_experiment_vae_fit_budget": "diagnostic_experiment_vae_fit_budget",
    "diagnostic_experiment_families": "diagnostic_experiment_families",
    "vae_categorical_representation": "vae_categorical_representation",
    "iforest_selection_top_k": "iforest_selection_top_k",
    "iforest_noise_seeds": "iforest_noise_seeds",
}


def _cli_explicit_fields(args: argparse.Namespace) -> set:
    """File-managed attributes the user gave on the command line (so the file never wins).

    ``main()`` records the dests actually present in argv; without that (programmatic
    callers, tests) a value different from the parser's default counts as explicit.
    """
    seen = getattr(args, "_explicit_dests", None)
    if seen is None:
        parser = build_arg_parser()
        seen = {d for d in _FILE_MANAGED_DESTS
                if getattr(args, d, None) != parser.get_default(d)}
    return {_FILE_MANAGED_DESTS[d] for d in seen if d in _FILE_MANAGED_DESTS}


def _explicit_dests(parser: argparse.ArgumentParser, argv: Optional[list]) -> set:
    tokens = list(sys.argv[1:] if argv is None else argv)
    out = set()
    for action in parser._actions:  # noqa: SLF001 - stable argparse internals
        for opt in action.option_strings:
            if any(tok == opt or tok.startswith(opt + "=") for tok in tokens):
                out.add(action.dest)
    return out


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
        rare_min_frequency=args.rare_min_frequency,
        impute_numeric=("median" if args.no_zero_impute else "zero"),
        add_missing_indicators=args.add_missing_indicators,
        supervised=args.supervised,
        live_view=args.live_view,
        console_ui=args.console_ui,
        panel_features=args.panel_features,
        tune=resolve("tune"),
        iforest_trials=resolve("iforest_trials"),
        vae_trials=resolve("vae_trials"),
        vae_epochs=resolve("vae_epochs"),
        # --top-n 0 (or negative) is the explicit "use percentile mode" escape
        # hatch; unset (None) already means percentile mode by default.
        top_n=(None if args.top_n is not None and args.top_n <= 0 else args.top_n),
        oot_min_percentile=args.oot_min_percentile,
        top_fraction=args.top_fraction,
        n_val_periods=args.n_val_periods,
        n_test_periods=args.n_test_periods,
        n_oot_periods=args.n_oot_periods,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_target_far=args.threshold_target_far,
        stack_iforest_into_vae=args.stack_iforest_into_vae,
        run_diagnostic_suite=args.run_diagnostic_suite,
        diagnostic_entity_view=args.diagnostic_entity_view,
        diagnostic_auto_install_suite=args.auto_install_suite,
        skip_dependency_check=args.skip_dependency_check,
        auto_install_deps=args.auto_install_deps,
        run_sensitivity_analysis=args.run_sensitivity_analysis,
        sensitivity_max_test_rows=args.sensitivity_max_test_rows,
        sensitivity_combination_top_k=args.sensitivity_combination_top_k,
        sensitivity_random_subsets_per_level=args.sensitivity_random_subsets,
        sensitivity_high_zero_cutoff=args.sensitivity_high_zero_cutoff,
        exact_zero_row_cutoff=args.exact_zero_row_cutoff,
        run_event_supervision=args.run_event_supervision,
        labels_path=args.labels_path,
        labels_entity_col=args.labels_entity_column,
        labels_period_col=args.labels_period_column,
        labels_target_col=args.labels_target_column,
        labels_status_col=args.labels_status_column,
        labels_unlisted_as_negative=args.labels_unlisted_as_negative,
        labels_audit_attested=args.labels_audit_attested,
        labels_formal_calc_ok=args.labels_formal_calc_ok,
        label_horizon_months=args.label_horizon_months,
        label_confirm_delay_months=args.label_confirm_delay_months,
        label_maturity_buffer_months=args.label_maturity_buffer_months,
        label_washout_months=args.label_washout_months,
        labels_as_of=args.labels_as_of,
        review_capacity_k=args.review_capacity_k,
        hazard_horizon_months=args.hazard_horizon_months,
        event_challengers=args.event_challengers,
        event_bootstrap_reps=args.event_bootstrap_reps,
        tune_with_labels=args.tune_with_labels,
        tune_min_positive_rows=args.tune_min_positive_rows,
    )
    config.config_file = args.config
    if args.vae_categorical_representation is not None:
        config.vae_categorical_representation = args.vae_categorical_representation
    config.cli_explicit = tuple(sorted(_cli_explicit_fields(args)))
    if args.iforest_max_samples_range is not None:
        lo, hi = args.iforest_max_samples_range
        if not 2 <= lo <= hi:
            raise SystemExit("--iforest-max-samples-range needs 2 <= LO <= HI.")
        config.iforest_max_samples_range = (int(lo), int(hi))
    if args.iforest_selection_top_k is not None:
        if args.iforest_selection_top_k < 0:
            raise SystemExit("--iforest-selection-top-k must be >= 0 (0 = skip the phase).")
        config.iforest_selection_top_k = args.iforest_selection_top_k
    if args.iforest_noise_seeds is not None:
        if args.iforest_noise_seeds < 2:
            raise SystemExit("--iforest-noise-seeds must be >= 2 (needs a standard deviation).")
        config.iforest_noise_seeds = args.iforest_noise_seeds
    if args.tune_min_positive_rows < 1:
        raise SystemExit("--tune-min-positive-rows must be at least 1.")
    if args.labels_dir is not None:
        config.labels_dir = args.labels_dir
    if args.labels_usable_statuses is not None:
        config.labels_usable_statuses = tuple(args.labels_usable_statuses)
    if args.identification_columns is not None:
        config.identification_columns = tuple(args.identification_columns)
    if args.analyst_identity_columns is not None:
        config.analyst_identity_columns = tuple(args.analyst_identity_columns)
    for _flag, _val in (("--label-horizon-months", args.label_horizon_months),
                        ("--label-confirm-delay-months", args.label_confirm_delay_months),
                        ("--label-maturity-buffer-months", args.label_maturity_buffer_months),
                        ("--hazard-horizon-months", args.hazard_horizon_months)):
        if _val < 0 or (_flag == "--hazard-horizon-months" and _val < 1):
            raise SystemExit(f"{_flag} must be a non-negative number of months (hazard: at least 1).")
    if args.label_washout_months < 1:
        raise SystemExit("--label-washout-months must be at least 1.")
    if args.review_capacity_k is not None and args.review_capacity_k < 1:
        raise SystemExit("--review-capacity-k must be a positive number of rows.")
    if args.event_bootstrap_reps < 0:
        raise SystemExit("--event-bootstrap-reps cannot be negative.")
    # Validated here rather than inside the diagnostic bridge: a malformed
    # grid should stop the run at argument-parsing time, not halfway through
    # a fitted pipeline. `is not None` (not truthiness): `--diagnostic-
    # sensitivity-grid` with no values parses to `[]`, which must override
    # the dataclass default to "disabled", not be indistinguishable from the
    # flag never being passed.
    if args.diagnostic_sensitivity_grid is not None:
        grid = tuple(float(p) for p in args.diagnostic_sensitivity_grid)
        if any(not 0.0 < p < 1.0 for p in grid):
            raise SystemExit(
                "--diagnostic-sensitivity-grid takes percentile thresholds in (0, 1)."
            )
        config.diagnostic_sensitivity_grid = grid
    if args.diagnostic_stability_refits is not None:
        if args.diagnostic_stability_refits < 0:
            raise SystemExit("--diagnostic-stability-refits cannot be negative.")
        config.diagnostic_stability_refits = args.diagnostic_stability_refits
    # `is not None` here too: an explicit empty string ("disable the
    # breakdown") must override the "segment" default, and must be told
    # apart from the flag never being passed at all.
    if args.diagnostic_segment_column is not None:
        config.diagnostic_segment_column = args.diagnostic_segment_column
    if args.diagnostic_experiment_contamination_grid is not None:
        grid = tuple(float(c) for c in args.diagnostic_experiment_contamination_grid)
        if any(not 0.0 < c < 0.5 for c in grid):
            raise SystemExit(
                "--diagnostic-experiment-contamination-grid takes values in (0, 0.5)."
            )
        config.diagnostic_experiment_contamination_grid = grid
    if args.diagnostic_experiment_capacity_grid is not None:
        grid = tuple(int(d) for d in args.diagnostic_experiment_capacity_grid)
        if any(d <= 0 for d in grid):
            raise SystemExit(
                "--diagnostic-experiment-capacity-grid takes positive integers."
            )
        config.diagnostic_experiment_capacity_grid = grid
    if args.diagnostic_experiment_beta_grid is not None:
        grid = tuple(float(b) for b in args.diagnostic_experiment_beta_grid)
        if any(b <= 0 for b in grid):
            raise SystemExit("--diagnostic-experiment-beta-grid takes positive values.")
        config.diagnostic_experiment_beta_grid = grid
    if args.diagnostic_experiment_vae_fit_budget is not None:
        if args.diagnostic_experiment_vae_fit_budget < 0:
            raise SystemExit("--diagnostic-experiment-vae-fit-budget cannot be negative.")
        config.diagnostic_experiment_vae_fit_budget = args.diagnostic_experiment_vae_fit_budget
    if args.diagnostic_experiment_families is not None:
        from src.evaluation.ifvae_experiments import ALL_FAMILIES

        bad = [f for f in args.diagnostic_experiment_families if f not in ALL_FAMILIES]
        if bad:
            raise SystemExit(f"Unknown §9 family {bad}; valid: {list(ALL_FAMILIES)}.")
        config.diagnostic_experiment_families = tuple(args.diagnostic_experiment_families)
    if args.sensitivity_max_test_rows <= 0:
        raise SystemExit("--sensitivity-max-test-rows must be positive.")
    if args.sensitivity_combination_top_k < 2:
        raise SystemExit("--sensitivity-combination-top-k must be at least 2.")
    if args.sensitivity_random_subsets <= 0:
        raise SystemExit("--sensitivity-random-subsets must be positive.")
    if not 0.0 < args.sensitivity_high_zero_cutoff <= 1.0:
        raise SystemExit("--sensitivity-high-zero-cutoff must be in (0, 1].")
    if not 0.0 < args.exact_zero_row_cutoff <= 1.0:
        raise SystemExit("--exact-zero-row-cutoff must be in (0, 1].")
    if args.sensitivity_missing_levels is not None:
        levels = tuple(float(p) for p in args.sensitivity_missing_levels)
        if any(not 0.0 < p <= 1.0 for p in levels):
            raise SystemExit("--sensitivity-missing-levels values must be in (0, 1].")
        config.sensitivity_missing_levels = levels
    # Both apply after construction so they override the dataclass's own
    # `default_factory` dict rather than requiring the CLI to rebuild it.
    if args.contamination is not None:
        config.iforest_params["contamination"] = args.contamination
    if args.p95_percentile is not None:
        config.p95_percentile = args.p95_percentile
    return _enforce_model_training_contract(config)


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
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args._explicit_dests = _explicit_dests(parser, argv)
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
