"""Evaluation module: OOT split, ground-truth join, metrics, scoring, reporting.

Ties the anomaly detectors to the hidden ground truth and produces the project's
headline business deliverable -- the out-of-time (OOT) top-decile anomaly Excel
export (ID - SCORE - VARIABLES).
"""

from src.evaluation.labels import load_ground_truth_labels, load_ground_truth_types
from src.evaluation.metrics import (
    metrics_by_anomaly_type,
    supervised_metrics,
    unsupervised_metrics,
)
from src.evaluation.oot_report import (
    BAND_COL,
    DEFAULT_TOP_N,
    PERCENTILE_BANDS,
    export_oot_top_anomalies,
    export_oot_top_decile,
    export_p95_checkpoint,
    months_present_by_entity,
)
from src.evaluation.scoring import build_scored_frame
from src.evaluation.splits import (
    ChronologicalSplit,
    chronological_split,
    oot_period,
    oot_split,
)
from src.evaluation.thresholds import (
    THRESHOLD_METHODS,
    apply_threshold,
    calibrate_threshold,
)
from src.evaluation.visualize import plot_embedding, plot_roc_pr

__all__ = [
    "oot_split",
    "oot_period",
    "chronological_split",
    "ChronologicalSplit",
    "calibrate_threshold",
    "apply_threshold",
    "THRESHOLD_METHODS",
    "export_oot_top_anomalies",
    "DEFAULT_TOP_N",
    "BAND_COL",
    "PERCENTILE_BANDS",
    "load_ground_truth_labels",
    "load_ground_truth_types",
    "supervised_metrics",
    "unsupervised_metrics",
    "metrics_by_anomaly_type",
    "build_scored_frame",
    "export_oot_top_decile",
    "export_p95_checkpoint",
    "months_present_by_entity",
    "plot_embedding",
    "plot_roc_pr",
]
