from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DiagnosticConfig:
    features: list[str]
    id_col: str = "id"
    # `None` runs the suite in label-free mode: percentiles, quadrants,
    # latent/drift diagnostics, and the report still run; metrics, coverage,
    # autopsies, and orientation-risk warnings (all inherently supervised)
    # are skipped rather than raising. Official, real-data runs with no
    # ground truth need this -- see README "Required inputs".
    label_col: str | None = "label"
    time_col: str | None = "event_time"
    segment_col: str | None = "segment"
    family_col: str | None = "anomaly_family"
    reconstruction_prefix: str = "recon__"
    latent_mu_prefix: str = "mu__"
    latent_logvar_prefix: str = "logvar__"
    if_score_col: str | None = None
    if_higher_is_anomalous: bool = True
    vae_primary_score: str = "recon_topk"
    top_k_residuals: int = 3
    percentile_threshold: float = 0.95
    alert_budgets: list[int] = field(default_factory=lambda: [25, 50, 100])
    random_seeds: list[int] = field(default_factory=lambda: [7, 19, 43, 71, 101])
    if_n_estimators: int = 300
    if_max_samples: int | float | str = "auto"
    if_max_features: float = 1.0
    n_jobs: int = -1
    active_variance_threshold: float = 1e-3

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("features must be non-empty")
        if self.top_k_residuals < 1:
            raise ValueError("top_k_residuals must be >= 1")
        if not 0.5 < self.percentile_threshold < 1.0:
            raise ValueError("percentile_threshold must be between 0.5 and 1")
        if any(k < 1 for k in self.alert_budgets):
            raise ValueError("alert_budgets must contain positive integers")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> DiagnosticConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML mapping")
    return DiagnosticConfig(**data)


def save_config(config: DiagnosticConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
