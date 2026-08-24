"""Interpretability module: SHAP / path-length for the Isolation Forest and
latent-space / per-feature reconstruction analysis for the VAE.

All figures are written under ``reports/figures/interpretability/`` per the
project-wide figures rule. Functions return plain ``dict``s / paths so the
reporting module can consume them directly.
"""

from src.interpretability.attribution_export import export_attribution_workbook
from src.interpretability.iforest_explain import (
    path_length_analysis,
    shap_summary_iforest,
)
from src.interpretability.vae_explain import (
    latent_space_plot,
    reconstruction_error_by_feature,
)

__all__ = [
    "shap_summary_iforest",
    "path_length_analysis",
    "latent_space_plot",
    "reconstruction_error_by_feature",
    "export_attribution_workbook",
]
