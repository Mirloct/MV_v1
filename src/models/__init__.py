"""Model package: anomaly detectors.

Exposes the Isolation Forest detector and the Variational Autoencoder (VAE)
detector, along with their Optuna tuning and plotting helpers. Both are thin,
project-consistent wrappers that share the same "higher = more anomalous"
score convention and Optuna/SQLite/YAML crash-recovery pattern.
"""

from src.models.iforest import (
    IsolationForestDetector,
    plot_score_distribution,
    tune_iforest,
)
from src.models.stacking import (
    DEFAULT_SCORE_FEATURE,
    StackedMatrix,
    build_stacked_matrix,
    score_shift_report,
)
from src.models.vae import (
    VAEDetector,
    VAEModel,
    collapse_verdict,
    plot_latent_space,
    plot_reconstruction_error,
    tune_vae,
    vae_loss,
)

__all__ = [
    "IsolationForestDetector",
    "tune_iforest",
    "plot_score_distribution",
    "build_stacked_matrix",
    "StackedMatrix",
    "score_shift_report",
    "DEFAULT_SCORE_FEATURE",
    "VAEModel",
    "VAEDetector",
    "tune_vae",
    "vae_loss",
    "collapse_verdict",
    "plot_reconstruction_error",
    "plot_latent_space",
]
