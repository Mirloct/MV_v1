from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from . import progress
from .config import DiagnosticConfig


def prepare_features(
    reference: pd.DataFrame, scored: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    reference_x = imputer.fit_transform(reference[features])
    scored_x = imputer.transform(scored[features])
    return reference_x, scored_x


def isolation_forest_scores(
    reference_x: np.ndarray,
    scored_x: np.ndarray,
    config: DiagnosticConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_runs: list[np.ndarray] = []
    scored_runs: list[np.ndarray] = []
    with progress.step("isolation_forest_scores", label="Isolation Forest, one fit per seed"):
        seeds = progress.track(
            config.random_seeds, desc="isolation_forest[seeds]", unit="seed",
            label=lambda seed: f"seed={seed}",
        )
        for seed in seeds:
            model = IsolationForest(
                n_estimators=config.if_n_estimators,
                max_samples=config.if_max_samples,
                max_features=config.if_max_features,
                contamination="auto",
                random_state=seed,
                n_jobs=config.n_jobs,
            )
            model.fit(reference_x)
            reference_runs.append(-model.score_samples(reference_x))
            scored_runs.append(-model.score_samples(scored_x))
    return (
        np.mean(reference_runs, axis=0),
        np.mean(scored_runs, axis=0),
        np.vstack(scored_runs),
    )

