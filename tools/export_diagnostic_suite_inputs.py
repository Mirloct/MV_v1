"""Export a real, freshly-fitted Modelo v0.1 IF+VAE run into the IF-VAE
Diagnostic Suite's ``reference.csv`` / ``scored.csv`` contract
(``tools/if_vae_diagnostic_suite``).

Mirrors ``main.py``'s own Phases 2 -> 3a -> 4 -> 6 -> 6b -> 7 (same functions,
same config shape, same seed) so every exported score/reconstruction/latent
value comes from a real, deterministically-reproducible IF+VAE pair fit under
this project's actual default configuration (stacking on) -- not a synthetic
stand-in. See ``README.md`` in the suite: "The suite does not train a generic
VAE and pretend it represents your model."

Ground truth is attached to ``scored.csv`` ONLY for this integration test.
The suite's own data contract *requires* a binary label column
(``contracts.py::_require_columns``) -- Modelo v0.1's official, real-data runs
are unsupervised and carry no target at all, so a real production run cannot
produce a valid ``scored.csv`` for this tool as-is. This script uses the
project's own synthetic ground truth (``ground_truth.parquet``, injected
labels for testing, never used to fit anything upstream) so the suite can be
exercised end to end against real project data. See CONTEXT.md "IF-VAE
Diagnostic Suite integration" for exactly which of the suite's outputs
remain meaningful without a label column, and which are inherently
supervised.

One row per (entity_id, period) in the OOT window -- not deduplicated to one
row per entity the way the OOT Excel deliverable is. Each (entity, month) is
a distinct scoring event, and the suite has no notion of "each entity's best
month" -- passing every row gives its drift/disagreement/family diagnostics
more to work with, at the cost of a known positive counted once per month it
was flagged rather than once per entity.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

from src.data import load_or_generate_panel  # noqa: E402
from src.evaluation import (  # noqa: E402
    chronological_split,
    load_ground_truth_labels,
    load_ground_truth_types,
)
from src.models import IsolationForestDetector, VAEDetector, build_stacked_matrix  # noqa: E402
from src.preprocessing import fit_transform_panel, split_matrix_for_model  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def _densify(X, dtype=np.float64) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=dtype)


def _vae_forward(detector: VAEDetector, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mu, logvar, reconstruction)``, batched -- same convention as
    ``VAEDetector.score_samples``/``vae_explain.py::explain_rows_vae``."""
    import torch

    model = detector._check_fitted()
    # float32, matching `src.models.vae._densify` -- the model's Linear
    # layers were built with float32 weights, and PyTorch does not
    # auto-promote a float64 input for them.
    Xd = _densify(X, dtype=np.float32)
    model.eval()
    n, bs = Xd.shape[0], detector.batch_size
    mus = np.empty((n, model.latent_dim), dtype=np.float64)
    logvars = np.empty((n, model.latent_dim), dtype=np.float64)
    recon = np.empty_like(Xd)
    with torch.no_grad():
        for start in range(0, n, bs):
            chunk = Xd[start:start + bs]
            xb = torch.from_numpy(chunk).to(detector.device)
            mu, logvar = model.encode(xb)
            xr = model.decode(mu)
            end = start + xb.size(0)
            mus[start:end] = mu.cpu().numpy()
            logvars[start:end] = logvar.cpu().numpy()
            recon[start:end] = xr.cpu().numpy()
    return mus, logvars, recon


def _build_frame(
    entity_ids: np.ndarray,
    periods: np.ndarray,
    features_matrix: np.ndarray,
    feature_names: list[str],
    recon: np.ndarray,
    mu: np.ndarray,
    logvar: np.ndarray,
    if_score: np.ndarray,
    labels: "np.ndarray | None" = None,
    families: "np.ndarray | None" = None,
    segments: "np.ndarray | None" = None,
) -> pd.DataFrame:
    # Built via one `pd.concat` rather than repeated `frame[col] = ...`
    # assignment (which fragments the block manager across ~2x feature-count
    # + latent-dim*2 individual inserts and triggers pandas'
    # PerformanceWarning at this column count).
    blocks = [
        pd.DataFrame({"alert_id": [f"{e}__{p}" for e, p in zip(entity_ids, periods)]}),
        pd.DataFrame({"scoring_timestamp": periods}),
        pd.DataFrame(features_matrix, columns=feature_names),
        pd.DataFrame(recon, columns=[f"recon__{f}" for f in feature_names]),
        pd.DataFrame(mu, columns=[f"mu__{i}" for i in range(mu.shape[1])]),
        pd.DataFrame(logvar, columns=[f"logvar__{i}" for i in range(logvar.shape[1])]),
        pd.DataFrame({"if_score": if_score}),
    ]
    if labels is not None:
        blocks.append(pd.DataFrame({"confirmed_case": labels}))
    if families is not None:
        blocks.append(pd.DataFrame({"anomaly_family": families}))
    if segments is not None:
        blocks.append(pd.DataFrame({"customer_segment": segments}))
    return pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        _REPO_ROOT, "tools", "if_vae_diagnostic_suite", "build", "modelo_run"))
    parser.add_argument("--n-individuals", type=int, default=500)
    parser.add_argument("--n-periods", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae-epochs", type=int, default=20)
    args = parser.parse_args()

    log = setup_logging()
    os.makedirs(args.out, exist_ok=True)

    log.info("Loading panel (n_individuals=%d, n_periods=%d, seed=%d)...",
              args.n_individuals, args.n_periods, args.seed)
    df, schema = load_or_generate_panel(
        n_individuals=args.n_individuals, n_periods=args.n_periods, seed=args.seed,
    )
    entity_col = schema.entity_col or "entity_id"
    time_col = schema.time_col or "period"

    split = chronological_split(
        df, time_col=time_col, n_val_periods=2, n_test_periods=3, n_oot_periods=3,
    )
    train_mask, val_mask = split.train_mask, split.val_mask
    oot_mask = split.oot_mask
    in_mask = train_mask | val_mask
    valid_local = val_mask[in_mask]

    labels = load_ground_truth_labels(schema, df[[entity_col, time_col]])
    label_types = load_ground_truth_types(schema, df[[entity_col, time_col]])

    log.info("Fitting preprocessing on the train block...")
    X, keys, feature_names = fit_transform_panel(
        df, schema, fit_mask=train_mask, rare_min_frequency=0.001,
    )
    X_if, _names_if = split_matrix_for_model(X, feature_names, "iforest")

    iforest_params = dict(
        n_estimators=200, contamination=0.02, max_samples="auto",
        max_features=1.0, bootstrap=False,
    )
    log.info("Fitting the production Isolation Forest (train+val)...")
    if_detector = IsolationForestDetector(random_state=args.seed, **iforest_params)
    if_detector.fit(X_if[in_mask])
    if_scores = if_detector.score_samples(X_if)

    # Same stacking rule as main.py Phase 6b: a SEPARATE forest fit on train
    # only produces the score column the VAE trains on -- the "production"
    # `if_detector` above is fit on train+val, which would otherwise leak an
    # in-sample column into the VAE's input.
    log.info("Fitting the stacking Isolation Forest (train only) + building X_vae...")
    stack_detector = IsolationForestDetector(
        random_state=args.seed, n_jobs=-1, **iforest_params,
    )
    stack_detector.fit(X_if[train_mask])
    stack_scores = stack_detector.score_samples(X_if)
    stacked = build_stacked_matrix(
        X, stack_scores, fit_mask=train_mask, feature_names=feature_names,
    )
    X_vae, vae_feature_names = stacked.X, stacked.feature_names

    log.info("Fitting the VAE (%d epochs) on %d features...",
              args.vae_epochs, len(vae_feature_names))
    vae_detector = VAEDetector(
        random_state=args.seed, epochs=args.vae_epochs,
        latent_dim=8, hidden_dim=64, n_layers=2, dropout=0.0, beta=1.0,
        lr=1e-3, optimizer="adam", batch_size=256, weight_decay=0.0,
        activation="relu", kl_anneal_epochs=10, early_stopping_patience=10,
    )
    vae_detector.fit(X_vae[in_mask], valid_mask=valid_local)

    log.info("Computing reconstructions/latents for every row...")
    mu, logvar, recon = _vae_forward(vae_detector, X_vae)
    X_vae_dense = _densify(X_vae)

    entity_ids = keys[entity_col].to_numpy()
    periods = keys[time_col].astype(str).to_numpy()
    segments = df["segment"].to_numpy() if "segment" in df.columns else None

    def _slice(mask, with_labels: bool) -> pd.DataFrame:
        return _build_frame(
            entity_ids[mask], periods[mask], X_vae_dense[mask], vae_feature_names,
            recon[mask], mu[mask], logvar[mask], if_scores[mask],
            labels=(labels[mask].astype(int) if with_labels else None),
            families=(label_types[mask] if with_labels and label_types is not None else None),
            segments=(segments[mask] if with_labels and segments is not None else None),
        )

    reference = _slice(train_mask, with_labels=False)
    scored = _slice(oot_mask, with_labels=True)

    reference_path = os.path.join(args.out, "reference.csv")
    scored_path = os.path.join(args.out, "scored.csv")
    reference.to_csv(reference_path, index=False)
    scored.to_csv(scored_path, index=False)

    config = {
        "features": vae_feature_names,
        "id_col": "alert_id",
        "label_col": "confirmed_case",
        "time_col": "scoring_timestamp",
        "segment_col": "customer_segment",
        "family_col": "anomaly_family",
        "reconstruction_prefix": "recon__",
        "latent_mu_prefix": "mu__",
        "latent_logvar_prefix": "logvar__",
        "if_score_col": "if_score",
        "if_higher_is_anomalous": True,
        "vae_primary_score": "recon_topk",
        "top_k_residuals": 5,
        "percentile_threshold": 0.95,
        "alert_budgets": [10, 25, 50],
        "random_seeds": [7, 19, 43],
        "if_n_estimators": 200,
        "if_max_samples": "auto",
        "if_max_features": 1.0,
        "n_jobs": -1,
        "active_variance_threshold": 0.001,
    }
    config_path = os.path.join(args.out, "modelo_config.yaml")
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    log.info(
        "Exported reference.csv (%d rows, train block) and scored.csv "
        "(%d rows, OOT block, %d known positives) -> %s",
        len(reference), len(scored), int(scored["confirmed_case"].sum()), args.out,
    )
    print(f"reference: {reference_path} ({len(reference)} rows)")
    print(f"scored:    {scored_path} ({len(scored)} rows, "
          f"{int(scored['confirmed_case'].sum())} known positives)")
    print(f"config:    {config_path}")


if __name__ == "__main__":
    main()
