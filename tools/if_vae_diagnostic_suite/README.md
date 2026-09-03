# IF–VAE Diagnostic Suite

An executable, evidence-oriented suite for diagnosing why an Isolation Forest
and a variational autoencoder disagree on rare tabular anomalies. It treats
disagreement as signal: IF-only, VAE-only, both, and neither are analyzed as
separate populations under the real investigation budget.

The suite does **not** train a generic VAE and pretend it represents your model.
It consumes reconstructions and latent parameters exported from the actual VAE,
because preprocessing, decoder likelihood, feature types, capacity, and training
contamination determine what its scores mean.

## What it executes

- reference-only ECDF anomaly percentiles (no calibration on test rows);
- Isolation Forest refits across seeds and top-K stability/Jaccard diagnostics;
- normalized mean, maximum, and top-K per-feature reconstruction scores;
- latent Ledoit–Wolf Mahalanobis distance, row KL, active units, and collapse risk;
- IF × VAE disagreement quadrants and known-positive feature autopsies;
- Precision@K, Recall@K, Lift@K, Average Precision, and unique/shared coverage;
- segment metrics and valid anomaly-family cohorts (family positives vs all negatives);
- drift, missingness, out-of-reference-range, timestamp-overlap, and leakage-name checks;
- deterministic reports, data fingerprints, tie-stable rankings, and local plots;
- non-tautological unit/adversarial tests, four killed source mutants, compilation,
  and a cyclomatic-complexity ceiling of 10.

## Required inputs

Provide two CSV files:

1. `reference.csv`: the past, preferably clean normal population used only to fit
   preprocessing/calibration and the diagnostic IF.
2. `scored.csv`: a strictly later validation/test population with binary labels.

Both files need:

- a unique ID;
- the exact numeric feature tensor used by the VAE and IF;
- one reconstruction column per feature: `recon__<feature>`;
- optionally latent means `mu__0...N` and log-variances `logvar__0...N`;
- preferably scoring timestamps; `scored.csv` additionally needs labels, segments,
  and anomaly families.

Observed values and reconstructions must be in the **same coordinate system**.
If the VAE consumes robust-scaled/log-transformed inputs, export those inputs and
their reconstructions—not raw business values mixed with scaled outputs. Keep a
separate business-value table for investigator presentation if required.

To diagnose the production IF rather than a fresh controlled refit, export its
score in both files and set `if_score_col`. Set `if_higher_is_anomalous: false`
when larger raw values mean *more normal* (as with `score_samples` in many APIs).

## Run

```bash
cd if_vae_diagnostic_suite
make chore-lint
PYTHONPATH=src python -m ifvae_diag run \
  --reference path/to/reference.csv \
  --scored path/to/scored.csv \
  --config example_config.yaml \
  --out build/real_run
```

Exercise the entire pipeline without production data:

```bash
PYTHONPATH=src python -m ifvae_diag simulate --out build/demo --seed 7
```

The synthetic run contains deliberately engineered point/extrapolation,
relationship-break, and shared anomalies. It proves that the diagnostics identify
known failure modes; it does not estimate live precision or recall.

## Outputs

| File | Purpose |
|---|---|
| `report.md` | Executive interpretation and risk summary |
| `disagreement.png` | IF percentile × VAE percentile plot |
| `scored_diagnostics.csv` | Row-level scores, ranks, ensembles, and quadrants |
| `autopsies.csv` | Known-positive observed/reconstructed values and residual ranks |
| `metrics.csv` | Operational metrics at each alert budget |
| `metrics_by_group.csv` | Segment and anomaly-family breakdowns |
| `coverage.json` | Shared and unique positive coverage at K |
| `if_stability.json` | Seed refit selection probabilities and top-K overlap |
| `drift.csv` | Feature-level population shift checks |
| `summary.json` | Machine-readable run result and input fingerprints |
| `warnings.json` | Machine-readable leakage, orientation, collapse, and availability risks |
| `resolved_config.json` | Exact run configuration |

Outputs inherit the sensitivity of the input data. Use pseudonymous IDs and keep
the run directory under the same access controls as the model-development data.

## Interpreting the primary VAE score

`vae_primary_score` controls the two-dimensional quadrant and is not silently
optimized by the suite. Recommended first comparison:

- `recon_mean`: broad reconstruction failure;
- `recon_topk`: localized failure that mean loss can dilute;
- `recon_max`: a deliberately sensitive single-feature alarm;
- `latent_mahalanobis`: unusual representation even when reconstruction is good;
- `vae_kl`: posterior divergence, if latent log-variance was exported.

Run all candidates and choose the operational score on a validation period. Lock
the choice before the final test. An expressive autoencoder may reconstruct
out-of-distribution inputs well, so low reconstruction error is not proof of
normality.

## Required model experiments after the first report

The code diagnoses an exported run. The experiment matrix in
`evidence/EXPERIMENT_MATRIX.md` specifies the reruns that require your actual VAE
training pipeline: contamination trimming, loss by feature type, latent/capacity
ablation, beta/KL schedules, preprocessing, feature-family ablation, and temporal
backtests. Each rerun should export the same contract and be compared under the
same alert budget.

Do not use the sequential rule `IF AND VAE` as the default. It makes the VAE a
veto and caps recall. Compare it against IF-only, VAE-only, percentile maximum,
percentile mean, and a validation-fitted ensemble; keep the final test untouched.

## Scientific boundaries

- Percentiles make heterogeneous scores comparable; they are not calibrated
  probabilities of fraud.
- The Ledoit–Wolf latent distance is a stable Gaussian second-order diagnostic,
  not proof that the latent distribution is Gaussian.
- Name-based leakage checks are sentinels, not a substitute for feature lineage.
- KS p-values become extremely sensitive at scale; inspect effect size and business
  meaning, not significance alone.
- Row bootstrap intervals are intentionally omitted because banking observations
  are often dependent by customer, case, and time. Use grouped or temporal block
  resampling aligned with your unit of deployment.
- No one model or metric is “SOTA” for every anomaly family. Benchmark coverage,
  stability, and operational yield—not architecture prestige.

## Research basis

- Liu, Ting, and Zhou introduced Isolation Forest as isolation by short random-tree
  paths and later analyzed subsampling, masking, swamping, and high dimensions:
  <https://doi.org/10.1109/ICDM.2008.17> and
  <https://doi.org/10.1145/2133360.2133363>.
- Scikit-learn documents that `max_samples="auto"` uses `min(256, n)` and that
  `contamination` defines a threshold: <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>.
- Bouman and Heskes demonstrate that autoencoders can reconstruct far-away
  anomalies, challenging reconstruction-only scoring:
  <https://openreview.net/forum?id=X8XQOLjLX6>.
- Akrami et al. analyze VAE robustness to contaminated mixed tabular data:
  <https://arxiv.org/abs/2006.08204>.
- ADBench compares 30 anomaly detectors across broad datasets and anomaly settings:
  <https://openreview.net/forum?id=foA_SFQ9zo0>.
- Time-ordered evaluation avoids training on the future:
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>.
- Mutation testing checks whether tests actually reject changed behavior:
  <https://cosmic-ray.readthedocs.io/en/latest/concepts.html>.

## Engineering contract

`AGENTS.md` captures the Solarize v2.2 Red→Green→Refactor→Validate gates derived
from `PillB/solarize_skill`. `CLAUDE.md` points every compatible agent to it.
`make chore-lint` must remain green; do not weaken a failing gate.
