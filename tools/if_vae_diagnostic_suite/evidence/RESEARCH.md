# Research synthesis

## Perspectives used

### Anomaly-detection scientist

IF and VAE encode different notions of abnormality. IF isolates points through
random partitions; a reconstruction model asks whether an input is difficult for
its learned mapping to reproduce. Agreement is useful, but disagreement is an
expected scientific object—not automatically a defect.

The strongest challenge to a reconstruction-only design is the demonstrated
ability of autoencoders to reconstruct inputs far outside their training support.
The suite therefore keeps reconstruction mean, localized top-K/max residual,
latent distance, and KL separate.

### Statistician

Raw model scores have incomparable units. Empirical percentiles are fit on a
reference population only. Metrics are evaluated under investigation capacity
with deterministic tie handling. Family analysis uses each family’s positives
against all known negatives; analyzing an all-positive family slice would yield
meaningless precision/average-precision results.

Ordinary row bootstrap intervals were rejected: dependence by account, network,
case, and time invalidates the i.i.d. assumption. A production project must define
its resampling block from the deployment unit.

### Fraud investigator

One total anomaly score is insufficient. The autopsy exports feature-level
observed/reconstructed values, absolute residual, standardized contribution, and
rank for every known positive. Coverage separates shared from unique IF/VAE hits,
which supports tiered queues instead of an opaque veto.

### ML reliability engineer

Reference and scoring data must be time-ordered. Imputation, IF fitting, residual
normalization, covariance, and ECDF calibration use reference rows only. All-
missing reference features, duplicate IDs, infinite values, reconstruction shape
mismatches, and timestamp overlaps fail or warn explicitly.

### Production operator

The diagnostic is local and deterministic, requires no API, fingerprints its
inputs, records the resolved configuration, and produces machine-readable files.
Precomputed IF scores remain supported, but seed stability is then marked
unavailable instead of fabricated.

## Sources

1. Liu, Ting & Zhou (2008), Isolation Forest — <https://doi.org/10.1109/ICDM.2008.17>
2. Liu, Ting & Zhou (2012), Isolation-Based Anomaly Detection — <https://doi.org/10.1145/2133360.2133363>
3. Scikit-learn IsolationForest API — <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>
4. Bouman & Heskes, Autoencoders for Anomaly Detection are Unreliable — <https://openreview.net/forum?id=X8XQOLjLX6>
5. Akrami et al., Robust VAE for Tabular Data with beta Divergence — <https://arxiv.org/abs/2006.08204>
6. Han et al., ADBench — <https://openreview.net/forum?id=foA_SFQ9zo0>
7. Scikit-learn TimeSeriesSplit — <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
8. Scikit-learn pipeline leakage guidance — <https://scikit-learn.org/stable/modules/compose.html>
9. Cosmic Ray mutation-testing concepts — <https://cosmic-ray.readthedocs.io/en/latest/concepts.html>

