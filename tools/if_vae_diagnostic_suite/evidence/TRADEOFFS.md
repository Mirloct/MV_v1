# Explicit trade-offs

| Decision | Benefit | Cost / unresolved boundary |
|---|---|---|
| Analyze exported VAE artifacts instead of training one | Faithful to the real model, loss, and preprocessing | The suite cannot perform capacity/contamination retraining without the user’s training pipeline |
| Reference-only empirical percentiles | Comparable scores without test leakage | Percentiles are cohort-dependent ranks, not anomaly probabilities |
| Top-K standardized absolute residual | Protects localized anomalies from mean dilution | Can overreact to a noisy feature; compare K and feature reliability on validation |
| Ledoit–Wolf latent Mahalanobis | Stable covariance in moderate/high dimension | Assumes second-order geometry is informative; multimodal latent populations may need conditional density models |
| Fresh multi-seed IF by default | Controlled stability evidence | It may differ from the production IF; use `if_score_col` to audit production scores, losing refit stability |
| High-severity timestamp overlap warning rather than hard stop | Permits diagnosis of legacy backtests | A warning can be ignored; production acceptance must fail the model gate if leakage is confirmed |
| No automatic threshold/weight optimization | Protects the final test and flagship case from overfitting | Users must maintain a validation period and lock choices themselves |
| No i.i.d. bootstrap confidence intervals | Avoids false certainty under account/network/time dependence | The organization must define grouped or temporal blocks before adding intervals |
| CSV/JSON/Markdown outputs | Auditable and tool-agnostic | Output files inherit source sensitivity and require access controls |
| Custom narrow mutation probe | Zero extra dependency, verifies critical formulas | It is not exhaustive mutation analysis; use Mutmut/Cosmic Ray in a fuller CI environment |

