# Model experiment matrix

Run these with the real model-training pipeline. Export every result using the
same input contract, keep the final test untouched, and compare under the same
alert budgets and anomaly-family cohorts.

| Experiment family | Controlled variants | Primary diagnostics | Reject when |
|---|---|---|---|
| Baseline architecture | IF only; VAE only; intersection; union/max; mean; validation-fitted ensemble | Recall@K, Precision@K, Lift@K, unique coverage | VAE acts as a veto without incremental yield |
| VAE score | mean residual; top-1/3/5; max; latent distance; KL; validation-locked combination | family recall, rank, stability | score wins only on flagship case |
| Training contamination | original; remove known positives; trim IF top 0.1%; trim top 0.5%; iterative clean-normal | positive percentile shifts, normal false alerts | improvement exists only by deleting test-like examples |
| Capacity | latent 2/4/8/16/32; narrow/wide decoder; several seeds | active units, KL/unit, recon distribution, Recall@K | near-identity reconstructs anomalies or collapse removes information |
| KL regime | beta 0.1/0.5/1/2/4; warm-up; free bits if justified | active units, KL, validation ELBO, anomaly coverage | latent metrics are inactive or unstable |
| Feature likelihood | Gaussian/Student-t continuous; Bernoulli binary; categorical CE; count likelihood | per-type residual calibration | one MSE scale dominates unrelated features |
| Preprocessing | standard; robust; log1p+robust; rank; no destructive clipping | feature contribution, tail recall, drift | winsorization erases known tails or uses future statistics |
| Feature family | raw; relative-to-self; relative-to-peer; velocity; interaction; leave-family-out | family lift, segment lift, stability | added features dilute localized signal without unique positives |
| Population model | global; segment-specific; conditional VAE | segment yield, latent multimodality, drift | global model treats segment identity as anomaly |
| IF sensitivity | trees 100/300/500/1000; samples 256/512/1024/2048/all; features 0.4/0.6/0.8/1.0; 20+ seeds | Stability@K, rank spread, family recall | top alerts are seed/config artifacts |
| Temporal validation | rolling origin; gap for label maturation; entity/network group exclusion | detection delay, Recall@K, drift | any feature or scaler observes future events |
| Negative controls | permuted labels; shuffled suspect feature; timestamp-shifted feature | AP/lift collapse toward baseline | performance survives label permutation or future-only feature removal |

## Feature-design questions

For every raw variable `x`, consider separately:

1. magnitude: `x`, `log1p(x)`, tail indicator;
2. relative to self: `(x - rolling_median_entity) / rolling_MAD_entity`;
3. relative to peers: deviation from a pre-event peer baseline;
4. change: 1h/1d/7d/30d velocity, acceleration, time since previous event;
5. structure: concentration, entropy, new counterparties, graph/network features;
6. hypothesis-led interactions: ratios/differences with clear fraud meaning.

Every rolling window must be left-closed at scoring time and built from events
available then. Every peer definition must be fixed without using the outcome.

