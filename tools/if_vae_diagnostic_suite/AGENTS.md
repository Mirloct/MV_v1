# Engineering contract

This repository follows PillB `solarize_skill` v2.2 with a bounded, single-agent
Red -> Green -> Refactor -> Validate loop. Every behavior change starts with a
test that fails for the intended reason. Logs, comments, mocks, and synthetic
data are never accepted as sole proof of production model quality.

## Required order

1. Research the current methodology and inspect prior evidence.
2. State the behavior and write the smallest non-tautological failing test.
3. Implement only enough to satisfy that behavior.
4. Refactor while all behavior tests remain green.
5. Run two consecutive quiet adversarial validation rounds, bounded to five.
6. Record commands, results, failures, trade-offs, and unresolved limits.

## Evidence and failure rules

- Every task and every validation round must execute code and retain the result.
- Never use `assert True`, self-equality, implementation-mirroring assertions,
  empty tests, or mocks as proof of a core behavior.
- A failed fix hypothesis is recorded. Before retrying the same class of fix,
  research how others resolved the error.
- Stop and surface a hard dependency when an external service, API, or agent is
  unstable, unavailable, or usage-limited. Do not hammer it.
- Synthetic cases prove diagnostic behavior only. They never prove detection
  performance on operational data.
- Test data, thresholds, model weights, scalers, and score calibration must not
  use future or held-out test information.

## Expert rejection test

For every feature and material design decision, ask what a skeptical anomaly-
detection scientist, statistician, fraud investigator, ML reliability engineer,
and production operator would reject and why. If the reason applies, revise the
choice. Prefer evidence-supported correctness over convenience. State every
material trade-off in `evidence/TRADEOFFS.md`; never absorb it silently.

## Quality gate

Run `make chore-lint`. It must enforce:

- unit and adversarial behavior tests;
- no tautological tests or placeholder code;
- cyclomatic complexity <= 10 per function;
- compile/import smoke checks;
- deterministic synthetic regression checks.

Do not weaken or delete a gate to make a change pass.

