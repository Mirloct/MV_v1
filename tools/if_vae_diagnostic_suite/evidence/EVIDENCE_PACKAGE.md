# Evidence package — cycle `ifvae-2026-09-02-001`

Status: **passed**

## Gates

| Phase | Result | Executed evidence |
|---|---|---|
| Research | pass | Primary papers/docs reviewed; STORM perspectives synthesized in `RESEARCH.md` |
| Red | pass | Initial 8 modules failed with missing package; later behavior tests failed for hidden generator shift, all-missing feature, invalid family cohort, and missing warnings output |
| Green | pass | Diagnostic package, CLI, reports, data contracts, model outputs, metrics, and tests implemented |
| Refactor | pass | Quality gate caught complexity 14; function split; ceiling now <=10 |
| Validate round 1 | pass | 25 tests; 4/4 mutants killed; direct + CSV CLI; exact 10 IF-only/10 VAE-only/10 both |
| Validate round 2 | pass | 26 tests; 4/4 mutants killed; independent seed 23; direct + CSV CLI; exact intended families; 30 autopsy IDs |
| Convergence | pass | 2 consecutive quiet rounds, below max 5 |

## Final commands

```text
make chore-lint
PYTHONPATH=src python -m ifvae_diag simulate --out build/validation2 --seed 23
PYTHONPATH=src python -m ifvae_diag run \
  --reference build/validation2/synthetic_reference.csv \
  --scored build/validation2/synthetic_scored.csv \
  --config build/validation2/synthetic_config.yaml \
  --out build/validation2_csv
```

## Final results

```text
26 tests passed
mutant percentile_tie_direction killed=True
mutant if_quadrant_threshold_exclusive killed=True
mutant vae_quadrant_threshold_exclusive killed=True
mutant lift_formula killed=True
compile=True tests=True mutations=True complexity_limit=10
positive families: IF_ONLY=10, VAE_ONLY=10, BOTH=10
CSV round trip: identical family/quadrant counts
known-positive autopsy IDs: 30
```

## Bugs and blind spots found and fixed

1. Synthetic reference/scored normals used separate loading matrices, introducing
   hidden covariate shift. Fixed by sharing the data-generating mechanism and
   adding a KS effect-size bound.
2. A one-feature extreme did not consistently clear the IF 95th percentile in an
   eight-feature space. After reviewing masking/high-dimensional literature, the
   point-anomaly fixture was made multivariate and family behavior locked by test.
3. All-missing reference features could be silently dropped by an imputer. They
   now fail the data contract.
4. Family-only slices contained no negatives and generated invalid evaluation.
   Family cohorts now contain the target family’s positives plus all negatives.
5. The tautology scanner itself exceeded the complexity ceiling. It was split;
   the gate scans its own scripts and grants no exemption.
6. A text mutant ambiguously changed two threshold sites. After reviewing
   location-specific mutation operators, it became two independent mutants.
7. Warnings existed only in Markdown. `warnings.json` now supports automated gates.

## Packaging check and transient tool block

A combined command containing the optional wheel build returned an environment
network/approval rejection, so that blocked command was not retried. A wheel
artifact had nevertheless been produced locally before the tool-level rejection.
It was subsequently validated without network access: ZIP integrity passed, it
installed into a temporary `--system-site-packages` virtual environment using
`--no-index --no-deps`, and the installed `ifvae-diagnose simulate` console command
completed and wrote its report. No external package service was retried.

## Addendum 2026-09-23 — live progress (`ifvae_diag/progress.py`)

Behavior added: named steps with measured durations, tqdm bars for loop-shaped
tests, and an observer hook, all instrumentation-only (no computed value changes).

| Phase | Result | Executed evidence |
|---|---|---|
| Red | pass | `tests/test_progress.py` failed with `ImportError` (module absent) before implementation; a second red/green cycle fixed `running()` not publishing the in-flight item (test asserts the newest event already names it) |
| Green | pass | 16 new tests in `tests/test_progress.py`, including one that drives the real `run_diagnostic` and asserts its stage names, ordering and inner bars |
| Refactor | pass | `run_diagnostic` stayed <=10 complexity by extracting `_percentile_frame`; `_write_outputs` became a tracked list of writers |
| Validate round 1 | pass | `scripts/quality_gate.py`: 47 tests, 4/4 mutants killed, compile ok, complexity <=10 |
| Validate round 2 | pass | identical result, no source change between rounds |

Result of both rounds: `Ran 47 tests ... OK`, `compile=True tests=True
mutations=True complexity_limit=10`. Unresolved: progress state is module-global
and not thread-safe (see `TRADEOFFS.md`); no dedicated mutant targets the new
module, so its tests are the only guard on it.

## Scope of proof

The evidence proves the diagnostic software behaviors above. It does not prove
the user’s production IF, VAE, feature set, thresholds, or ensemble meet a target
yield. Those require the real exported artifacts and the locked temporal backtest.
