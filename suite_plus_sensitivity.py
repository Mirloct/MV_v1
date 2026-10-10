"""Run the full report/diagnostic/sensitivity suite on an already-trained model.

`python main.py` always runs the complete flow: tune, fit, then suite. Once a
model has been trained at least once, re-running that full flow just to
regenerate the report, the IF-VAE diagnostic suite, the sensitivity analysis
and the OOT Excel deliverables is wasted time -- the expensive part (Optuna
tuning + VAE training) produces the same detectors every time nothing about
the data or the preprocessing config changed.

This script is `python main.py --reuse-trained` under a convenience name: it
loads the Isolation Forest and VAE already saved under `artifacts/models/`
(and the optimal hyperparameters already saved under `artifacts/tuning/
best_params_*.yaml`) instead of tuning/fitting new ones, then runs every
other phase exactly as a full run would -- same report, same dashboard, same
Excel deliverables, same diagnostic suite, same sensitivity analysis.

Usage::

    python suite_plus_sensitivity.py              # same flags main.py accepts
    python suite_plus_sensitivity.py --quick       # e.g. against the quick preset

Fails loudly, it does not fall back to training, if no trained model is found
under `artifacts/models/` yet, or if the saved VAE's architecture fingerprint
does not match the current data/preprocessing config (see `--reuse-trained`'s
help in `main.py` for exactly what is checked). Run `python main.py` once
first in that case.
"""
from __future__ import annotations

import sys

from main import main

_FLAG = "--reuse-trained"


def _forward_argv(argv: list[str]) -> list[str]:
    return argv if _FLAG in argv else [*argv, _FLAG]


if __name__ == "__main__":
    main(_forward_argv(sys.argv[1:]))
