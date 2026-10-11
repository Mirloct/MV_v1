"""Run the core deliverables straight from the last trained model and its
optimal hyperparameters -- no tuning, no fitting, no diagnostic/sensitivity
suite.

Of the three ways to run this project:

* `python main.py`                   -- full flow: tune, fit, suite.
* `python suite_plus_sensitivity.py`  -- reuse the trained model, DO run the suite.
* `python train_no_suite.py`         -- tune/fit normally, skip the suite.
* `python best_params_no_suite.py`   -- (this script) reuse the trained model,
                                         skip the suite too: the fastest path
                                         to the OOT Excel/report/dashboard
                                         from an already-trained model.

This script is `python main.py --best-params` under a convenience name --
itself shorthand for `--reuse-trained --no-suite` combined. It loads the
Isolation Forest and VAE already saved under `artifacts/models/` (and the
optimal hyperparameters already saved under `artifacts/tuning/
best_params_*.yaml`) instead of tuning/fitting new ones, skips Phase 9c/9d,
and runs every other phase -- evaluation, OOT Excel exports, report,
dashboard, feature attribution -- exactly as a full run would.

Usage::

    python best_params_no_suite.py            # same flags main.py accepts
    python best_params_no_suite.py --quick     # e.g. against the quick preset

Fails loudly, it does not fall back to training, if no trained model is found
under `artifacts/models/` yet, or if the saved VAE's architecture fingerprint
does not match the current data/preprocessing config (see `--reuse-trained`'s
help in `main.py` for exactly what is checked). Run `python main.py` once
first in that case.
"""
from __future__ import annotations

import sys

from main import main

_FLAG = "--best-params"


def _forward_argv(argv: list[str]) -> list[str]:
    return argv if _FLAG in argv else [*argv, _FLAG]


if __name__ == "__main__":
    main(_forward_argv(sys.argv[1:]))
