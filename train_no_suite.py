"""Run the full training flow and core deliverables, skipping the IF-VAE
diagnostic suite and the post-training sensitivity analysis.

`python main.py` runs everything: tune, fit, evaluate, export the OOT Excel
deliverables, build the report/dashboard, AND run the IF-VAE diagnostic suite
(Phase 9c) plus the post-training sensitivity analysis (Phase 9d). Those two
phases are the most expensive ones after tuning itself (several VAE refits
each) and are not always wanted on every run -- e.g. a quick turnaround while
iterating on preprocessing, or a machine without the vendored diagnostic
package installed.

This script is `python main.py --no-suite` under a convenience name: tuning
and fitting happen exactly as in a full run, but Phase 9c/9d are skipped.
Everything else -- data load, preprocessing, evaluation, OOT Excel exports,
report, dashboard, feature attribution -- runs unchanged.

Usage::

    python train_no_suite.py             # same flags main.py accepts
    python train_no_suite.py --quick      # e.g. against the quick preset

See also:
    suite_plus_sensitivity.py   -- the opposite combination: skip tuning/
                                    fitting (reuse an already-trained model)
                                    but DO run the suite.
    best_params_no_suite.py     -- both combined: reuse an already-trained
                                    model AND skip the suite.
"""
from __future__ import annotations

import sys

from main import main

_FLAG = "--no-suite"


def _forward_argv(argv: list[str]) -> list[str]:
    return argv if _FLAG in argv else [*argv, _FLAG]


if __name__ == "__main__":
    main(_forward_argv(sys.argv[1:]))
