"""Shared trial-level early stopping for Optuna studies (both `tune_iforest` and `tune_vae`).

Distinct from the VAE's *per-epoch* early stopping inside a single `.fit()`
call (`VAEDetector.early_stopping_patience`, `src/models/vae.py`) -- that one
already exists and is untouched here. This module stops the *outer* Optuna
trial loop, which is not epoch-based for either model's search: each trial is
an independent draw from the hyperparameter space, not one more step of the
same optimization.

THEORY. There is no "loss stopped decreasing" notion for an independent draw
the way there is for a training epoch, and Optuna's TPE sampler carries no
convergence guarantee to test for. What *is* well-defined and testable is
economic, not statistical: has the best-seen value moved by at least
`min_delta` (relative) over the last `patience` trials -- i.e. is the next
trial still paying for its own (fixed, real) compute cost at the observed
improvement rate. `Study.stop()` (verified against the installed
optuna==4.9.0 API before writing this) requests the running
`study.optimize()` loop exit after the current trial finishes; it changes
nothing about how already-completed trials were scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.utils import observability
from src.utils.logging_config import setup_logging


@dataclass
class TrialPatienceStopper:
    """Optuna `study.optimize(..., callbacks=[...])` callback.

    Stops the study once `patience` consecutive trials have passed without a
    `min_delta`-relative improvement in `study.best_value`, and never before
    `min_trials` trials have completed (a handful of early draws are not
    evidence of anything, in either direction).
    """

    direction: str                    # "maximize" | "minimize", as resolved by the caller
    model_name: str                   # "iforest" | "vae" -- logging/event label only
    n_trials_requested: int
    patience: int = 10
    min_delta: float = 0.005          # relative improvement, e.g. 0.005 = 0.5%
    min_trials: int = 10

    stopped: bool = field(default=False, init=False)
    stop_reason: Optional[str] = field(default=None, init=False)
    trials_skipped: int = field(default=0, init=False)
    _best: Optional[float] = field(default=None, init=False)
    _best_trial_number: int = field(default=-1, init=False)
    _trials_since_improvement: int = field(default=0, init=False)

    def _improved(self, candidate: float) -> bool:
        if self._best is None:
            return True
        if self.direction == "maximize":
            threshold = (
                self._best * (1.0 + self.min_delta) if self._best > 0
                else self._best + abs(self._best) * self.min_delta + 1e-12
            )
            return candidate > threshold
        threshold = (
            self._best * (1.0 - self.min_delta) if self._best > 0
            else self._best - abs(self._best) * self.min_delta - 1e-12
        )
        return candidate < threshold

    def __call__(self, study, trial) -> None:
        if self.stopped:
            return
        completed = [t for t in study.trials if t.value is not None]
        if not completed:
            return

        candidate = study.best_value
        if self._improved(candidate):
            self._best = candidate
            self._best_trial_number = trial.number
            self._trials_since_improvement = 0
        else:
            self._trials_since_improvement += 1

        n_done = len(completed)
        observability.emit(
            observability.current_run(), "tuning_trial",
            model=self.model_name, trial_number=trial.number, value=trial.value,
            best_value=study.best_value, trials_since_improvement=self._trials_since_improvement,
        )

        if n_done >= self.min_trials and self._trials_since_improvement >= self.patience:
            log = setup_logging()
            self.stopped = True
            self.trials_skipped = max(0, self.n_trials_requested - n_done)
            self.stop_reason = (
                f"no >= {self.min_delta:.2%} relative improvement in {self.patience} "
                f"consecutive trials (best={self._best!r} at trial {self._best_trial_number})"
            )
            log.info(
                "[%s tuning] Early-stopping after %d/%d trials: %s. Skipping %d remaining trial(s).",
                self.model_name, n_done, self.n_trials_requested, self.stop_reason, self.trials_skipped,
            )
            observability.check(
                name=f"{self.model_name}.tuning_early_stopped", category="tuning",
                definition="Optuna study stopped before n_trials because further trials "
                            "were judged marginal under the patience-over-trials rule.",
                expected=(
                    f"stops only once >= {self.min_trials} trials ran and "
                    f">= {self.patience} of the most recent had < {self.min_delta:.2%} relative improvement"
                ),
                severity="info", passed=True,
                observed={
                    "trials_run": n_done, "trials_requested": self.n_trials_requested,
                    "trials_skipped": self.trials_skipped, "best_value": self._best,
                    "best_trial_number": self._best_trial_number,
                    "patience": self.patience, "min_delta": self.min_delta,
                },
                evidence=f"optuna study, model={self.model_name}",
            )
            study.stop()
