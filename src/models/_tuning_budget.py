"""How many Optuna trials are spent exploring before TPE starts optimising.

Optuna's :class:`~optuna.samplers.TPESampler` draws its first
``n_startup_trials`` **at random** to seed the density model it later optimises
against; the default is ``10``. That default silently assumes a budget much
larger than 10, and this project's default presets are not:

===========================  =======  ==============  ==============
Preset                       Trials   Random (was)    TPE-guided (was)
===========================  =======  ==============  ==============
``--quick``  iForest / VAE    5 / 5    5 / 5           **0 / 0**
default      iForest / VAE   15 / 10   10 / 10         5 / **0**
``--full``   iForest / VAE   50 / 30   10 / 10         40 / 20
===========================  =======  ==============  ==============

So on the *default* preset the VAE's "tuning" was pure random search -- TPE
never engaged once -- and ``--quick`` was random for both models. The study
still returned the best of N random draws (not wrong, just not optimisation),
which is exactly the kind of defect that hides: it produces plausible numbers
and a populated ``best_params_*.yaml``.

:func:`tpe_startup_trials` scales the exploration budget to the trial budget
instead, so a small study still gets guided trials while a large one keeps
Optuna's own behaviour.
"""

from __future__ import annotations

__all__ = ["tpe_startup_trials"]

#: Never explore fewer than this: TPE needs a handful of observations before
#: its density estimate carries any signal, and 1-2 points would make the
#: "guided" trials little better than random anyway.
_MIN_STARTUP = 3
#: Optuna's own default, kept as the ceiling so large studies are unchanged.
_MAX_STARTUP = 10
#: Fraction of the budget spent exploring. One third is the usual
#: explore/exploit split for small budgets, and it keeps the majority of
#: trials for the guided phase, which is the point of using TPE at all.
_EXPLORE_FRACTION = 3


def tpe_startup_trials(n_trials: int) -> int:
    """Random-exploration trials to use for a study of ``n_trials`` total.

    Args:
        n_trials: The study's trial budget.

    Returns:
        ``clamp(n_trials // 3, 3, 10)`` -- e.g. 5 trials -> 3 explore / 2
        guided; 10 -> 3 / 7; 15 -> 5 / 10; 30 -> 10 / 20; 50 -> 10 / 40.
        At budgets of 30+ this is exactly Optuna's default of 10, so tuned
        runs that already worked keep behaving identically.
    """
    n = max(1, int(n_trials))
    return max(_MIN_STARTUP, min(_MAX_STARTUP, n // _EXPLORE_FRACTION))
