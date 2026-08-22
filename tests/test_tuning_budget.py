"""Validation for `src.models._tuning_budget`.

The regression this guards: with Optuna's fixed default of ``n_startup_trials
=10``, the project's default VAE budget of 10 trials produced **zero**
TPE-guided trials -- the study was pure random search while reporting itself
as tuned.
"""

from __future__ import annotations

import pytest

from src.models._tuning_budget import tpe_startup_trials


class TestTpeStartupTrials:
    @pytest.mark.parametrize("n_trials,expected", [
        (5, 3),    # --quick: was 5 random / 0 guided -> now 3 / 2
        (10, 3),   # default VAE: was 10 / 0 -> now 3 / 7
        (15, 5),   # default iForest: was 10 / 5 -> now 5 / 10
        (30, 10),  # --full VAE: unchanged at Optuna's default
        (50, 10),  # --full iForest: unchanged
    ])
    def test_budget_split_for_each_preset(self, n_trials, expected):
        assert tpe_startup_trials(n_trials) == expected

    @pytest.mark.parametrize("n_trials", [5, 10, 15, 30, 50])
    def test_every_preset_gets_at_least_one_guided_trial(self, n_trials):
        """The whole point: no budget may be spent entirely on exploration."""
        assert tpe_startup_trials(n_trials) < n_trials

    def test_large_budgets_keep_optunas_own_default(self):
        """Runs that already worked must not change behaviour."""
        for n in (30, 60, 200, 1000):
            assert tpe_startup_trials(n) == 10

    def test_never_below_the_floor(self):
        """Below ~3 observations a TPE density estimate carries no signal."""
        for n in (1, 2, 3, 4, 6):
            assert tpe_startup_trials(n) >= 3

    def test_is_monotone_non_decreasing(self):
        vals = [tpe_startup_trials(n) for n in range(1, 200)]
        assert vals == sorted(vals)

    @pytest.mark.parametrize("bad", [0, -1, -50])
    def test_degenerate_input_does_not_raise(self, bad):
        assert tpe_startup_trials(bad) == 3


class TestSamplersUseIt:
    """Both tuners must actually wire the helper in, not just import it."""

    @pytest.mark.parametrize("module,func", [
        ("src.models.iforest", "tune_iforest"),
        ("src.models.vae", "tune_vae"),
    ])
    def test_tuner_source_references_the_helper(self, module, func):
        import importlib
        import inspect

        src = inspect.getsource(getattr(importlib.import_module(module), func))
        assert "tpe_startup_trials" in src, (
            f"{func} does not scale n_startup_trials to the trial budget"
        )
        assert "n_startup_trials" in src
