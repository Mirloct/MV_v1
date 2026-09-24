"""Behavior: every long loop in the pipeline reports progress the same way --
a tqdm bar on the terminal AND ``progress`` events for the console dashboard
and the live/static flow HTML -- without changing what the loop computes.

Before this, only the diagnostic suite (Phase 9c) was followed live; the VAE's
epochs, both Optuna studies and the permutation-importance loop drew bare tqdm
bars that a repainting dashboard tears apart and that the flow page never saw.

Run: ``python -m pytest tests/ -q``
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager, redirect_stderr

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import console_ui, observability, progress  # noqa: E402


@contextmanager
def recorded(min_interval_s: float = 0.0):
    events: list[dict] = []
    observability.add_progress_observer(events.append)
    progress.configure(min_interval_s=min_interval_s)
    try:
        yield events
    finally:
        observability.remove_progress_observer(events.append)
        progress.configure(min_interval_s=progress.DEFAULT_MIN_INTERVAL_S)


@contextmanager
def live_dashboard_owns_terminal():
    previous = console_ui._ACTIVE
    console_ui._ACTIVE = console_ui.ConsoleUI()  # constructed, never started
    try:
        yield
    finally:
        console_ui._ACTIVE = previous


class BarIterationTests(unittest.TestCase):
    def test_iterating_yields_every_item_and_reports_start_updates_end(self):
        with recorded() as events, live_dashboard_owns_terminal():
            seen = list(progress.Bar([5, 6, 7], desc="loop", unit="item"))
        self.assertEqual(seen, [5, 6, 7])
        self.assertEqual([e["state"] for e in events], ["start", "update", "update", "update", "end"])
        self.assertEqual([e["n"] for e in events], [0, 1, 2, 3, 3])
        self.assertTrue(all(e["total"] == 3 and e["desc"] == "loop" for e in events))
        self.assertEqual(events[-1]["unit"], "item")

    def test_an_item_counts_only_after_the_caller_finished_it(self):
        with recorded() as events, live_dashboard_owns_terminal():
            for item in progress.Bar(range(3), desc="work"):
                self.assertEqual(events[-1]["n"], item)

    def test_initial_offsets_the_count_for_resumed_work(self):
        with recorded() as events, live_dashboard_owns_terminal():
            list(progress.Bar(range(3, 5), desc="resumed", total=5, initial=3))
        self.assertEqual((events[0]["n"], events[0]["total"]), (3, 5))
        self.assertEqual((events[-1]["state"], events[-1]["n"]), ("end", 5))

    def test_stopping_early_closes_the_bar_once_with_the_finished_count(self):
        with recorded() as events, live_dashboard_owns_terminal():
            bar = progress.Bar(range(10), desc="early", unit="epoch")
            for item in bar:
                if item == 2:
                    break
            bar.close()  # callers also close explicitly; must not double-report
        ends = [e for e in events if e["state"] == "end"]
        self.assertEqual([(e["n"], e["total"]) for e in ends], [(2, 10)])

    def test_label_publishes_the_item_in_flight(self):
        seen = []
        with recorded() as events, live_dashboard_owns_terminal():
            for _ in progress.Bar([11, 22], desc="fits", label=lambda s: f"seed={s}"):
                seen.append((events[-1]["current"], events[-1]["n"]))
        self.assertEqual(seen, [("seed=11", 0), ("seed=22", 1)])


class ManualBarTests(unittest.TestCase):
    def test_update_and_set_postfix_drive_a_bar_without_an_iterable(self):
        with recorded() as events, live_dashboard_owns_terminal():
            bar = progress.Bar(desc="optuna[study]", total=3, unit="trial")
            bar.update(1)
            bar.set_postfix(best=0.12345678, note="ok")
            bar.update(1)
            bar.close()
        self.assertEqual([e["n"] for e in events if e["state"] == "update"][-1], 2)
        self.assertIn("best=0.1235", events[-2]["current"])
        self.assertIn("note=ok", events[-2]["current"])
        self.assertEqual((events[-1]["state"], events[-1]["n"]), ("end", 2))

    def test_context_manager_closes_the_bar(self):
        with recorded() as events, live_dashboard_owns_terminal():
            with progress.Bar(desc="ctx", total=1) as bar:
                bar.update(1)
        self.assertEqual(events[-1]["state"], "end")

    def test_a_bar_closed_by_an_exception_still_reports_its_end(self):
        with recorded() as events, live_dashboard_owns_terminal():
            with self.assertRaises(RuntimeError):
                with progress.Bar(desc="boom", total=5) as bar:
                    bar.update(1)
                    raise RuntimeError("x")
        self.assertEqual((events[-1]["state"], events[-1]["n"]), ("end", 1))


class ThrottleAndElapsedTests(unittest.TestCase):
    def test_updates_are_throttled_but_start_and_end_always_fire(self):
        with recorded(min_interval_s=3600.0) as events, live_dashboard_owns_terminal():
            list(progress.Bar(range(400), desc="fast"))
        self.assertEqual([e["state"] for e in events], ["start", "end"])
        self.assertEqual(events[-1]["n"], 400)

    def test_elapsed_time_grows_across_events(self):
        with recorded() as events, live_dashboard_owns_terminal():
            for _ in progress.Bar(range(2), desc="slow"):
                time.sleep(0.03)
        elapsed = [e["elapsed_s"] for e in events]
        self.assertEqual(elapsed, sorted(elapsed))
        self.assertGreaterEqual(elapsed[-1], 0.06)


class TqdmOwnershipTests(unittest.TestCase):
    def test_real_tqdm_bars_are_drawn_when_no_dashboard_owns_the_terminal(self):
        stream = io.StringIO()
        with recorded(), redirect_stderr(stream):
            list(progress.Bar(range(4), desc="vae[epochs]", unit="epoch"))
        text = stream.getvalue()
        self.assertIn("vae[epochs]", text)
        self.assertIn("4/4", text)

    def test_tqdm_stays_silent_while_a_live_dashboard_owns_the_terminal(self):
        stream = io.StringIO()
        with recorded(), live_dashboard_owns_terminal(), redirect_stderr(stream):
            list(progress.Bar(range(4), desc="vae[epochs]"))
        self.assertEqual(stream.getvalue(), "")


class RealCallSiteTests(unittest.TestCase):
    """The loops that used bare tqdm now report through this module."""

    def test_vae_fit_reports_one_tick_per_epoch(self):
        from src.models import VAEDetector

        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 4))
        with tempfile.TemporaryDirectory() as tmp, recorded() as events, live_dashboard_owns_terminal():
            VAEDetector(random_state=1, epochs=3, latent_dim=2, hidden_dim=8, n_layers=1,
                        batch_size=32, early_stopping_patience=None).fit(
                x, checkpoint_dir=tmp, resume=False)
        bar = [e for e in events if e["desc"] == "vae[epochs]"]
        self.assertEqual((bar[0]["state"], bar[0]["total"], bar[0]["unit"]), ("start", 3, "epoch"))
        self.assertEqual((bar[-1]["state"], bar[-1]["n"]), ("end", 3))
        self.assertTrue(any("train=" in e["current"] for e in bar))  # loss shown as postfix

    def test_iforest_tuning_reports_one_tick_per_trial_with_the_best_value(self):
        from src.models import tune_iforest

        rng = np.random.default_rng(5)
        x = rng.normal(size=(200, 4))
        with tempfile.TemporaryDirectory() as tmp, recorded() as events, live_dashboard_owns_terminal():
            tune_iforest(
                x, n_trials=2, random_state=1,
                best_params_path=os.path.join(tmp, "best.yaml"),
                model_out=os.path.join(tmp, "iforest.joblib"),
                storage="sqlite:///" + os.path.join(tmp, "study.db").replace("\\", "/"),
                early_stopping_patience=None,
            )
        bar = [e for e in events if e["desc"].startswith("optuna[")]
        self.assertEqual((bar[-1]["state"], bar[-1]["n"], bar[-1]["total"]), ("end", 2, 2))
        self.assertTrue(any(e["current"].startswith("best=") for e in bar))


if __name__ == "__main__":
    unittest.main()
