"""Behavior: the suite reports live progress -- named steps with their own
elapsed time, and tqdm bars for every loop-shaped test -- without changing a
single computed value.

Why this exists: a full run (stability refits, experiment sweeps) can take
minutes with nothing on screen. The operator needs to see WHICH function is
running, HOW LONG it has been running, and how far along each test is. The
suite stays host-agnostic: it emits plain-dict events to registered observers
(the host pipeline forwards them to its dashboard / live flow view) and draws
tqdm bars itself when nothing else owns the terminal.

These tests assert on concrete event content and on real tqdm output, and the
last one drives the real ``run_diagnostic`` to prove the instrumentation is
wired into the actual code path rather than only into the helper.
"""

import io
import tempfile
import time
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path

from ifvae_diag import progress
from ifvae_diag.pipeline import run_diagnostic
from ifvae_diag.simulation import generate_synthetic_case


@contextmanager
def recorded_events(*, tqdm_enabled: bool = False, min_interval_s: float = 0.0):
    events: list[dict] = []
    progress.add_observer(events.append)
    progress.configure(tqdm_enabled=tqdm_enabled, min_interval_s=min_interval_s)
    try:
        yield events
    finally:
        progress.remove_observer(events.append)
        progress.configure(tqdm_enabled=True, min_interval_s=progress.DEFAULT_MIN_INTERVAL_S)


def _kinds(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["event"] == kind]


class StepTests(unittest.TestCase):
    def test_step_reports_start_then_completion_with_measured_duration(self):
        with recorded_events() as events:
            with progress.step("compare_populations", label="Population drift"):
                time.sleep(0.05)
        self.assertEqual([e["event"] for e in events], ["step_started", "step_completed"])
        self.assertEqual({e["name"] for e in events}, {"compare_populations"})
        self.assertEqual(events[0]["label"], "Population drift")
        self.assertGreaterEqual(events[1]["duration_s"], 0.05)
        self.assertLess(events[1]["duration_s"], 5.0)

    def test_nested_steps_report_their_depth(self):
        with recorded_events() as events:
            with progress.step("outer"):
                with progress.step("inner"):
                    pass
        depth = {e["name"]: e["depth"] for e in _kinds(events, "step_started")}
        self.assertEqual(depth, {"outer": 0, "inner": 1})

    def test_failed_step_reports_failure_and_reraises(self):
        with recorded_events() as events:
            with self.assertRaises(ZeroDivisionError):
                with progress.step("explodes"):
                    1 / 0
        self.assertEqual([e["event"] for e in events], ["step_started", "step_failed"])
        self.assertIn("ZeroDivisionError", events[1]["error"])
        # The failed step must not stay on the depth stack for later steps.
        with recorded_events() as later:
            with progress.step("after"):
                pass
        self.assertEqual(later[0]["depth"], 0)

    def test_a_raising_observer_is_dropped_and_never_breaks_the_computation(self):
        calls = []

        def broken(event):
            calls.append(event["event"])
            raise RuntimeError("observer bug")

        progress.add_observer(broken)
        try:
            with progress.step("still_runs"):
                result = 21 * 2
        finally:
            progress.remove_observer(broken)
        self.assertEqual(result, 42)
        self.assertEqual(calls, ["step_started"])  # dropped after its first failure


class TrackTests(unittest.TestCase):
    def test_track_yields_every_item_unchanged_and_reports_full_progress(self):
        with recorded_events() as events:
            seen = list(progress.track([10, 20, 30], desc="seeds", unit="seed"))
        self.assertEqual(seen, [10, 20, 30])
        updates = _kinds(events, "progress")
        self.assertEqual([u["state"] for u in updates], ["start", "update", "update", "update", "end"])
        self.assertEqual([u["n"] for u in updates], [0, 1, 2, 3, 3])
        self.assertTrue(all(u["total"] == 3 and u["desc"] == "seeds" for u in updates))
        self.assertEqual(updates[-1]["unit"], "seed")

    def test_track_counts_an_item_only_after_the_caller_finished_it(self):
        with recorded_events() as events:
            for item in progress.track(range(3), desc="work"):
                done_so_far = [u["n"] for u in _kinds(events, "progress")][-1]
                self.assertEqual(done_so_far, item)  # item N running => N completed

    def test_track_closes_the_bar_when_the_caller_stops_early(self):
        with recorded_events() as events:
            for item in progress.track(range(10), desc="early"):
                if item == 2:
                    break
        end = _kinds(events, "progress")[-1]
        self.assertEqual((end["state"], end["n"], end["total"]), ("end", 2, 10))

    def test_track_infers_total_from_sized_iterables_and_tolerates_generators(self):
        with recorded_events() as events:
            list(progress.track(iter(range(2)), desc="generator"))
        self.assertIsNone(_kinds(events, "progress")[0]["total"])
        self.assertEqual(_kinds(events, "progress")[-1]["n"], 2)

    def test_label_is_already_published_while_that_item_is_running(self):
        in_flight = []
        with recorded_events() as events:
            for _ in progress.track([11, 22], desc="fits", label=lambda seed: f"seed={seed}"):
                latest = _kinds(events, "progress")[-1]
                in_flight.append((latest["current"], latest["n"]))
        # A dashboard reading the newest event mid-item sees THAT item, with
        # only the finished ones counted.
        self.assertEqual(in_flight, [("seed=11", 0), ("seed=22", 1)])

    def test_updates_are_throttled_but_start_and_end_are_always_reported(self):
        with recorded_events(min_interval_s=3600.0) as events:
            list(progress.track(range(500), desc="fast"))
        states = [u["state"] for u in _kinds(events, "progress")]
        self.assertEqual(states, ["start", "end"])
        self.assertEqual(_kinds(events, "progress")[-1]["n"], 500)

    def test_elapsed_time_is_reported_on_every_progress_event(self):
        with recorded_events() as events:
            for _ in progress.track(range(2), desc="slow"):
                time.sleep(0.03)
        elapsed = [u["elapsed_s"] for u in _kinds(events, "progress")]
        self.assertEqual(elapsed, sorted(elapsed))
        self.assertGreaterEqual(elapsed[-1], 0.06)


class StagesTests(unittest.TestCase):
    def test_each_stage_is_a_step_and_advances_the_bar_when_it_finishes(self):
        in_flight = []
        with recorded_events() as events:
            with progress.stages("suite", total=2) as stages:
                with stages.stage("first", label="First test"):
                    in_flight.append(_kinds(events, "progress")[-1])
                with stages.stage("second"):
                    in_flight.append(_kinds(events, "progress")[-1])
        self.assertEqual([(p["current"], p["n"]) for p in in_flight], [("first", 0), ("second", 1)])
        self.assertTrue(all(p["total"] == 2 for p in in_flight))
        end = _kinds(events, "progress")[-1]
        self.assertEqual((end["state"], end["n"], end["total"]), ("end", 2, 2))
        self.assertEqual(
            [e["name"] for e in _kinds(events, "step_completed")], ["first", "second"]
        )

    def test_a_failing_stage_does_not_advance_the_bar(self):
        with recorded_events() as events:
            with self.assertRaises(ValueError):
                with progress.stages("suite", total=2) as stages:
                    with stages.stage("boom"):
                        raise ValueError("bad")
        self.assertEqual(max(u["n"] for u in _kinds(events, "progress")), 0)
        self.assertEqual(_kinds(events, "progress")[-1]["state"], "end")
        self.assertEqual(_kinds(events, "step_failed")[0]["name"], "boom")


class TqdmRenderingTests(unittest.TestCase):
    def test_enabled_bars_render_description_and_counts_on_stderr(self):
        stream = io.StringIO()
        with recorded_events(tqdm_enabled=True), redirect_stderr(stream):
            list(progress.track(range(4), desc="stability[vae]", unit="refit"))
        text = stream.getvalue()
        self.assertIn("stability[vae]", text)
        self.assertIn("4/4", text)
        self.assertIn("refit", text)

    def test_disabled_bars_write_nothing_so_a_live_dashboard_is_not_torn(self):
        stream = io.StringIO()
        with recorded_events(tqdm_enabled=False), redirect_stderr(stream):
            list(progress.track(range(4), desc="stability[vae]"))
        self.assertEqual(stream.getvalue(), "")


class RealPipelineWiringTests(unittest.TestCase):
    def test_run_diagnostic_reports_its_named_stages_and_finishes_the_bar(self):
        reference, scored, config = generate_synthetic_case(seed=7)
        with tempfile.TemporaryDirectory() as tmp, recorded_events() as events:
            run_diagnostic(reference, scored, config, Path(tmp))
        completed = [e["name"] for e in _kinds(events, "step_completed")]
        for expected in (
            "validate_frames", "isolation_forest_scores", "compare_populations",
            "write_outputs",
        ):
            self.assertIn(expected, completed)
        self.assertLess(completed.index("validate_frames"), completed.index("write_outputs"))
        overall = [
            u for u in _kinds(events, "progress") if u["desc"] == "run_diagnostic"
        ]
        self.assertEqual(overall[-1]["state"], "end")
        self.assertEqual(overall[-1]["n"], overall[-1]["total"])
        # Loop-shaped tests carry their own bars (drift per feature, IF per seed).
        descs = {u["desc"] for u in _kinds(events, "progress")}
        self.assertIn("compare_populations[features]", descs)
        self.assertIn("isolation_forest[seeds]", descs)
        self.assertIn("write_outputs[files]", descs)


if __name__ == "__main__":
    unittest.main()
