"""Acceptance tests for live progress in the diagnostic suite: which function is
running, for how long, and how far along each test is.

The path under test, end to end::

    ifvae_diag.progress (steps + tqdm bars, in the vendored suite)
      -> _forward_suite_event  (src/evaluation/ifvae_diagnostic.py)
      -> phase events / progress events  (logging_config, observability)
      -> console dashboard (rich)  and  live flow view (/state, browser)

Everything drives real code: the bridge tests run the real ``diagnose_frames``
and a real (tiny) Isolation Forest refit; the flow tests build state from real
JSONL written by ``observability`` and fetch it over a real local HTTP server.
Where a test feeds hand-written events (the flow-state edge cases) it says so.

Run: ``python -m pytest tests/ -q``
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.request
from contextlib import redirect_stderr

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ifvae_diag import progress as suite_progress  # noqa: E402
from src.evaluation import ifvae_diagnostic as bridge  # noqa: E402
from src.reporting.flow_visualization import (  # noqa: E402
    _annotate_live,
    _build_nodes,
    start_live_view,
)
from src.utils import console_ui, observability  # noqa: E402
from src.utils import logging_config  # noqa: E402
from test_diagnostic_section import synthetic_frames  # noqa: E402


class _Recorder:
    """Registers as phase + progress observer and keeps what it hears."""

    def __enter__(self):
        self.phases: list[tuple] = []
        self.bars: list[dict] = []
        logging_config.add_phase_observer(self._phase)
        observability.add_progress_observer(self._bar)
        return self

    def __exit__(self, *exc):
        logging_config.remove_phase_observer(self._phase)
        observability.remove_progress_observer(self._bar)

    def _phase(self, name, event, duration_s):
        self.phases.append((name, event, duration_s))

    def _bar(self, fields):
        self.bars.append(fields)

    def started(self) -> list[str]:
        return [n for n, e, _ in self.phases if e == "phase_started"]


class ObservabilityProgressTests(unittest.TestCase):
    def test_progress_event_reaches_observers_and_the_jsonl_stream(self):
        with tempfile.TemporaryDirectory() as tmp, _Recorder() as rec:
            path = os.path.join(tmp, "events.jsonl")
            ctx = observability.start_run({"k": 1}, seed=1, events_path=path)
            try:
                observability.progress_event(
                    "stability_refit[VAEDetector]", 2, 3, unit="refit", state="update",
                    current="seed=2042", elapsed_s=12.3456,
                )
            finally:
                observability.end_run(ctx, "success")
                observability._ACTIVE_RUN.set(None)
            with open(path, encoding="utf-8") as fh:
                events = [json.loads(line) for line in fh]
        written = [e for e in events if e["event"] == "progress"]
        self.assertEqual(len(written), 1)
        self.assertEqual(
            (written[0]["desc"], written[0]["n"], written[0]["total"], written[0]["current"]),
            ("stability_refit[VAEDetector]", 2, 3, "seed=2042"),
        )
        self.assertEqual(written[0]["elapsed_s"], 12.346)
        self.assertEqual(rec.bars[0]["n"], 2)

    def test_events_carry_millisecond_epoch_time_next_to_whole_second_ts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            before = time.time()
            ctx = observability.start_run({}, events_path=path)
            observability.end_run(ctx, "success")
            observability._ACTIVE_RUN.set(None)
            with open(path, encoding="utf-8") as fh:
                first = json.loads(fh.readline())
        self.assertGreaterEqual(first["t"], round(before, 3))
        self.assertLessEqual(first["t"], time.time())
        self.assertIn("ts", first)

    def test_progress_event_without_an_active_run_still_notifies_observers(self):
        with _Recorder() as rec:
            observability.progress_event("no_run", 1, None, state="start")
        self.assertEqual((rec.bars[0]["desc"], rec.bars[0]["total"]), ("no_run", None))

    def test_report_phase_event_notifies_phase_observers(self):
        with _Recorder() as rec:
            logging_config.report_phase_event("ifvae_diag.x", "phase_started")
            logging_config.report_phase_event("ifvae_diag.x", "phase_completed", 1.5)
            logging_config.report_phase_event("ifvae_diag.y", "phase_failed", 0.25)
        self.assertEqual(rec.phases, [
            ("ifvae_diag.x", "phase_started", None),
            ("ifvae_diag.x", "phase_completed", 1.5),
            ("ifvae_diag.y", "phase_failed", 0.25),
        ])


class SuiteBridgeTests(unittest.TestCase):
    def test_diagnose_frames_reports_every_function_and_finishes_its_bar(self):
        reference, scored, features = synthetic_frames()
        with tempfile.TemporaryDirectory() as tmp, _Recorder() as rec:
            bridge.diagnose_frames(reference, scored, features, tmp)
        started = rec.started()
        for expected in (
            "ifvae_diag.validate_frames", "ifvae_diag.compare_populations",
            "ifvae_diag.write_outputs", "ifvae_diagnostic._build_agreement",
            "ifvae_diagnostic._build_stability", "ifvae_diagnostic.build_interpretation_contract",
        ):
            self.assertIn(expected, started)
        self.assertLess(
            started.index("ifvae_diag.validate_frames"),
            started.index("ifvae_diagnostic.build_interpretation_contract"),
        )
        finished = {n: d for n, e, d in rec.phases if e == "phase_completed"}
        self.assertEqual(set(started), set(finished))  # nothing left running
        self.assertTrue(all(d >= 0 for d in finished.values()))
        overall = [b for b in rec.bars if b["desc"] == "ifvae_diagnostic"]
        self.assertEqual((overall[-1]["state"], overall[-1]["n"], overall[-1]["total"]),
                         ("end", 11, 11))
        self.assertEqual(overall[-1]["unit"], "prueba")

    def test_a_failing_test_is_reported_failed_and_the_wiring_is_removed(self):
        reference, scored, features = synthetic_frames()
        scored.loc[scored.index[0], features[0]] = np.inf
        with tempfile.TemporaryDirectory() as tmp, _Recorder() as rec:
            with self.assertRaises(Exception):
                bridge.diagnose_frames(reference, scored, features, tmp)
        self.assertIn("phase_failed", [e for _, e, _ in rec.phases])
        self.assertNotIn(bridge._forward_suite_event, suite_progress._OBSERVERS)
        self.assertEqual(bridge._BRIDGE_DEPTH, 0)

    def test_seeded_refits_are_tracked_one_step_and_one_bar_tick_per_seed(self):
        from src.models import IsolationForestDetector

        rng = np.random.default_rng(3)
        x_fit = rng.normal(size=(60, 4))
        x_score = rng.normal(size=(40, 4))
        base = IsolationForestDetector(
            random_state=1, n_estimators=20, contamination=0.1,
            max_samples="auto", max_features=1.0, bootstrap=False,
        )
        base.fit(x_fit)
        seeds = (1001, 2001, 3001)
        # These refits take ~0.06s each, well under the 0.5s update throttle
        # that keeps a JSONL log from growing per iteration; lift it so every
        # seed's update is observable.
        suite_progress.configure(min_interval_s=0.0)
        try:
            with bridge._suite_progress(), _Recorder() as rec:
                bridge._seeded_refit_stability(
                    IsolationForestDetector, base,
                    ("n_estimators", "max_samples", "max_features", "contamination", "bootstrap"),
                    x_fit, x_score, seeds, k=10,
                )
        finally:
            suite_progress.configure(min_interval_s=suite_progress.DEFAULT_MIN_INTERVAL_S)
        refits = [n for n in rec.started() if "refit[" in n]
        self.assertEqual(refits, [
            f"ifvae_diagnostic.refit[IsolationForestDetector seed={s}]" for s in seeds
        ])
        bar = [b for b in rec.bars if b["desc"] == "stability_refit[IsolationForestDetector]"]
        self.assertEqual([b["n"] for b in bar if b["state"] == "end"], [3])
        self.assertIn("seed=2001", {b["current"] for b in bar})

    def test_tqdm_is_left_to_the_dashboard_only_while_a_live_dashboard_is_up(self):
        stream = io.StringIO()
        with redirect_stderr(stream), bridge._suite_progress() as progress:
            list(progress.track(range(3), desc="visible_bar"))
        self.assertIn("visible_bar", stream.getvalue())  # no dashboard -> real tqdm bars

        previous = console_ui._ACTIVE
        console_ui._ACTIVE = console_ui.ConsoleUI()  # constructed, never started
        try:
            stream = io.StringIO()
            with redirect_stderr(stream), bridge._suite_progress() as progress:
                list(progress.track(range(3), desc="hidden_bar"))
            self.assertEqual(stream.getvalue(), "")
        finally:
            console_ui._ACTIVE = previous
        self.assertTrue(suite_progress._SETTINGS["tqdm_enabled"])  # restored afterwards


class FlowStateTests(unittest.TestCase):
    """Hand-written events -- these pin the state machine's edge cases."""

    RUN = {"event": "run_started", "run_id": "r1", "ts": "2026-09-23T10:00:00", "t": 1000.0}

    def _events(self, *rest):
        return [self.RUN, *rest]

    @staticmethod
    def _phase(event, name, t, duration=None):
        record = {"event": event, "run_id": "r1", "phase": name, "t": t, "ts": "2026-09-23T10:00:00"}
        if duration is not None:
            record["duration_s"] = duration
        return record

    @staticmethod
    def _bar(desc, n, total, t, state="update", current="", elapsed=0.0):
        return {"event": "progress", "run_id": "r1", "desc": desc, "n": n, "total": total,
                "unit": "refit", "state": state, "current": current, "elapsed_s": elapsed, "t": t}

    def test_running_functions_form_a_stack_with_their_own_start_times(self):
        data = _build_nodes(self._events(
            self._phase("phase_started", "Phase 9c: IF-VAE diagnostic suite", 1001.0),
            self._phase("phase_started", "ifvae_diagnostic._build_stability", 1002.0),
            self._phase("phase_started", "ifvae_diagnostic.refit[VAEDetector seed=1]", 1010.0),
        ))
        data = _annotate_live(data, now=1025.0)
        stack = data["live"]["running_functions"]
        self.assertEqual([f["name"] for f in stack], [
            "ifvae_diagnostic._build_stability", "ifvae_diagnostic.refit[VAEDetector seed=1]",
        ])
        self.assertEqual([f["elapsed_s"] for f in stack], [23.0, 15.0])
        self.assertEqual(data["nodes"][0]["elapsed_s"], 24.0)

    def test_a_finished_function_leaves_the_stack_and_joins_the_recent_list(self):
        data = _build_nodes(self._events(
            self._phase("phase_started", "Phase 9c: x", 1001.0),
            self._phase("phase_started", "a", 1002.0),
            self._phase("phase_completed", "a", 1004.0, duration=2.0),
            self._phase("phase_started", "b", 1005.0),
            self._phase("phase_failed", "b", 1006.0, duration=1.0),
        ))
        live = data["live"]
        self.assertEqual(live["running_functions"], [])
        self.assertEqual(
            [(f["name"], f["status"], f["duration_s"]) for f in live["recent_functions"]],
            [("a", "completed", 2.0), ("b", "failed", 1.0)],
        )

    def test_recent_list_is_bounded(self):
        events = []
        for i in range(25):
            events.append(self._phase("phase_started", f"f{i}", 1001.0 + i))
            events.append(self._phase("phase_completed", f"f{i}", 1001.5 + i, duration=0.5))
        recent = _build_nodes(self._events(*events))["live"]["recent_functions"]
        self.assertEqual(len(recent), 10)
        self.assertEqual(recent[-1]["name"], "f24")

    def test_a_running_bar_keeps_counting_time_between_events(self):
        data = _build_nodes(self._events(
            self._bar("stability_refit[VAEDetector]", 1, 3, t=1100.0, current="seed=2", elapsed=40.0),
        ))
        bar = _annotate_live(data, now=1130.0)["live"]["progress"][0]
        self.assertEqual((bar["n"], bar["total"], bar["current"]), (1, 3, "seed=2"))
        self.assertEqual(bar["elapsed_now_s"], 70.0)  # 40s at the event + 30s since

    def test_a_finished_bar_stops_its_clock(self):
        data = _build_nodes(self._events(
            self._bar("done_bar", 3, 3, t=1100.0, state="end", elapsed=55.0),
        ))
        bar = _annotate_live(data, now=9999.0)["live"]["progress"][0]
        self.assertEqual(bar["elapsed_now_s"], 55.0)

    def test_after_the_run_ends_nothing_is_shown_as_still_running(self):
        data = _build_nodes(self._events(
            self._phase("phase_started", "Phase 9c: x", 1001.0),
            self._phase("phase_started", "a", 1002.0),
            self._bar("bar", 1, 3, t=1003.0),
            {"event": "run_ended", "run_id": "r1", "status": "cancelled", "t": 1004.0,
             "ts": "2026-09-23T10:00:04"},
        ))
        self.assertEqual(data["live"]["running_functions"], [])
        self.assertEqual([b["state"] for b in data["live"]["progress"]], ["end"])

    def test_older_event_files_without_t_fall_back_to_whole_second_ts(self):
        old = {"event": "phase_started", "run_id": "r1", "phase": "legacy",
               "ts": "2026-09-23T10:00:00"}
        data = _build_nodes([{"event": "run_started", "run_id": "r1",
                              "ts": "2026-09-23T10:00:00"}, old])
        expected = time.mktime(time.strptime("2026-09-23T10:00:00", "%Y-%m-%dT%H:%M:%S"))
        self.assertEqual(data["live"]["running_functions"][0]["started_epoch"], expected)


class FlowPhaseBarsTests(unittest.TestCase):
    """Every phase's own node carries the bars that ran inside it, in both the
    live page and the static post-run HTML."""

    def _events(self):
        base = {"run_id": "r1", "ts": "2026-09-23T10:00:00"}
        bar = lambda desc, n, total, t, state="update", current="": {  # noqa: E731
            **base, "event": "progress", "desc": desc, "n": n, "total": total, "unit": "epoch",
            "state": state, "current": current, "elapsed_s": 3.0, "t": t}
        return [
            {**base, "event": "run_started", "t": 1000.0},
            {**base, "event": "phase_started", "phase": "Phase 7: VAE", "t": 1001.0},
            bar("vae[epochs]", 4, 10, 1003.0, current="train=0.5"),
            {**base, "event": "phase_completed", "phase": "Phase 7: VAE", "t": 1010.0, "duration_s": 9.0},
            {**base, "event": "phase_started", "phase": "Phase 8: evaluation [vae]", "t": 1011.0},
            bar("rank_stability[bootstraps]", 10, 10, 1012.0, state="end"),
            {**base, "event": "phase_completed", "phase": "Phase 8: evaluation [vae]", "t": 1013.0,
             "duration_s": 2.0},
        ]

    def test_each_phase_node_holds_only_the_bars_that_ran_inside_it(self):
        nodes = _build_nodes(self._events())["nodes"]
        self.assertEqual([[b["desc"] for b in n["progress"]] for n in nodes],
                         [["vae[epochs]"], ["rank_stability[bootstraps]"]])
        self.assertEqual((nodes[0]["progress"][0]["n"], nodes[0]["progress"][0]["total"]), (4, 10))

    def test_a_bar_still_open_when_its_run_is_over_is_shown_as_ended_not_running(self):
        events = self._events() + [{"event": "run_ended", "run_id": "r1", "status": "cancelled",
                                    "t": 1020.0, "ts": "2026-09-23T10:00:20"}]
        nodes = _build_nodes(events)["nodes"]
        self.assertEqual(nodes[0]["progress"][0]["state"], "end")
        self.assertEqual(nodes[0]["progress"][0]["n"], 4)  # partial count kept: it did not finish

    def test_a_running_phase_bar_keeps_counting_time_in_the_live_state(self):
        events = self._events()[:3]  # Phase 7 open, its bar mid-way
        data = _annotate_live(_build_nodes(events), now=1013.0)
        bar = data["nodes"][0]["progress"][0]
        self.assertEqual(bar["elapsed_now_s"], 13.0)  # 3s at the event + 10s since

    def test_static_html_lists_each_phase_bars_and_the_live_page_draws_them_on_the_card(self):
        from src.reporting.flow_visualization import _LIVE_HTML, _render_html

        html = _render_html(_build_nodes(self._events()))
        self.assertIn("Progress bars run during this phase", html)
        self.assertIn('"desc": "vae[epochs]"', html)  # the payload the page renders from
        self.assertIn("miniBars(n)", _LIVE_HTML)


class LiveViewServerTests(unittest.TestCase):
    def test_state_endpoint_serves_running_function_and_bar_from_real_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            ctx = observability.start_run({}, events_path=path)
            try:
                url = start_live_view(events_path=path)
                logging_config.report_phase_event("ifvae_diagnostic._build_stability", "phase_started")
                observability.progress_event(
                    "stability_refit[VAEDetector]", 1, 3, unit="refit", current="seed=2",
                    elapsed_s=5.0,
                )
                time.sleep(0.05)
                with urllib.request.urlopen(url + "state", timeout=5) as resp:
                    state = json.loads(resp.read())
                with urllib.request.urlopen(url, timeout=5) as resp:
                    page = resp.read().decode("utf-8")
            finally:
                observability.end_run(ctx, "success")
                observability._ACTIVE_RUN.set(None)
        stack = state["live"]["running_functions"]
        self.assertEqual(stack[0]["name"], "ifvae_diagnostic._build_stability")
        self.assertGreaterEqual(stack[0]["elapsed_s"], 0.04)
        self.assertLess(stack[0]["elapsed_s"], 30)
        bar = state["live"]["progress"][0]
        self.assertEqual((bar["n"], bar["total"], bar["current"]), (1, 3, "seed=2"))
        self.assertGreaterEqual(bar["elapsed_now_s"], 5.0)
        self.assertIn("server_now", state)
        # The page ships the panels that render this state.
        for element_id in ("live-functions", "live-bars", "live-recent"):
            self.assertIn(f'id="{element_id}"', page)


class ConsoleDashboardTests(unittest.TestCase):
    def _render_text(self, ui) -> str:
        from rich.console import Console

        ui._console = Console(width=150, record=True, force_terminal=False)
        ui._console.print(ui._render())
        return ui._console.export_text()

    def test_dashboard_shows_running_functions_with_elapsed_and_tqdm_style_bars(self):
        ui = console_ui.ConsoleUI()
        ui._on_phase("Phase 9c: IF-VAE diagnostic suite", "phase_started", None)
        ui._on_phase("ifvae_diagnostic._build_stability", "phase_started", None)
        ui._on_phase("ifvae_diagnostic.refit[VAEDetector seed=1042]", "phase_started", None)
        ui._on_progress({
            "desc": "stability_refit[VAEDetector]", "n": 1, "total": 3, "unit": "refit",
            "state": "update", "current": "seed=1042", "elapsed_s": 41.0,
        })
        text = self._render_text(ui)
        self.assertIn("↳ función", text)
        self.assertIn("ifvae_diagnostic._build_stability", text)
        self.assertIn("ifvae_diagnostic.refit[VAEDetector seed=1042]", text)
        self.assertIn("stability_refit[VAEDetector]", text)
        self.assertIn("1/3", text)
        self.assertIn("33%", text)
        self.assertIn("seed=1042", text)
        self.assertIn("00:4", text)  # a bar 41s old (+ a hair) renders as 00:41..00:4x

    def test_finished_bars_and_functions_disappear_from_the_live_lines(self):
        ui = console_ui.ConsoleUI()
        ui._on_phase("Phase 9c: x", "phase_started", None)
        ui._on_phase("ifvae_diag.compare_populations", "phase_started", None)
        ui._on_progress({"desc": "compare_populations[features]", "n": 8, "total": 8,
                         "unit": "feature", "state": "end", "current": "", "elapsed_s": 2.0})
        ui._on_phase("ifvae_diag.compare_populations", "phase_completed", 2.0)
        text = self._render_text(ui)
        self.assertNotIn("↳ función", text)
        self.assertNotIn("compare_populations[features]", text)

    def test_nested_functions_do_not_inflate_the_phase_counter(self):
        ui = console_ui.ConsoleUI()
        ui._on_phase("Phase 9c: IF-VAE diagnostic suite", "phase_started", None)
        for i in range(30):
            ui._on_phase(f"ifvae_diag.step{i}", "phase_started", None)
            ui._on_phase(f"ifvae_diag.step{i}", "phase_completed", 0.1)
        ui._on_phase("Phase 9c: IF-VAE diagnostic suite", "phase_completed", 5.0)
        _fraction, n_done, n_total = ui._progress_fraction()
        self.assertEqual((n_done, n_total), (1, len(console_ui._PHASE_PLAN)))

    def test_is_live_is_false_for_the_silent_stand_in(self):
        previous = console_ui._ACTIVE
        try:
            console_ui._ACTIVE = console_ui._NullUI()
            self.assertFalse(console_ui.is_live())
            console_ui._ACTIVE = None
            self.assertFalse(console_ui.is_live())
        finally:
            console_ui._ACTIVE = previous

    def test_format_bar_matches_tqdm_layout(self):
        text = console_ui._format_bar(
            {"desc": "d", "n": 2, "total": 4, "unit": "refit", "current": "seed=7"}, 10.0, 90,
        )
        self.assertTrue(text.startswith("d:  50%|"))
        self.assertIn("2/4", text)
        self.assertIn("seed=7", text)


if __name__ == "__main__":
    unittest.main()
