"""Live terminal dashboard for a pipeline run.

Replaces the scrolling wall of log lines with a fixed dashboard that answers,
at a glance: *where am I, how long has it taken, what has the run learned so
far, and is anything wrong.*

It attaches to :func:`src.utils.logging_config.log_phase` through the phase
observer hook, so none of the ~15 ``with log_phase(...)`` call sites in
``main.py`` know it exists, and the pipeline runs identically with the
dashboard off.

Layout::

    ┌ header ─────────────────────────────────────────────┐  run id, elapsed,
    ├ progress + current phase ─────────────────────────── ┤  live-view URL,
    ├ ↳ interpretability sub-step (Phase 10 only) ─────────┤  a resource
    ├ equipo (RAM proceso/sistema, CPU) ────────────────────┤  reading every
    ├ checklist          │ supuestos ─────────────────────┤  ~1s -- most
    │ (all phases,       │ (assumption/training checks,   │  useful during
    │  fixed, boxes light│  pass/fail tally + recent)      │  training /
    │  up as they run)   ├ datos de la corrida ────────────┤  interpretability
    ├───────────────────┴─────────────────────────────────┤
    └ log tail ───────────────────────────────────────────┘

The interpretability sub-step line is a second, finer-grained progress
readout beneath the phase-level one: the phase checklist can only say "Phase
10 is running," but Phase 10 is exactly the phase most prone to stalling
(SHAP over the forest, a hard-kill-guarded subprocess -- see
`src/interpretability/iforest_explain.py`), so its own `interpretability.*`
checkpoints (routed to a separate deque from the assumption checks below, not
mixed into "supuestos") get a dedicated line showing the last few reached and
how long it has sat on the latest one -- a frozen sub-step with a growing
timer is a stall; an unmoving top-level progress bar alone cannot say that.

The phase checklist is the full, fixed plan from the very first frame --
nothing scrolls or appends into view. Each row starts pending (dim, empty
box) and switches in place to running / done / failed as the pipeline
actually gets there, so the whole run's shape is visible before it starts.

Degrading gracefully is a hard requirement -- this is decoration around a
pipeline whose output is the real product:

* no ``rich`` installed -> no-op, plain logging continues;
* output piped / redirected / not a TTY (CI, ``> run.log``) -> no-op, because
  a repainting Live display writes ANSI escapes that corrupt a log file;
* ``--no-console-ui`` -> no-op;
* any error inside the dashboard -> the dashboard is torn down and the run
  continues on plain logging.

Interactivity (only when stdin is a TTY):

===== ==================================================================
Key   Action
===== ==================================================================
``v`` Toggle the log tail between compact (5 lines) and verbose (16)
``o`` Open the live flow view in a browser
``p`` Freeze the display so it can be read; any key resumes
===== ==================================================================

Every key is read-only with respect to the pipeline: nothing here can cancel,
skip or alter a phase. Cancelling stays ``Ctrl-C``, which ``main.py`` already
handles by closing the run with a terminating event.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import webbrowser
from collections import deque
from typing import Optional

from src.utils.logging_config import (
    DEFAULT_LOGGER_NAME,
    add_phase_observer,
    remove_phase_observer,
)
from src.utils.observability import add_check_observer, remove_check_observer
from src.utils import resource_monitor

__all__ = [
    "ConsoleUI", "console_dashboard", "supports_dashboard",
    "start_dashboard", "stop_dashboard", "set_stat", "active",
]

#: The phases `main.py` runs, in order, with a short human label and a rough
#: relative cost. Weights only shape the progress bar's pacing -- a run whose
#: tuning is disabled finishes the heavy phases quickly and the bar catches up.
#: Unknown phases (a new one, or a per-model repeat like "[vae]") are appended
#: as they appear, so this list never needs to be exhaustive to stay correct.
_PHASE_PLAN: tuple[tuple[str, str, float], ...] = (
    ("Phase 2",  "Carga / generación de datos",       1.0),
    ("Phase 3a", "División cronológica",              0.3),
    ("Phase 3b", "Validación de supuestos",           0.3),
    ("Phase 3",  "Etiquetas de verdad base",          0.3),
    ("Phase 4",  "Preprocesamiento",                  2.0),
    ("Phase 5",  "Verificación de la división",       0.3),
    ("Phase 6",  "Isolation Forest",                  3.0),
    ("Phase 6b", "Stacking IF -> VAE",                0.5),
    ("Phase 6c", "Checkpoint P95 del IF",             0.3),
    ("Phase 7",  "VAE",                               4.0),
    ("Phase 8",  "Evaluación",                        0.5),
    ("Phase 8b", "Calibración del umbral",            0.3),
    ("Phase 9",  "Entregable Excel OOT",               0.5),
    ("Phase 10", "Interpretabilidad",                 2.0),
    ("Phase 11", "Reporte",                           1.0),
)

#: Checklist box per phase status. ``pending`` is an empty box (dim) so the
#: full plan reads as a checklist from the very first frame -- every row is
#: visible before the run reaches it, and only the box itself changes as
#: progress advances (nothing appends or scrolls into view).
_STATUS_STYLE = {
    "pending": ("[dim]□[/dim]", "dim"),
    "running": ("[cyan]▣[/cyan]", "cyan"),
    "done": ("[green]■[/green]", "green"),
    "failed": ("[red]■[/red]", "red"),
}
#: Same box glyphs, for the assumption/training check list (pass/fail only).
_CHECK_STYLE = {
    True: ("[green]✓[/green]", "green"),
    False: ("[red]✗[/red]", "red"),
}


def supports_dashboard(stream=None) -> bool:
    """True when a repainting dashboard is safe on ``stream`` (default stdout).

    Requires ``rich`` and a real terminal. Writing ANSI repaints into a pipe
    or a redirected file produces unreadable output, which is worse than the
    plain logging it replaced.
    """
    stream = stream or sys.stdout
    try:
        import rich  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("MODELO_NO_CONSOLE_UI"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


class _TailHandler(logging.Handler):
    """Collects formatted log lines into a bounded deque for the tail panel."""

    def __init__(self, buffer: deque, level=logging.INFO):
        super().__init__(level=level)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append((record.levelno, record.getMessage()))
        except Exception:  # noqa: BLE001 - a log line must never raise
            pass


class _KeyReader:
    """Non-blocking single-key reader; a no-op where stdin is not a TTY.

    Implemented per-platform rather than with a library so the dashboard adds
    no dependency beyond ``rich``: ``msvcrt`` on Windows, ``termios`` +
    ``select`` elsewhere. Both are standard library.
    """

    def __init__(self) -> None:
        self._active = False
        self._restore = None

    def __enter__(self) -> "_KeyReader":
        if not getattr(sys.stdin, "isatty", lambda: False)():
            return self
        if os.name == "nt":
            self._active = True
            return self
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            self._active = True
        except Exception:  # noqa: BLE001 - no raw mode available; keys off
            self._active = False
        return self

    def __exit__(self, *exc) -> None:
        if self._restore is not None:
            try:
                self._restore()
            except Exception:  # noqa: BLE001
                pass

    def poll(self) -> Optional[str]:
        """The pressed key as a lowercase char, or ``None`` if none is waiting."""
        if not self._active:
            return None
        try:
            if os.name == "nt":
                import msvcrt

                if msvcrt.kbhit():
                    return msvcrt.getwch().lower()
                return None
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
        except Exception:  # noqa: BLE001 - key handling is never load-bearing
            self._active = False
        return None


class ConsoleUI:
    """The live dashboard. Use via :func:`console_dashboard`, not directly."""

    def __init__(self, run_id: str = "", live_url: str = "",
                 logger_name: str = DEFAULT_LOGGER_NAME) -> None:
        from rich.console import Console

        self.run_id = run_id
        self.live_url = live_url
        self._logger_name = logger_name
        self._console = Console()

        self._start = time.perf_counter()
        self._phases: list[dict] = []           # ordered, as encountered
        self._plan = {code: (label, w) for code, label, w in _PHASE_PLAN}
        self._stats: dict[str, str] = {}
        self._tail: deque = deque(maxlen=40)
        # Assumption/training checks (`observability.check(...)`, which is
        # what every `iforest.*` / `vae.*` gate in `src/utils/assumptions.py`
        # calls), most-recent-last.
        self._checks: deque = deque(maxlen=8)
        self._check_counts = {"pass": 0, "fail": 0}
        # Interpretability's own progress checkpoints (`interpretability.*` --
        # `iforest_explain.py` / `vae_explain.py` / `attribution_export.py`)
        # are kept separate from `_checks` rather than mixed in: they are
        # always-passing progress pings, not assumption gates, and Phase 10
        # alone can fire a few dozen of them in quick succession, which would
        # otherwise flush every real assumption result out of the 8-slot
        # `_checks` deque and make both harder to read. `(short_name, ts)`,
        # most-recent-last -- `ts` (`time.perf_counter()`) is what lets the
        # "now" line show how long it has sat on the current sub-step, the
        # extra level of detail a phase-only progress bar cannot give: a stall
        # inside Phase 10 shows up here as a frozen sub-step and a growing
        # "hace Ns", not just an unmoving progress bar.
        self._interp_checks: deque = deque(maxlen=3)
        self._interp_total = 0
        # Sampled roughly once a second (see `_loop`), not on every ~100ms
        # repaint -- a psutil call per frame would be needless overhead for a
        # number that does not change that fast.
        self._resource: resource_monitor.ResourceSample | None = None
        self._last_resource_sample = 0.0
        self._lock = threading.Lock()

        self._verbose = False
        self._paused = False
        self._live = None
        self._handler: Optional[_TailHandler] = None
        self._silenced: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- public API used by the pipeline ---------------------------------- #
    def set_stat(self, label: str, value) -> None:
        """Publish a headline number to the stats panel (idempotent)."""
        with self._lock:
            self._stats[str(label)] = str(value)

    # -- phase observer ---------------------------------------------------- #
    def _on_phase(self, name: str, event: str, duration_s) -> None:
        with self._lock:
            if event == "phase_started":
                self._phases.append(
                    {"name": name, "status": "running", "t0": time.perf_counter(),
                     "duration": None}
                )
                return
            for rec in reversed(self._phases):
                if rec["name"] == name and rec["status"] == "running":
                    rec["status"] = "done" if event == "phase_completed" else "failed"
                    rec["duration"] = duration_s
                    return

    # -- assumption/training-check observer --------------------------------- #
    @staticmethod
    def _interp_short_name(name: str) -> str:
        prefix = "interpretability."
        return name[len(prefix):] if name.startswith(prefix) else name

    def _on_check(self, hc) -> None:
        """Record one `observability.HealthCheck` (a passed or failed gate).

        Covers every `observability.check(...)` call project-wide, which
        includes the `iforest.*` / `vae.*` assumption gates run immediately
        before each model's `.fit()` (`src/utils/assumptions.py`) -- the
        "pruebas de los supuestos" this panel exists to surface live.

        `interpretability.*` checkpoints (Phase 10's own progress pings, see
        the docstring on `self._interp_checks`) are routed to that separate
        deque instead of `_checks`/`_check_counts`, so they get their own
        clearly-labelled place in the dashboard rather than diluting the
        assumption-gate tally and list with routine progress noise.
        """
        with self._lock:
            if hc.name.startswith("interpretability."):
                self._interp_total += 1
                self._interp_checks.append(
                    (self._interp_short_name(hc.name), time.perf_counter())
                )
                return
            self._check_counts["pass" if hc.passed else "fail"] += 1
            self._checks.append((hc.name, bool(hc.passed), hc.category))

    # -- rendering --------------------------------------------------------- #
    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _label_for(self, name: str) -> str:
        """Human label for a phase name, keeping any ``[model]`` suffix."""
        code = name.split(":", 1)[0].strip()
        label, _ = self._plan.get(code, (None, 0.0))
        if label is None:
            # Unplanned phase: fall back to the text after the colon.
            label = name.split(":", 1)[-1].strip() or name
        suffix = ""
        if "[" in name and "]" in name:
            suffix = " " + name[name.index("["):name.rindex("]") + 1]
        return f"{label}{suffix}"

    def _progress_fraction(self) -> tuple[float, int, int]:
        """``(fraction, completed, total)`` weighted by the plan's cost model."""
        done_codes = {
            p["name"].split(":", 1)[0].strip()
            for p in self._phases if p["status"] == "done"
        }
        planned_total = sum(w for _, _, w in _PHASE_PLAN)
        done_weight = sum(
            w for code, _, w in _PHASE_PLAN if code in done_codes
        )
        n_done = sum(1 for p in self._phases if p["status"] == "done")
        n_total = max(len(_PHASE_PLAN), len(self._phases))
        frac = min(done_weight / planned_total, 1.0) if planned_total else 0.0
        return frac, n_done, n_total

    def _render(self):
        from rich.align import Align
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        with self._lock:
            phases = list(self._phases)
            stats = dict(self._stats)
            tail = list(self._tail)
            checks = list(self._checks)
            check_counts = dict(self._check_counts)
            resource = self._resource
            interp_checks = list(self._interp_checks)
            interp_total = self._interp_total

        elapsed = time.perf_counter() - self._start
        frac, n_done, n_total = self._progress_fraction()

        # -- header ------------------------------------------------------- #
        head = Table.grid(expand=True)
        head.add_column(justify="left", ratio=1)
        head.add_column(justify="right")
        left = Text("Pipeline de detección de anomalías", style="bold")
        if self.run_id:
            left.append(f"   run {self.run_id[:12]}", style="dim")
        right = Text(self._fmt_elapsed(elapsed), style="bold cyan")
        if self._paused:
            right.append("  || PAUSA", style="bold yellow")
        head.add_row(left, right)

        # -- progress bar -------------------------------------------------- #
        width = max(20, min(self._console.width, 120) - 26)
        filled = int(round(frac * width))
        bar = Text()
        bar.append("█" * filled, style="cyan")
        bar.append("░" * (width - filled), style="dim")
        bar.append(f"  {frac * 100:5.1f}%  ", style="bold")
        bar.append(f"{n_done}/{n_total} fases", style="dim")

        running = [p for p in phases if p["status"] == "running"]
        if running:
            cur = running[-1]
            # ASCII spinner on purpose: simple and portable across terminals,
            # unlike a braille-dot spinner which some fonts render poorly.
            spin = "|/-\\"[int(time.perf_counter() * 8) % 4]
            now = Text()
            now.append(f"{spin} ", style="cyan")
            now.append(self._label_for(cur["name"]), style="bold")
            now.append(
                f"   {self._fmt_elapsed(time.perf_counter() - cur['t0'])}", style="dim"
            )
        else:
            now = Text("· en espera", style="dim")

        # -- interpretability sub-step: one level of detail below "now" ----- #
        # Phase-level progress says "Phase 10 is running"; this says exactly
        # which internal checkpoint it last reached and for how long it has
        # sat there -- the detail that tells the difference between "SHAP is
        # still busy" and "SHAP is hung" without reading `execution.log`. Kept
        # visible after Phase 10 finishes too (like the phase checklist boxes
        # staying green), so the final state is still legible right up to the
        # next phase's first checkpoint replacing it.
        interp_sub = None
        if interp_checks:
            trail = " -> ".join(name for name, _ts in interp_checks)
            last_ts = interp_checks[-1][1]
            interp_sub = Text("  ↳ interpretabilidad  ", style="dim")
            interp_sub.append(trail, style="cyan")
            interp_sub.append(
                f"   {self._fmt_elapsed(time.perf_counter() - last_ts)}"
                f"   ({interp_total} checkpoint{'s' if interp_total != 1 else ''})",
                style="dim",
            )

        # -- resource health: RAM/CPU, most useful during training / interpretability #
        # A single always-visible line rather than gated to specific phases:
        # a psutil sample is cheap enough (~1/s) that gating it on the current
        # phase would add complexity for no real saving, and the reader
        # benefits from seeing the baseline before the heavy phases start.
        health = None
        if resource is not None:
            pct = resource.system_used_pct
            pct_style = "red" if pct >= 90 else "yellow" if pct >= 75 else "green"
            health = Text("Equipo  ", style="dim")
            health.append("RAM proceso ", style="dim")
            health.append(f"{resource.process_rss_mb:,.0f} MB", style="bold")
            health.append("   RAM sistema ", style="dim")
            health.append(f"{pct:.0f}%", style=f"bold {pct_style}")
            health.append(f" ({resource.system_available_mb / 1024:.1f} GB libres)",
                          style="dim")
            health.append("   CPU ", style="dim")
            health.append(f"{resource.cpu_pct:.0f}%", style="bold")

        # -- phase checklist: the full fixed plan, boxes light up in place ---- #
        # One row per planned phase *code* (not per model-suffixed instance),
        # so the list never grows or scrolls -- a phase that runs once per
        # model (e.g. "Phase 8: evaluation [iforest]" then "... [vae]") is one
        # row that goes running -> done -> running -> done, which is also an
        # accurate read of what is actually happening.
        by_code: dict[str, list[dict]] = {}
        for p in phases:
            by_code.setdefault(p["name"].split(":", 1)[0].strip(), []).append(p)

        ptab = Table.grid(padding=(0, 1), expand=True)
        ptab.add_column(width=1)
        ptab.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        ptab.add_column(justify="right", width=8, no_wrap=True)
        for code, label, _w in _PHASE_PLAN:
            instances = by_code.get(code, [])
            if not instances:
                status = "pending"
            elif any(i["status"] == "failed" for i in instances):
                status = "failed"
            elif any(i["status"] == "running" for i in instances):
                status = "running"
            else:
                status = "done"
            icon, style = _STATUS_STYLE[status]
            total_dur = sum(i["duration"] for i in instances if i["duration"] is not None)
            dur = f"{total_dur:.1f}s" if total_dur else ""
            row_style = style if status != "pending" else "dim"
            ptab.add_row(
                Text.from_markup(icon),
                Text(label, style=row_style),
                Text(dur, style="dim"),
            )

        # -- supuestos: assumption/training checks (iforest.*, vae.*, ...) --- #
        ctab = Table.grid(padding=(0, 1), expand=True)
        ctab.add_column(width=1)
        ctab.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        tally = Text()
        tally.append(f"✓ {check_counts.get('pass', 0)}  ", style="green")
        tally.append(f"✗ {check_counts.get('fail', 0)}", style="red" if check_counts.get("fail") else "dim")
        if not checks:
            ctab.add_row("", Text("(ningún supuesto evaluado aún)", style="dim"))
        for name, passed, _category in checks:
            icon, style = _CHECK_STYLE[passed]
            ctab.add_row(Text.from_markup(icon), Text(name, style=style if not passed else "dim"))

        # -- stats / KPIs ------------------------------------------------------ #
        # expand=True so the value column is flushed to the panel's right edge
        # -- a ragged right makes a KPI list much harder to scan.
        stab = Table.grid(padding=(0, 1), expand=True)
        stab.add_column(ratio=1)
        stab.add_column(justify="right", no_wrap=True)
        if not stats:
            stab.add_row(Text("(sin datos aún)", style="dim"), "")
        for key, value in stats.items():
            stab.add_row(Text(key, style="dim"), Text(str(value), style="bold"))

        body = Table.grid(expand=True)
        body.add_column(ratio=3)
        body.add_column(ratio=3)
        body.add_row(
            Panel(ptab, title="[dim]Fases (plan completo)[/dim]", border_style="dim",
                  padding=(0, 1)),
            Group(
                Panel(Group(tally, ctab), title="[dim]Supuestos (IF / VAE)[/dim]",
                      border_style="dim", padding=(0, 1)),
                Panel(stab, title="[dim]Datos de la corrida[/dim]", border_style="dim",
                      padding=(0, 1)),
            ),
        )

        # -- log tail -------------------------------------------------------- #
        n_lines = 16 if self._verbose else 5
        ltab = Table.grid(padding=(0, 1))
        ltab.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        for levelno, message in tail[-n_lines:]:
            style = ("red" if levelno >= logging.ERROR
                     else "yellow" if levelno >= logging.WARNING else "dim")
            ltab.add_row(Text(message, style=style))

        hint = Text()
        for key, label in (("v", "detalle"), ("o", "vista web"), ("p", "pausa")):
            hint.append(f" {key} ", style="reverse")
            hint.append(f"{label}  ", style="dim")
        if self.live_url:
            hint.append(f"  {self.live_url}", style="dim underline")

        info_rows = [head, Text(""), bar, now]
        if interp_sub is not None:
            info_rows.append(interp_sub)
        if health is not None:
            info_rows.append(health)
        return Panel(
            Group(
                *info_rows,
                Text(""),
                body,
                Panel(ltab, title="[dim]Registro[/dim]", border_style="dim",
                      padding=(0, 1)),
                Align.left(hint),
            ),
            border_style="cyan",
            padding=(1, 2),
        )

    # -- lifecycle ---------------------------------------------------------- #
    def _silence_console_logging(self) -> None:
        """Detach stream handlers so log lines cannot tear the Live display.

        The file handler is untouched -- ``artifacts/logs/execution.log`` still
        receives every line, so nothing is lost by not printing it.
        """
        logger = logging.getLogger(self._logger_name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                logger.removeHandler(handler)
                self._silenced.append(handler)
        self._handler = _TailHandler(self._tail)
        logger.addHandler(self._handler)

    def _restore_console_logging(self) -> None:
        logger = logging.getLogger(self._logger_name)
        if self._handler is not None:
            logger.removeHandler(self._handler)
            self._handler = None
        for handler in self._silenced:
            logger.addHandler(handler)
        self._silenced.clear()

    def _loop(self) -> None:
        """Repaint + key polling, ~10fps, on a daemon thread."""
        with _KeyReader() as keys:
            while not self._stop.is_set():
                key = keys.poll()
                if key:
                    if self._paused:
                        self._paused = False
                    elif key == "v":
                        self._verbose = not self._verbose
                    elif key == "p":
                        self._paused = True
                    elif key == "o" and self.live_url:
                        try:
                            webbrowser.open(self.live_url)
                        except Exception:  # noqa: BLE001
                            pass
                now_t = time.perf_counter()
                if now_t - self._last_resource_sample >= 1.0:
                    self._last_resource_sample = now_t
                    reading = resource_monitor.sample()
                    if reading is not None:
                        with self._lock:
                            self._resource = reading
                if not self._paused and self._live is not None:
                    try:
                        self._live.update(self._render())
                    except Exception:  # noqa: BLE001 - never kill the run
                        self._stop.set()
                        return
                self._stop.wait(0.1)

    def start(self) -> "ConsoleUI":
        from rich.live import Live

        self._silence_console_logging()
        add_phase_observer(self._on_phase)
        add_check_observer(self._on_check)
        self._live = Live(self._render(), console=self._console,
                          refresh_per_second=10, transient=False)
        self._live.start()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="console-ui")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        remove_phase_observer(self._on_phase)
        remove_check_observer(self._on_check)
        if self._live is not None:
            try:
                self._live.update(self._render())
                self._live.stop()
            except Exception:  # noqa: BLE001
                pass
            self._live = None
        self._restore_console_logging()


class _NullUI:
    """Stand-in with the same surface, so callers never branch on availability."""

    run_id = ""
    live_url = ""

    def set_stat(self, label: str, value) -> None:  # noqa: D102
        pass

    def start(self):  # noqa: D102
        return self

    def stop(self) -> None:  # noqa: D102
        pass


class console_dashboard:  # noqa: N801 - used as a context manager, not a class
    """Context manager yielding a :class:`ConsoleUI` or a silent stand-in.

    Usage::

        with console_dashboard(enabled=cfg.console_ui, run_id=ctx.run_id) as ui:
            ui.set_stat("Filas", f"{len(df):,}")
            ...

    Always yields something with ``set_stat``, so call sites never guard.
    """

    def __init__(self, enabled: bool = True, run_id: str = "", live_url: str = "") -> None:
        self.enabled = bool(enabled) and supports_dashboard()
        self._run_id = run_id
        self._live_url = live_url
        self._ui = None

    def __enter__(self):
        if not self.enabled:
            self._ui = _NullUI()
            return self._ui
        try:
            self._ui = ConsoleUI(run_id=self._run_id, live_url=self._live_url).start()
        except Exception:  # noqa: BLE001 - decoration must never break the run
            self._ui = _NullUI()
        return self._ui

    def __exit__(self, *exc) -> None:
        try:
            self._ui.stop()
        except Exception:  # noqa: BLE001
            pass
        return None


# --------------------------------------------------------------------------- #
# Module-level singleton                                                       #
# --------------------------------------------------------------------------- #
# `run_pipeline` is one long function whose body would have to be re-indented
# wholesale to sit inside a `with` block. A module-level active dashboard lets
# call sites deep inside it publish a stat without threading a `ui` argument
# through every signature -- the same reason `observability.check()` reaches
# for the active run instead of taking a `ctx` parameter.
_ACTIVE = None


def active():
    """The running dashboard, or ``None``."""
    return _ACTIVE


def start_dashboard(enabled: bool = True, run_id: str = "", live_url: str = ""):
    """Start the dashboard and install it as the process-wide active one.

    Returns the dashboard (or a silent stand-in). Safe to call when ``rich``
    is missing or the output is redirected -- it degrades to a no-op.
    """
    global _ACTIVE
    stop_dashboard()
    if not (enabled and supports_dashboard()):
        _ACTIVE = _NullUI()
        return _ACTIVE
    try:
        _ACTIVE = ConsoleUI(run_id=run_id, live_url=live_url).start()
    except Exception:  # noqa: BLE001 - decoration must never break the run
        _ACTIVE = _NullUI()
    return _ACTIVE


def stop_dashboard() -> None:
    """Tear down the active dashboard and restore plain console logging."""
    global _ACTIVE
    if _ACTIVE is not None:
        try:
            _ACTIVE.stop()
        except Exception:  # noqa: BLE001
            pass
    _ACTIVE = None


def set_stat(label: str, value) -> None:
    """Publish a headline number to the active dashboard; no-op when off."""
    if _ACTIVE is not None:
        try:
            _ACTIVE.set_stat(label, value)
        except Exception:  # noqa: BLE001
            pass
