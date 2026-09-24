"""Live progress reporting for the diagnostic suite.

A full diagnostic can spend minutes inside one loop (seeded stability refits,
experiment sweeps, one KS test per feature) with nothing on screen. This module
answers the three questions an operator has while it runs: *which function is
executing*, *for how long*, and *how far along is each test*.

Two primitives, both pure instrumentation -- neither changes a computed value:

* :func:`step` wraps one named function/stage and reports when it starts and
  ends, with its own measured duration.
* :func:`track` wraps a loop-shaped test in a tqdm bar. :func:`stages` does the
  same for a fixed sequence of named tests (one bar tick per test, and each
  test is also a :func:`step`).

The suite stays host-agnostic. Everything is also published as plain-dict
events to registered observers (:func:`add_observer`), which is how a host
pipeline mirrors the same progress into its own dashboard or live flow view
without the suite importing anything from it. Event shapes:

``{"event": "step_started" | "step_completed" | "step_failed", "name", "label",
"depth", "duration_s" (end events), "error" (failed)}``

``{"event": "progress", "state": "start" | "update" | "end", "desc", "n",
"total", "unit", "elapsed_s", "current"}``

Every event also carries ``t``, the epoch seconds it was emitted at.

tqdm draws to stderr, which would tear a host's repainting terminal dashboard;
such a host calls ``configure(tqdm_enabled=False)`` and renders the events
itself. Observers are best-effort: one that raises is dropped rather than
allowed to break the computation it was only watching.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Optional

__all__ = [
    "DEFAULT_MIN_INTERVAL_S", "add_observer", "configure", "remove_observer",
    "stages", "step", "track",
]

Observer = Callable[[dict[str, Any]], None]

#: Minimum seconds between two "update" events of one bar. tqdm throttles its
#: own drawing; this keeps a host that persists every event (a JSONL log) from
#: writing one line per iteration of a fast loop. Start and end always fire.
DEFAULT_MIN_INTERVAL_S = 0.5

_OBSERVERS: list[Observer] = []
_SETTINGS: dict[str, Any] = {
    "tqdm_enabled": True,
    "min_interval_s": DEFAULT_MIN_INTERVAL_S,
}
_OPEN_STEPS: list[str] = []


def add_observer(callback: Observer) -> None:
    """Register ``callback(event: dict)`` for every step/progress event."""
    if callback not in _OBSERVERS:
        _OBSERVERS.append(callback)


def remove_observer(callback: Observer) -> None:
    """Unregister a previously added observer; silent if absent."""
    if callback in _OBSERVERS:
        _OBSERVERS.remove(callback)


def configure(
    *, tqdm_enabled: Optional[bool] = None, min_interval_s: Optional[float] = None
) -> None:
    """Turn tqdm drawing on/off and/or set the update-event throttle."""
    if tqdm_enabled is not None:
        _SETTINGS["tqdm_enabled"] = bool(tqdm_enabled)
    if min_interval_s is not None:
        _SETTINGS["min_interval_s"] = float(min_interval_s)


def _emit(event: dict[str, Any]) -> None:
    payload = {"t": round(time.time(), 3), **event}
    for callback in list(_OBSERVERS):
        try:
            callback(payload)
        except Exception:  # noqa: BLE001 - an observer must never break a test
            remove_observer(callback)


@contextmanager
def step(name: str, *, label: str = "") -> Iterator[None]:
    """Report one named function/stage: start, then completion or failure,
    each with the step's own depth in the currently open stack."""
    depth = len(_OPEN_STEPS)
    _OPEN_STEPS.append(name)
    started = time.perf_counter()
    _emit({"event": "step_started", "name": name, "label": label, "depth": depth})
    try:
        yield
    except BaseException as exc:  # includes Ctrl-C: the step did not finish
        _OPEN_STEPS.pop()
        _emit({
            "event": "step_failed", "name": name, "label": label, "depth": depth,
            "duration_s": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    _OPEN_STEPS.pop()
    _emit({
        "event": "step_completed", "name": name, "label": label, "depth": depth,
        "duration_s": round(time.perf_counter() - started, 3),
    })


def _make_tqdm(desc: str, total: Optional[int], unit: str) -> Any:
    if not _SETTINGS["tqdm_enabled"]:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # tqdm is a convenience here, never a requirement
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


class _Bar:
    """One progress bar: a tqdm display plus the events mirroring it."""

    def __init__(self, desc: str, total: Optional[int], unit: str) -> None:
        self.desc = desc
        self.total = total
        self.unit = unit
        self.n = 0
        self.current = ""
        self._started = time.perf_counter()
        self._last_emit = self._started
        self._tqdm = _make_tqdm(desc, total, unit)
        self._report("start")

    def _report(self, state: str) -> None:
        self._last_emit = time.perf_counter()
        _emit({
            "event": "progress", "state": state, "desc": self.desc, "n": self.n,
            "total": self.total, "unit": self.unit, "current": self.current,
            "elapsed_s": round(self._last_emit - self._started, 3),
        })

    def _report_if_due(self) -> None:
        if time.perf_counter() - self._last_emit >= _SETTINGS["min_interval_s"]:
            self._report("update")

    def running(self, current: str) -> None:
        """Name the item now in flight (tqdm postfix + event); does not advance."""
        self.current = current
        if self._tqdm is not None:
            self._tqdm.set_postfix_str(current)
        self._report_if_due()

    def advance(self) -> None:
        self.n += 1
        if self._tqdm is not None:
            self._tqdm.update(1)
        self._report_if_due()

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
        self._report("end")


def track(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    label: Optional[Callable[[Any], str]] = None,
) -> Iterator[Any]:
    """Yield the items of ``iterable`` unchanged, under a tqdm bar ``desc``.

    An item counts as done only once the caller has finished with it (the
    generator is resumed), so ``n`` never runs ahead of the work. The bar is
    closed -- and its end event sent -- even if the caller stops early or
    raises. ``label(item)``, when given, names the item in flight (shown after
    the bar and sent as ``current``), e.g. the seed or file being processed.
    """
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)  # type: ignore[arg-type]
    bar = _Bar(desc, total, unit)
    try:
        for item in iterable:
            if label is not None:
                bar.running(label(item))
            yield item
            bar.advance()
    finally:
        bar.close()


class _Stages:
    def __init__(self, bar: _Bar) -> None:
        self._bar = bar

    @contextmanager
    def stage(self, name: str, *, label: str = "") -> Iterator[None]:
        """Run one named test as a :func:`step`; the bar ticks on success only."""
        self._bar.running(name)
        with step(name, label=label):
            yield
        self._bar.advance()


@contextmanager
def stages(desc: str, *, total: int, unit: str = "test") -> Iterator[_Stages]:
    """A bar over a fixed sequence of ``total`` named tests::

        with progress.stages("run_diagnostic", total=3) as st:
            with st.stage("validate_frames"):
                ...
    """
    bar = _Bar(desc, total, unit)
    try:
        yield _Stages(bar)
    finally:
        bar.close()
