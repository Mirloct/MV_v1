"""Progress bars for the pipeline's long loops: tqdm on the terminal AND
``progress`` events for the console dashboard and the flow HTML.

A bare ``tqdm`` bar is invisible to everything but a plain terminal: a
repainting ``rich`` dashboard and tqdm's carriage-return redraws on stderr tear
each other apart, and the live/static flow pages read ``run_events.jsonl``,
which a bare bar never writes to. :class:`Bar` is a drop-in for the tqdm calls
this project uses (iterate it, or drive it with ``update`` / ``set_postfix``)
that does both jobs:

* **events** -- every bar publishes ``start`` / ``update`` / ``end`` through
  :func:`src.utils.observability.progress_event` (log file, dashboard, flow
  pages). ``update`` events are throttled to one per ``min_interval_s`` so a
  fast loop does not write a line per iteration; start and end always fire.
* **tqdm** -- drawn on stderr only when no live dashboard owns the terminal
  (:func:`src.utils.console_ui.is_live`). With the dashboard up, the same bar is
  rendered *inside* it from the events, in tqdm's exact layout.

Instrumentation only: iterating a :class:`Bar` yields exactly the items given.
The diagnostic suite has its own standalone twin (``ifvae_diag.progress``); it
must not import from ``src/``, so the two are kept separate on purpose.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Iterator, Optional

from src.utils import observability

__all__ = ["Bar", "DEFAULT_MIN_INTERVAL_S", "configure", "show_best_trial", "track"]

#: Seconds between two "update" events of one bar (start/end always fire).
DEFAULT_MIN_INTERVAL_S = 0.5

_SETTINGS: dict[str, float] = {"min_interval_s": DEFAULT_MIN_INTERVAL_S}


def configure(*, min_interval_s: Optional[float] = None) -> None:
    """Set the update-event throttle (tests and tools; the default suits runs)."""
    if min_interval_s is not None:
        _SETTINGS["min_interval_s"] = float(min_interval_s)


def _make_tqdm(desc: str, total: Optional[int], unit: str, initial: int) -> Any:
    """A real tqdm bar, or ``None`` when a live dashboard owns the terminal
    (or tqdm is missing -- it is a convenience here, never a requirement)."""
    from src.utils import console_ui

    if console_ui.is_live():
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, unit=unit, initial=initial, dynamic_ncols=True)


def _format_postfix(fields: dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        text = f"{value:.4g}" if isinstance(value, float) else str(value)
        parts.append(f"{key}={text}")
    return ", ".join(parts)


class Bar:
    """One progress bar. ``Bar(iterable, desc=...)`` iterates like tqdm;
    ``Bar(desc=..., total=N)`` is driven by :meth:`update`."""

    def __init__(
        self,
        iterable: Optional[Iterable[Any]] = None,
        *,
        desc: str,
        total: Optional[int] = None,
        unit: str = "it",
        initial: int = 0,
        label: Optional[Callable[[Any], str]] = None,
    ) -> None:
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable) + initial  # type: ignore[arg-type]
        self._iterable = iterable
        self._label = label
        self.desc = desc
        self.total = total
        self.unit = unit
        self.n = initial
        self.current = ""
        self._closed = False
        self._started = time.perf_counter()
        self._last_emit = self._started
        self._tqdm = _make_tqdm(desc, total, unit, initial)
        self._report("start")

    # -- events --------------------------------------------------------------- #
    def _report(self, state: str) -> None:
        self._last_emit = time.perf_counter()
        observability.progress_event(
            self.desc, self.n, self.total, unit=self.unit, state=state,
            current=self.current, elapsed_s=self._last_emit - self._started,
        )

    def _report_if_due(self) -> None:
        if time.perf_counter() - self._last_emit >= _SETTINGS["min_interval_s"]:
            self._report("update")

    # -- tqdm-compatible surface --------------------------------------------- #
    def update(self, n: int = 1) -> None:
        self.n += n
        if self._tqdm is not None:
            self._tqdm.update(n)
        self._report_if_due()

    def set_postfix_str(self, text: str) -> None:
        """Name what is in flight / the latest figure (shown after the bar)."""
        self.current = text
        if self._tqdm is not None:
            self._tqdm.set_postfix_str(text)
        self._report_if_due()

    def set_postfix(self, **fields: Any) -> None:
        self.set_postfix_str(_format_postfix(fields))

    def close(self) -> None:
        """Finish the bar; safe to call more than once (only the first reports)."""
        if self._closed:
            return
        self._closed = True
        if self._tqdm is not None:
            self._tqdm.close()
        self._report("end")

    def __enter__(self) -> "Bar":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[Any]:
        if self._iterable is None:
            raise TypeError("this Bar has no iterable; drive it with update()")
        try:
            for item in self._iterable:
                if self._label is not None:
                    self.set_postfix_str(self._label(item))
                yield item
                self.update(1)
        finally:
            self.close()


def show_best_trial(bar: Bar, study: Any) -> None:
    """Show an Optuna study's best value after the bar (a bar cannot say whether
    the search is still improving; the value can). Silent before the first
    completed trial, when the study has no best value yet."""
    try:
        bar.set_postfix(best=float(study.best_value))
    except (ValueError, RuntimeError, TypeError):
        pass


def track(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    label: Optional[Callable[[Any], str]] = None,
) -> Bar:
    """``for x in track(items, desc="...")`` -- a :class:`Bar` over ``items``."""
    return Bar(iterable, desc=desc, total=total, unit=unit, label=label)
