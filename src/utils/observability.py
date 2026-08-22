"""Structured, machine-readable observability layer.

Complements `src.utils.logging_config` (which writes a human-readable text
log) with a second, additive channel: one JSON object per event, appended to
`artifacts/logs/run_events.jsonl`. Nothing here changes what the text logger
does -- `log_phase` degrades to its current (pre-existing) behavior whenever
no run is active, so every caller that never calls `start_run()` (all 232
existing tests included) is unaffected byte-for-byte.

Two purposes this module exists to serve, both requested for this project:
  1. Reproducibility/audit: every event carries a `run_id`, so a run's full
     lifecycle (config, dataset, phases, health checks, final status) can be
     reconstructed from one file without re-running anything.
  2. A prerequisite for flow visualization: a renderer needs real, timestamped
     events to draw from -- this module is the source of those events, not
     the renderer itself.

Design choice -- JSON Lines (one `json.dumps(...)` object per line, appended,
never rewritten) over a single JSON array: the pipeline can crash mid-run
(that is one of the things being observed) and a truncated JSONL file still
parses every complete line that was written before the crash; a truncated
single JSON array does not parse at all.
"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
import tracemalloc
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils import paths

_ACTIVE_RUN: "contextvars.ContextVar[Optional[RunContext]]" = contextvars.ContextVar(
    "_ACTIVE_RUN", default=None
)

#: Callables notified every time a `HealthCheck` is recorded, as
#: ``callback(check: HealthCheck)``. Mirrors
#: `src.utils.logging_config`'s phase-observer hook: it exists so a console
#: dashboard can show assumption/training checks (e.g. `iforest.*`, `vae.*`)
#: live, without any of the ~10 call sites that invoke `check()` (here and in
#: `src/utils/assumptions.py`) knowing a subscriber exists. Best-effort: a
#: raising observer is dropped rather than allowed to break the check it was
#: only supposed to be watching.
_CHECK_OBSERVERS: list = []


def add_check_observer(callback) -> None:
    """Register ``callback(check)`` to be called on every recorded HealthCheck."""
    if callback not in _CHECK_OBSERVERS:
        _CHECK_OBSERVERS.append(callback)


def remove_check_observer(callback) -> None:
    """Unregister a previously added check observer; silent if absent."""
    if callback in _CHECK_OBSERVERS:
        _CHECK_OBSERVERS.remove(callback)


def _notify_check(hc: "HealthCheck") -> None:
    for callback in list(_CHECK_OBSERVERS):
        try:
            callback(hc)
        except Exception:  # noqa: BLE001 - an observer must never break a check
            _CHECK_OBSERVERS.remove(callback)

SEVERITIES = ("info", "warning", "critical")

# The 8 categories requested for the run-health schema. Not all are populated
# by this module alone -- data/assumption/training/validation/tuning health
# checks are the responsibility of the callers that actually run those
# stages (the assumption-validation gate, the tuning loop, ...); this module
# only defines the shared shape and records whatever it is given.
HEALTH_CATEGORIES = (
    "data",
    "assumption",
    "training",
    "validation",
    "tuning",
    "resource",
    "artifact",
    "reproducibility",
)


def _hash_obj(obj: Any) -> str:
    """Stable short hash of a JSON-serializable object (e.g. an effective config)."""
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class DatasetFingerprint:
    """Cheap, non-invasive dataset identity: file stats + shape, not a full-content hash.

    A full-content hash of a ~192MB data.csv would add real I/O cost for
    marginal benefit here; path + size + mtime + shape + column-name hash is
    enough to detect "this is not the file the run thinks it is" (wrong path,
    stale file, schema drift) without reading the whole file twice.
    """

    path: str
    exists: bool
    size_bytes: Optional[int] = None
    mtime: Optional[str] = None
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    columns_hash: Optional[str] = None

    @classmethod
    def from_path(cls, path: str, df=None) -> "DatasetFingerprint":
        if not os.path.exists(path):
            return cls(path=path, exists=False)
        stat = os.stat(path)
        fp = cls(
            path=path,
            exists=True,
            size_bytes=stat.st_size,
            mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
        )
        if df is not None:
            fp.n_rows, fp.n_cols = int(df.shape[0]), int(df.shape[1])
            fp.columns_hash = _hash_obj(list(df.columns))
        return fp

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class HealthCheck:
    """One evaluated run-health check.

    Every field the mega-brief asked for is mandatory, not optional, so a
    check can never be recorded half-specified: metric name, definition,
    expected range, severity, failure action, evidence location.
    """

    name: str
    category: str
    definition: str
    expected: str
    severity: str
    passed: bool
    observed: Any = None
    failure_action: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.category not in HEALTH_CATEGORIES:
            raise ValueError(
                f"Unknown health category {self.category!r}; expected one of {HEALTH_CATEGORIES}"
            )
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity {self.severity!r}; expected one of {SEVERITIES}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class RunContext:
    run_id: str
    pipeline_version: str
    config: dict
    config_hash: str
    seed: Optional[int]
    python_version: str
    platform: str
    started_at: str
    events_path: str
    dataset: Optional[DatasetFingerprint] = None
    health_checks: list = field(default_factory=list)

    def record_health(self, check: HealthCheck) -> HealthCheck:
        self.health_checks.append(check)
        emit(self, "health_check", **check.to_dict())
        return check

    def set_dataset(self, fingerprint: DatasetFingerprint) -> None:
        self.dataset = fingerprint
        emit(self, "dataset_fingerprint", **fingerprint.to_dict())

    def summary(self) -> dict:
        by_severity = {s: 0 for s in SEVERITIES}
        failed = []
        for c in self.health_checks:
            if not c.passed:
                by_severity[c.severity] += 1
                failed.append(c.name)
        return {
            "run_id": self.run_id,
            "total_checks": len(self.health_checks),
            "failed_checks": failed,
            "failed_by_severity": by_severity,
        }


def start_run(
    config: dict,
    seed: Optional[int] = None,
    pipeline_version: str = "modelo-v0.1",
    events_path: str = paths.RUN_EVENTS_LOG,
) -> RunContext:
    """Begin a run: allocate a run_id, open the JSONL event stream, emit `run_started`.

    Sets the run as "active" for this execution context (a plain
    `contextvars.ContextVar`, not a global) so `log_phase` (in
    `logging_config.py`) can emit phase-lifecycle events without every call
    site having to thread a context object through.
    """
    parent = os.path.dirname(events_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    tracemalloc.start()
    ctx = RunContext(
        run_id=run_id,
        pipeline_version=pipeline_version,
        config=config,
        config_hash=_hash_obj(config),
        seed=seed,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        started_at=_now(),
        events_path=events_path,
    )
    _ACTIVE_RUN.set(ctx)
    emit(
        ctx, "run_started",
        config=config, config_hash=ctx.config_hash, seed=seed,
        pipeline_version=pipeline_version,
        python_version=ctx.python_version, platform=ctx.platform,
    )
    return ctx


def current_run() -> Optional[RunContext]:
    return _ACTIVE_RUN.get()


def peak_memory_mb() -> Optional[float]:
    """Peak Python-level allocation since `start_run`, via stdlib `tracemalloc`.

    Not process RSS (that needs `psutil`, which is not a project dependency
    and is not added here uninvited) -- this is a real, if partial, number:
    it undercounts C-extension allocations (numpy/torch buffers largely live
    outside the Python allocator `tracemalloc` tracks), so treat it as a
    lower bound on actual memory use, not the full picture.
    """
    if not tracemalloc.is_tracing():
        return None
    _current, peak = tracemalloc.get_traced_memory()
    return round(peak / (1024 * 1024), 2)


def end_run(ctx: RunContext, status: str, **extra: Any) -> dict:
    """Close a run. `status`: 'success' | 'failed' | 'stopped_early' | 'skipped'."""
    summary = ctx.summary()
    summary["peak_python_memory_mb"] = peak_memory_mb()
    emit(ctx, "run_ended", status=status, summary=summary, **extra)
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    return summary


def emit(ctx: Optional[RunContext], event: str, **fields: Any) -> None:
    """Append one JSON Lines event. Silent no-op if `ctx` is None (no active run).

    Best-effort by design: a logging I/O failure (disk full, permissions)
    must never take down the pipeline it is only supposed to be observing --
    that would invert the point of an observability layer.
    """
    if ctx is None:
        return
    record = {"ts": _now(), "run_id": ctx.run_id, "event": event, **fields}
    try:
        with open(ctx.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def check(
    name: str,
    category: str,
    definition: str,
    expected: str,
    severity: str,
    passed: bool,
    observed: Any = None,
    failure_action: str = "",
    evidence: str = "",
) -> Optional[HealthCheck]:
    """Convenience wrapper: record a `HealthCheck` against the active run, if any.

    Lets call sites deep inside `main.py` (which never hold a `RunContext`
    reference) log a health check without threading `ctx` through every
    function signature -- mirrors how `phase_event` already works for
    `log_phase`.
    """
    hc = HealthCheck(
        name=name, category=category, definition=definition, expected=expected,
        severity=severity, passed=passed, observed=observed,
        failure_action=failure_action, evidence=evidence,
    )
    _notify_check(hc)
    ctx = current_run()
    if ctx is None:
        return None
    return ctx.record_health(hc)


def phase_event(name: str, event: str, **fields: Any) -> None:
    """Emit a phase-lifecycle event against the currently active run, if any.

    Called from `log_phase` so every existing `with log_phase(...):` call
    across the codebase gets structured events for free, with zero call-site
    changes, and zero behavior change when no run is active (tests).
    """
    emit(current_run(), event, phase=name, **fields)
