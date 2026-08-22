"""System resource sampling for the live console dashboard.

Deliberately separate from `src.utils.observability.peak_memory_mb()`, which
stays stdlib-only (`tracemalloc`) by design -- see `CONTEXT.md`. This module
exists for a different, narrower purpose: an at-a-glance "is this machine
under strain" readout in the terminal while the heaviest phases run (VAE
training, Isolation Forest fitting/tuning, interpretability), where a
Python-level allocator trace undercounts the real cost (torch tensors and
numpy buffers mostly live outside it). Answering that needs process RSS and
system-wide memory pressure, which only `psutil` can give.

Degrades to ``None`` wherever ``psutil`` is unavailable, so a machine without
it (or a future environment where the dependency is dropped) loses only this
readout, never the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["ResourceSample", "sample", "available"]


@dataclass(frozen=True)
class ResourceSample:
    """One point-in-time reading. All fields ``None`` together on failure."""

    process_rss_mb: float
    system_used_pct: float
    system_available_mb: float
    system_total_mb: float
    cpu_pct: float


def available() -> bool:
    """True when ``psutil`` is importable -- check before relying on :func:`sample`."""
    try:
        import psutil  # noqa: F401
        return True
    except ImportError:
        return False


def sample() -> Optional[ResourceSample]:
    """One resource reading, or ``None`` if ``psutil`` is missing or errors.

    ``cpu_pct`` is process-wide system CPU utilisation since the *previous*
    call (``psutil.cpu_percent(interval=None)``, non-blocking); the first
    call in a process always reads ``0.0`` -- this is psutil's own documented
    behaviour, not a bug here, and self-corrects from the second sample on.
    """
    try:
        import psutil

        proc = psutil.Process()
        vm = psutil.virtual_memory()
        return ResourceSample(
            process_rss_mb=round(proc.memory_info().rss / (1024 * 1024), 1),
            system_used_pct=round(vm.percent, 1),
            system_available_mb=round(vm.available / (1024 * 1024), 1),
            system_total_mb=round(vm.total / (1024 * 1024), 1),
            cpu_pct=round(psutil.cpu_percent(interval=None), 1),
        )
    except Exception:  # noqa: BLE001 - a monitoring readout must never break the run
        return None
