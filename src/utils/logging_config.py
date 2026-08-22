"""Shared logging utilities for the project.

Every module (data generation, preprocessing, model training, tuning,
evaluation, reporting, ...) should call ``setup_logging()`` once to get a
configured logger, and wrap long-running phases with ``log_phase`` so that
start/end/duration are logged consistently across the whole project.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator

from src.utils import paths

DEFAULT_LOGGER_NAME = "modelo"
_CONFIGURED_LOGGERS: set[str] = set()

#: Callables notified on every phase transition, as
#: ``(phase_name, event, duration_s_or_None)`` where ``event`` is one of
#: ``"phase_started" | "phase_completed" | "phase_failed"``.
#:
#: This exists so a console dashboard can follow the pipeline without any of
#: the ~15 ``with log_phase(...)`` call sites knowing it is there. Observers
#: are best-effort: one that raises is dropped rather than allowed to break
#: the phase it was only supposed to be watching.
_PHASE_OBSERVERS: list = []


def add_phase_observer(callback) -> None:
    """Register ``callback(name, event, duration_s)`` for phase transitions."""
    if callback not in _PHASE_OBSERVERS:
        _PHASE_OBSERVERS.append(callback)


def remove_phase_observer(callback) -> None:
    """Unregister a previously added phase observer; silent if absent."""
    if callback in _PHASE_OBSERVERS:
        _PHASE_OBSERVERS.remove(callback)


def _notify_phase(name: str, event: str, duration_s=None) -> None:
    for callback in list(_PHASE_OBSERVERS):
        try:
            callback(name, event, duration_s)
        except Exception:  # noqa: BLE001 - an observer must never break a phase
            _PHASE_OBSERVERS.remove(callback)


def setup_logging(
    log_dir: str = paths.LOGS_DIR,
    log_file: str = paths.LOG_FILE,
    level: int = logging.INFO,
    logger_name: str = DEFAULT_LOGGER_NAME,
) -> logging.Logger:
    """Configure and return the project logger.

    Writes to ``<log_dir>/<log_file>`` and to the console. Safe to call
    multiple times (idempotent) - repeated calls will not attach duplicate
    handlers, but will update the logger's level.

    Args:
        log_dir: Directory the log file lives in. Created if missing.
        log_file: Log file name.
        level: Logging level for the logger and its handlers.
        logger_name: Name of the logger to configure.

    Returns:
        The configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger_name in _CONFIGURED_LOGGERS:
        # Already configured in this process; just sync the level.
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    _CONFIGURED_LOGGERS.add(logger_name)
    return logger


@contextmanager
def log_phase(
    name: str, logger: logging.Logger | None = None
) -> Iterator[None]:
    """Context manager that logs the start/end and duration of a phase.

    Usage:
        with log_phase("Data generation"):
            generate_data(...)

    Logs "Starting <name>" on entry and "Finished <name> in Ys" on a clean
    exit. If the wrapped block raises, logs "Failed <name> after Ys" (with
    the exception info) and re-raises.

    Args:
        name: Human-readable phase name, e.g. "Data generation".
        logger: Logger to use. Defaults to the project logger
            (``setup_logging()`` with defaults) if not provided.
    """
    log = logger or setup_logging()
    # Local import: `observability` imports `src.utils.paths`, and importing
    # it here (not at module level) keeps this already-widely-imported module
    # free of any import-order dependency on the newer one.
    from src.utils import observability

    log.info("Starting %s", name)
    observability.phase_event(name, "phase_started")
    _notify_phase(name, "phase_started")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        log.exception("Failed %s after %.2fs", name, elapsed)
        observability.phase_event(name, "phase_failed", duration_s=round(elapsed, 3))
        _notify_phase(name, "phase_failed", elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        log.info("Finished %s in %.2fs", name, elapsed)
        observability.phase_event(name, "phase_completed", duration_s=round(elapsed, 3))
        _notify_phase(name, "phase_completed", elapsed)
