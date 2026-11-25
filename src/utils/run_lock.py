"""Serialize writers to a shared artifact directory with OS-managed locks.

Data sources / inputs: artifacts/.pipeline.lock, released by the OS on exit.
Created: 2026-11-25
Last modified: 2026-11-25
Changelog: 2026-11-25 — reject overlapping runs before artifact mutation.
"""
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


def _lock(stream, release=False):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        mode = msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK
        msvcrt.locking(stream.fileno(), mode, 1)
    else:
        import fcntl
        mode = fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(stream.fileno(), mode)


@contextmanager
def artifact_lock(path="artifacts/.pipeline.lock"):
    """Fail immediately on contention; no stale PID file removal is needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        try:
            _lock(stream)
        except OSError as exc:
            raise RuntimeError(
                f"Another pipeline run owns {destination.resolve()}. "
                "Wait for it to finish or use an isolated workspace/artifact directory."
            ) from exc
        try:
            yield
        finally:
            _lock(stream, release=True)


def serialized_artifacts(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with artifact_lock():
            return function(*args, **kwargs)
    return wrapped
