"""Atomic file replace, with a Windows-safe retry.

`os.replace(src, dst)` is how this project writes every checkpoint/best-params
file without ever leaving a half-written file at the real path (write to
``dst + ".tmp"``, then replace). On POSIX this call cannot fail from another
process merely having ``dst`` open. On Windows it can: antivirus real-time
scanning or the search indexer briefly opening a just-written file is enough
to make the immediately-following ``os.replace`` raise
``PermissionError: [WinError 5] Access is denied`` -- transient lock
contention, not a logic bug, and not reproducible on every call (this project
hit it once, on one Optuna trial out of dozens, writing
``vae_tuning/trial_29/best_model.pth``). Retrying with a short backoff is the
standard workaround for exactly this Windows behavior.
"""

from __future__ import annotations

import os
import time

__all__ = ["atomic_replace"]


def atomic_replace(src: str, dst: str, *, retries: int = 5, base_delay: float = 0.1) -> None:
    """``os.replace(src, dst)``, retrying only on a transient ``PermissionError``.

    Any other exception (e.g. ``FileNotFoundError`` because ``src`` was never
    written) propagates immediately -- retrying those would hide a real bug
    instead of a transient lock. Backoff doubles each attempt (0.1s, 0.2s,
    0.4s, 0.8s by default), so the worst case adds well under two seconds to
    a single checkpoint write, and only on the rare run that actually needs
    a retry at all.
    """
    last_exc: PermissionError | None = None
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(base_delay * (2**attempt))
    assert last_exc is not None
    raise last_exc
