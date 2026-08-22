"""Optuna storage helper: avoid SQLAlchemy connection pooling for SQLite.

``optuna.create_study(storage=<sqlite URI string>)`` builds a pooled
SQLAlchemy engine internally. SQLite has no real concurrent-connection story
(one writer at a time, file-locked), so a pooled idle connection is dead
weight -- it just sits open until Python's garbage collector eventually reaps
it, which surfaces as ``ResourceWarning: unclosed database in
<sqlite3.Connection ...>`` at some unrelated later point in the process (a
later ``gc.collect()`` inside an *unrelated* Optuna call, in this project's
own test suite). Optuna's own documentation recommends ``NullPool`` for
exactly this reason: skip pooling entirely, so each connection is opened and
closed around the work it does instead of being held open indefinitely.
"""

from __future__ import annotations

from typing import Any

__all__ = ["resolve_storage"]


def resolve_storage(storage: Any) -> Any:
    """Wrap a ``sqlite:``-scheme URI in a ``NullPool``-backed ``RDBStorage``.

    Anything else (an in-memory/other-DB URI, or an already-constructed
    storage object) passes through unchanged -- this only touches the one
    combination (string + sqlite) that both accounts for every storage this
    project actually uses and is where the leaked connection happens.
    """
    if not isinstance(storage, str) or not storage.startswith("sqlite"):
        return storage
    import optuna
    from sqlalchemy.pool import NullPool

    return optuna.storages.RDBStorage(storage, engine_kwargs={"poolclass": NullPool})
