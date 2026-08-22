"""Shared pytest configuration for the project test suite.

Two responsibilities, both of which future test modules rely on:

1. Make the project root importable (`import src...`) regardless of the
   working directory pytest was invoked from.
2. Run the whole session inside a throwaway sandbox directory. Several
   modules in this project resolve paths *relative to the current working
   directory* -- `src.utils.logging_config.setup_logging` writes to
   `logs/execution.log`, and `src.data.loader.load_or_generate_panel`
   defaults `data_path` to `data/data.csv` (its ground-truth file is now
   written next to `data_path`, so with the default it lands in `data/`).
   Chdir'ing into a sandbox for the session guarantees the real `data/` and
   `logs/` directories are never touched by tests.

The chdir happens at *collection* time (module import), before any test or
fixture runs, because `setup_logging` is idempotent per-process: whichever
call happens first pins the log file location for the rest of the session.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- session sandbox: entered at import time, before any logger is created ---
_ORIGINAL_CWD = os.getcwd()
_SANDBOX = Path(tempfile.mkdtemp(prefix="modelo_tests_"))
os.chdir(_SANDBOX)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Return to the original cwd so pytest's own teardown behaves."""
    os.chdir(_ORIGINAL_CWD)


@pytest.fixture(scope="session")
def sandbox() -> Path:
    """Session-wide scratch directory; the process cwd for the whole run."""
    return _SANDBOX


@pytest.fixture(scope="session", autouse=True)
def _assert_sandboxed(sandbox: Path) -> None:
    """Guard: fail loudly if anything escaped the sandbox."""
    assert Path(os.getcwd()) == sandbox, "tests must run inside the sandbox cwd"
