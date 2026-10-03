"""Pre-flight check that every package `requirements.txt` floors is actually
installed at or above that floor, run once at the very start of the pipeline
(`main.run_pipeline`) -- before any phase does real work.

By explicit design (and explicit user request, 2026-10-02): a missing or
outdated dependency is fixed automatically, in code, by installing it via
`pip` -- it never stops the run with an error for a human to act on
afterwards. `auto_install=True` is the default for exactly that reason.
Passing `auto_install=False` switches to check-only (log the problem and the
`pip install` command, let the caller decide) for the rare case where
mutating the active Python environment is not wanted at all (`main.py`'s
`--no-auto-install-deps`).

Only plain ``name>=X.Y`` lines are understood (every line in this project's
own `requirements.txt` today); anything else (a different operator, a
comment, a blank line) is skipped rather than guessed at.
"""

from __future__ import annotations

import os
import re
import sys
from importlib import metadata
from typing import NamedTuple, Optional

__all__ = ["DependencyIssue", "check_dependencies"]

_REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*>=\s*([0-9][0-9A-Za-z.\-]*)\s*$"
)


class DependencyIssue(NamedTuple):
    name: str
    required: str
    installed: Optional[str]  # None == not installed at all


def _parse_requirements(path: str) -> list[tuple[str, str]]:
    reqs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = _REQUIREMENT_RE.match(line)
            if m:
                reqs.append((m.group(1), m.group(2)))
    return reqs


def _version_tuple(version: str) -> tuple:
    """Leading dotted-numeric run only (``"2.2.1rc1"`` -> ``(2, 2, 1)``, the
    ``rc1`` suffix never contributes a ``1`` of its own) -- enough to compare
    against this project's plain ``>=X.Y`` floors without a hard dependency on
    the `packaging` library just to run this check."""
    m = re.match(r"\d+(?:\.\d+)*", version.split("+", 1)[0])
    return tuple(int(p) for p in m.group(0).split(".")) if m else (0,)


def _find_issues(requirements: list) -> list:
    issues: list[DependencyIssue] = []
    for name, required in requirements:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            issues.append(DependencyIssue(name, required, None))
            continue
        if _version_tuple(installed) < _version_tuple(required):
            issues.append(DependencyIssue(name, required, installed))
    return issues


def _pip_install_command(issues: list) -> str:
    pins = " ".join(f'"{i.name}>={i.required}"' for i in issues)
    return f'"{sys.executable}" -m pip install --upgrade {pins}'


def check_dependencies(
    requirements_path: str,
    *,
    logger=None,
    auto_install: bool = True,
) -> bool:
    """``True`` when every package in ``requirements_path`` ends up installed
    at or above its floor version.

    With ``auto_install=True`` (the default), a missing/outdated package is
    installed directly via ``pip`` -- the run is never stopped over something
    this function can fix itself. ``auto_install=False`` only logs the
    problem and the exact ``pip install`` command, leaving it to the caller.

    A missing ``requirements_path`` only warns and returns ``True`` -- an
    unusual deployment without the manifest is not this check's problem to
    solve, and should not block a run that may otherwise be fine.
    """
    from src.utils.logging_config import setup_logging

    log = logger or setup_logging()
    if not os.path.isfile(requirements_path):
        log.warning("Dependency check skipped: %s not found.", requirements_path)
        return True

    requirements = _parse_requirements(requirements_path)
    issues = _find_issues(requirements)

    if not issues:
        log.info(
            "Dependency check: %d required package(s) from %s all satisfy "
            "their minimum version.",
            len(requirements), os.path.basename(requirements_path),
        )
        return True

    for issue in issues:
        state = (
            "no instalado" if issue.installed is None
            else f"instalado {issue.installed}"
        )
        log.warning(
            "Dependencia insuficiente: %s requiere >= %s (%s).",
            issue.name, issue.required, state,
        )
    command = _pip_install_command(issues)

    if not auto_install:
        log.error("Instala las dependencias faltantes con:\n    %s", command)
        return False

    log.warning(
        "Instalando automáticamente %d dependencia(s) en %s:\n    %s",
        len(issues), sys.executable, command,
    )
    import subprocess

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             *[f"{i.name}>={i.required}" for i in issues]],
            capture_output=True, text=True, timeout=1800,
        )
    except Exception as exc:  # noqa: BLE001 - a failed install must not raise here
        log.error(
            "No se pudo ejecutar pip (%s). Instala manualmente con:\n    %s",
            exc, command,
        )
        return False
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        log.error(
            "pip devolvió código %d al instalar dependencias (%s). Instala "
            "manualmente con:\n    %s",
            completed.returncode, " | ".join(tail[-5:]), command,
        )
        return False

    # A 0 exit code from pip does not guarantee the resolver landed on a
    # version that actually satisfies our floor (a conflicting pin elsewhere
    # in the environment could win) -- trust the installed metadata after the
    # fact, not the exit code alone.
    still_bad = _find_issues([(i.name, i.required) for i in issues])
    if still_bad:
        log.error(
            "pip terminó sin error pero %d paquete(s) siguen sin satisfacer su "
            "mínimo tras instalar (%s). Instala manualmente con:\n    %s",
            len(still_bad),
            ", ".join(f"{i.name} (queda {i.installed or 'no instalado'})"
                      for i in still_bad),
            command,
        )
        return False

    log.info("Dependencias instaladas correctamente: %s",
              ", ".join(f"{i.name}>={i.required}" for i in issues))
    return True
