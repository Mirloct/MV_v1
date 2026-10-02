"""Pre-flight check that every package `requirements.txt` floors is actually
installed at or above that floor, run once at the very start of the pipeline
(`main.run_pipeline`) -- before any phase does real work -- so a missing or
outdated library fails fast with the exact fix, not an opaque ``ImportError``
several phases in.

Deliberately read-only by default: upgrading a third-party package in the
caller's Python environment is a side effect with real blast radius (it can
affect other projects sharing the same interpreter), so this only ever
prints/logs the `pip install` command that would fix every problem at once
and lets the caller decide -- ``auto_install=True`` (wired to
``--auto-install-deps``) is an explicit opt-in to actually run it.

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


def _pip_install_command(issues: list) -> str:
    pins = " ".join(f'"{i.name}>={i.required}"' for i in issues)
    return f'"{sys.executable}" -m pip install --upgrade {pins}'


def check_dependencies(
    requirements_path: str,
    *,
    logger=None,
    auto_install: bool = False,
) -> bool:
    """``True`` when every package in ``requirements_path`` is installed at or
    above its floor version.

    Logs every problem found plus the single ``pip install`` command that
    fixes all of them at once. With ``auto_install=True`` it also runs that
    command (see the module docstring for why that is opt-in, never the
    default) and returns whether the install succeeded.

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
    issues: list[DependencyIssue] = []
    for name, required in requirements:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            issues.append(DependencyIssue(name, required, None))
            continue
        if _version_tuple(installed) < _version_tuple(required):
            issues.append(DependencyIssue(name, required, installed))

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
        log.error(
            "Dependencia insuficiente: %s requiere >= %s (%s).",
            issue.name, issue.required, state,
        )
    command = _pip_install_command(issues)
    log.error("Instala las dependencias faltantes con:\n    %s", command)

    if not auto_install:
        return False

    log.warning(
        "--auto-install-deps: instalando automáticamente %d paquete(s) en "
        "%s (esto modifica el entorno Python activo).",
        len(issues), sys.executable,
    )
    import subprocess

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             *[f"{i.name}>={i.required}" for i in issues]],
            capture_output=True, text=True, timeout=1800,
        )
    except Exception as exc:  # noqa: BLE001 - a failed install must not raise here
        log.error("No se pudo ejecutar pip: %s", exc)
        return False
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        log.error(
            "pip devolvió código %d al instalar dependencias: %s",
            completed.returncode, " | ".join(tail[-5:]),
        )
        return False
    log.info("Dependencias instaladas correctamente: %s",
              ", ".join(f"{i.name}>={i.required}" for i in issues))
    return True
