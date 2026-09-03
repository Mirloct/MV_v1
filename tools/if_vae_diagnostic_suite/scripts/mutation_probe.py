from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutant:
    name: str
    file: str
    old: str
    new: str
    test: str


MUTANTS = [
    Mutant(
        "percentile_tie_direction",
        "ifvae_diag/scoring.py",
        'side="right"',
        'side="left"',
        "tests.test_scoring",
    ),
    Mutant(
        "if_quadrant_threshold_exclusive",
        "ifvae_diag/diagnostics.py",
        "if_high = np.asarray(if_percentile) >= threshold",
        "if_high = np.asarray(if_percentile) > threshold",
        "tests.test_diagnostics",
    ),
    Mutant(
        "vae_quadrant_threshold_exclusive",
        "ifvae_diag/diagnostics.py",
        "vae_high = np.asarray(vae_percentile) >= threshold",
        "vae_high = np.asarray(vae_percentile) > threshold",
        "tests.test_diagnostics",
    ),
    Mutant(
        "lift_formula",
        "ifvae_diag/metrics.py",
        "precision / prevalence",
        "precision * prevalence",
        "tests.test_metrics",
    ),
]


def _apply_mutant(source_root: Path, mutant: Mutant) -> None:
    path = source_root / mutant.file
    source = path.read_text(encoding="utf-8")
    if source.count(mutant.old) != 1:
        raise RuntimeError(f"Mutant {mutant.name} expected one source match")
    path.write_text(source.replace(mutant.old, mutant.new), encoding="utf-8")


def _mutant_is_killed(mutant: Mutant) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        source_root = Path(tmp) / "src"
        shutil.copytree(ROOT / "src", source_root)
        _apply_mutant(source_root, mutant)
        # `os.pathsep` (":" on POSIX, ";" on Windows) -- a hardcoded ":" here
        # silently breaks on Windows, where the drive-letter colon in a path
        # like "C:\\Users\\...\\src" combines with the intended separator
        # into one unparseable PYTHONPATH entry. Python then falls back to
        # the real, pip-installed (unmutated) package for every mutant,
        # so `_mutant_is_killed` always returns False regardless of whether
        # the tests would actually have caught the mutation -- a false
        # "gate is broken" reading, not a real test-quality problem. Root
        # cause found integrating this suite into Modelo v0.1's CI (Windows).
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(source_root), str(ROOT)])}
        process = subprocess.run(
            [sys.executable, "-m", "unittest", mutant.test, "-q"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return process.returncode != 0


def main() -> int:
    results = {mutant.name: _mutant_is_killed(mutant) for mutant in MUTANTS}
    for name, killed in results.items():
        print(f"mutant={name} killed={killed}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
