"""Standalone environment validator for the anomaly detection framework.

Checks the Python version and that all required third-party packages are
importable. Any missing package triggers a best-effort ``pip install``
attempt. Safe to run in an environment where some/all packages are missing -
it will not crash, only report and (if needed) exit non-zero at the end.

Usage:
    python setup_validator.py
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass

from src.utils.logging_config import setup_logging

REQUIRED_PYTHON_VERSION = (3, 12)

# Mapping of pip package name -> import module name (only where they differ).
# pip name is the dict key, used both for `pip install <key>` and for display.
PACKAGES: dict[str, str] = {
    "scikit-learn": "sklearn",
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "optuna": "optuna",
    "shap": "shap",
    "umap-learn": "umap",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "joblib": "joblib",
    "pyyaml": "yaml",
    # Parquet engine: without it the synthetic generator silently degrades the
    # ground-truth file to CSV (see src/data/synthetic.py::_write_ground_truth).
    "pyarrow": "pyarrow",
    # Excel engine for the OOT top-decile deliverable (pandas .to_excel).
    "openpyxl": "openpyxl",
    # Test suite runner (tests/).
    "pytest": "pytest",
}


@dataclass
class PackageResult:
    pip_name: str
    import_name: str
    initially_available: bool
    install_attempted: bool = False
    install_succeeded: bool | None = None
    final_available: bool = False


def check_python_version(logger) -> bool:
    current = sys.version_info[:2]
    ok = current >= REQUIRED_PYTHON_VERSION
    version_str = f"{current[0]}.{current[1]}"
    required_str = f"{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}"
    if ok:
        logger.info("Python version %s satisfies >=%s requirement.", version_str, required_str)
    else:
        logger.warning(
            "Python version %s is below the recommended >=%s. Continuing anyway.",
            version_str,
            required_str,
        )
    return ok


def is_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def attempt_install(pip_name: str, logger) -> bool:
    install_cmd = [sys.executable, "-m", "pip", "install", pip_name]
    logger.info("Attempting install: %s", " ".join(install_cmd))
    try:
        subprocess.run(install_cmd, check=True)
        return True
    except Exception as exc:
        logger.error("Install command failed for %s: %s", pip_name, exc)
        return False


def validate_packages(logger) -> list[PackageResult]:
    results: list[PackageResult] = []

    for pip_name, import_name in PACKAGES.items():
        available = is_importable(import_name)
        result = PackageResult(
            pip_name=pip_name,
            import_name=import_name,
            initially_available=available,
            final_available=available,
        )

        if available:
            logger.info("Package OK: %s (import %s)", pip_name, import_name)
        else:
            logger.warning(
                "Package MISSING: %s (import %s). Install with: pip install %s",
                pip_name,
                import_name,
                pip_name,
            )
            print(
                f"[MISSING] {pip_name} is not importable. "
                f"Install manually with: pip install {pip_name}"
            )
            result.install_attempted = True
            result.install_succeeded = attempt_install(pip_name, logger)

            if result.install_succeeded:
                # Re-check importability after install (fresh module cache lookup).
                importlib.invalidate_caches()
                result.final_available = is_importable(import_name)
            else:
                result.final_available = False

            if result.final_available:
                logger.info("Package %s successfully installed and importable.", pip_name)
            else:
                logger.error("Package %s could not be satisfied after install attempt.", pip_name)

        results.append(result)

    return results


def print_summary(results: list[PackageResult], python_ok: bool) -> None:
    header = f"{'Package':<15} {'Import name':<12} {'Initial':<10} {'Installed?':<12} {'Final':<8}"
    separator = "-" * len(header)
    print("\n" + "=" * len(header))
    print("SETUP VALIDATION SUMMARY")
    print("=" * len(header))
    print(f"Python version check passed: {python_ok}")
    print(separator)
    print(header)
    print(separator)
    for r in results:
        initial = "OK" if r.initially_available else "MISSING"
        if not r.install_attempted:
            installed = "n/a"
        else:
            installed = "yes" if r.install_succeeded else "no"
        final = "OK" if r.final_available else "FAILED"
        print(f"{r.pip_name:<15} {r.import_name:<12} {initial:<10} {installed:<12} {final:<8}")
    print(separator)

    failed = [r.pip_name for r in results if not r.final_available]
    if failed:
        print(f"UNSATISFIED PACKAGES: {', '.join(failed)}")
    else:
        print("All required packages are available.")
    print("=" * len(header))


def main() -> int:
    logger = setup_logging()
    logger.info("Starting environment validation.")

    python_ok = check_python_version(logger)
    results = validate_packages(logger)

    print_summary(results, python_ok)

    failed = [r for r in results if not r.final_available]
    if failed:
        names = ", ".join(r.pip_name for r in failed)
        logger.error("Environment validation FAILED. Unsatisfied packages: %s", names)
        return 1

    logger.info("Environment validation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
