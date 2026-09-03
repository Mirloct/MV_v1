from __future__ import annotations

import ast
import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPLEXITY_LIMIT = 10


def cyclomatic_complexity(function: ast.AST) -> int:
    complexity = 1
    decision_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)
    for node in ast.walk(function):
        if isinstance(node, decision_nodes):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += max(0, len(node.values) - 1)
        elif isinstance(node, ast.comprehension):
            complexity += 1 + len(node.ifs)
        elif isinstance(node, ast.Match):
            complexity += max(0, len(node.cases) - 1)
    return complexity


def _same_expression(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _assert_violation(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Assert):
        return None
    if isinstance(node.test, ast.Constant) and node.test.value is True:
        return f"line {node.lineno}: assert True"
    return None


def _call_violation(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr == "assertTrue" and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value is True:
            return f"line {node.lineno}: assertTrue(True)"
    if node.func.attr not in {"assertEqual", "assertAlmostEqual"} or len(node.args) < 2:
        return None
    if _same_expression(node.args[0], node.args[1]):
        return f"line {node.lineno}: self-equality assertion"
    return None


def tautology_violations(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        violation = _assert_violation(node) or _call_violation(node)
        if violation:
            violations.append(violation)
    return violations


def _complexity_violations(paths: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                score = cyclomatic_complexity(node)
                if score > COMPLEXITY_LIMIT:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name} complexity={score}")
    return violations


def _test_violations(paths: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        for violation in tautology_violations(path.read_text(encoding="utf-8")):
            violations.append(f"{path.relative_to(ROOT)}: {violation}")
    return violations


def _placeholder_violations(paths: list[Path]) -> list[str]:
    forbidden = ("TODO" + ":", "FIXME" + ":", "NotImplemented" + "Error")
    violations: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                violations.append(f"{path.relative_to(ROOT)} contains forbidden placeholder {token}")
    return violations


def _run_tests() -> int:
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        check=False,
    )
    return process.returncode


def _run_mutation_probe() -> int:
    process = subprocess.run(
        [sys.executable, "scripts/mutation_probe.py"],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        check=False,
    )
    return process.returncode


def main() -> int:
    source_paths = sorted((ROOT / "src").rglob("*.py"))
    script_paths = sorted((ROOT / "scripts").rglob("*.py"))
    test_paths = sorted((ROOT / "tests").glob("test_*.py"))
    violations = _complexity_violations([*source_paths, *script_paths])
    violations.extend(_test_violations(test_paths))
    violations.extend(_placeholder_violations(source_paths))
    compiled = compileall.compile_dir(ROOT / "src", quiet=1)
    tests_ok = _run_tests() == 0
    mutations_ok = _run_mutation_probe() == 0
    if violations:
        print("QUALITY VIOLATIONS")
        print("\n".join(f"- {item}" for item in violations))
    print(
        f"compile={compiled} tests={tests_ok} mutations={mutations_ok} "
        f"complexity_limit={COMPLEXITY_LIMIT}"
    )
    return 0 if compiled and tests_ok and mutations_ok and not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
