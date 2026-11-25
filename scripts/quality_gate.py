from __future__ import annotations

import argparse
import ast
import json
import subprocess
import tokenize
from pathlib import Path

from radon.visitors import ComplexityVisitor


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 10
BASELINE = "scripts/complexity_baseline.json"


def source_paths(root: Path) -> list[Path]:
    paths = list(root.glob("*.py"))
    for directory in ("src", "scripts", "tests"):
        paths.extend((root / directory).rglob("*.py"))
    return sorted(paths)


class FunctionScores(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.scores = {}
        self.occurrences = {}

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        name = ".".join([*self.scope, node.name])
        self.occurrences[name] = self.occurrences.get(name, 0) + 1
        if self.occurrences[name] > 1:
            name = f"{name}#{self.occurrences[name]}"
        self.scores[name] = ComplexityVisitor.from_ast(node).functions[0].complexity
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


def complexity_scores(tree: ast.AST, filename: str) -> dict[str, int]:
    visitor = FunctionScores()
    visitor.visit(tree)
    return {f"{filename}::{name}": score for name, score in visitor.scores.items()}


def load_baseline(source: str) -> dict[str, int]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate baseline key: {key}")
            result[key] = value
        return result

    entries = json.loads(source, object_pairs_hook=unique_pairs)
    if not isinstance(entries, dict):
        raise ValueError("baseline must be a JSON object")
    for name, score in entries.items():
        if "::" not in name or type(score) is not int or score <= LIMIT:
            raise ValueError(f"invalid baseline entry: {name}={score!r}")
    return entries


def complexity_violations(scores: dict[str, int], baseline: dict[str, int]) -> list[str]:
    violations = []
    for name, score in scores.items():
        ceiling = baseline.get(name, LIMIT)
        if score > ceiling:
            violations.append(f"{name}: complexity {score} exceeds {ceiling}")
    for name, ceiling in baseline.items():
        current = scores.get(name)
        if current is None or current < ceiling:
            violations.append(f"{name}: stale debt {ceiling}; ratchet baseline to {current}")
    return violations


def _same_expression(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _tautological(expression: ast.AST) -> bool:
    if isinstance(expression, ast.Constant):
        return bool(expression.value)
    if not isinstance(expression, ast.Compare):
        return False
    pairs = zip([expression.left, *expression.comparators], expression.comparators, expression.ops)
    return all(isinstance(op, (ast.Eq, ast.Is)) and _same_expression(a, b) for a, b, op in pairs)


def _assertion_violation(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assert) and _tautological(node.test):
        return "tautological assert"
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    arguments = _assertion_arguments(node)
    if node.func.attr == "assertTrue" and arguments and _tautological(arguments[0]):
        return "tautological assertTrue"
    return _equality_violation(node.func.attr, arguments)


def _assertion_arguments(node: ast.Call) -> list[ast.AST]:
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    names = ("expr",) if node.func.attr == "assertTrue" else ("first", "second")
    arguments = list(node.args)
    for key in names[len(arguments):]:
        if key in keywords:
            arguments.append(keywords[key])
    return arguments


def _equality_violation(method: str, arguments: list[ast.AST]) -> str | None:
    equality = {"assertEqual", "assertAlmostEqual", "assertSequenceEqual", "assertListEqual",
                "assertTupleEqual", "assertDictEqual", "assertSetEqual", "assertMultiLineEqual",
                "assertCountEqual", "assertIs"}
    if method in equality and len(arguments) >= 2:
        if _same_expression(arguments[0], arguments[1]):
            return "self-equality assertion"
    return None


def _empty_statement(node: ast.AST) -> bool:
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Constant)
    if isinstance(node, ast.Return):
        return node.value is None or isinstance(node.value, ast.Constant)
    return False


def tautology_violations(tree: ast.AST) -> list[str]:
    violations = []
    for node in ast.walk(tree):
        violation = _assertion_violation(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test") and all(map(_empty_statement, node.body)):
                violation = "empty test"
        if violation:
            violations.append(f"line {node.lineno}: {violation}")
    return violations


def scan(root: Path) -> tuple[dict[str, int], list[str]]:
    scores, violations = {}, []
    for path in source_paths(root):
        name = path.relative_to(root).as_posix()
        with tokenize.open(path) as stream:
            tree = ast.parse(stream.read(), filename=name)
        scores.update(complexity_scores(tree, name))
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            violations.extend(f"{name}: {item}" for item in tautology_violations(tree))
    return scores, violations


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, encoding="utf-8").stdout


def reference_baseline(root: Path, ref: str) -> dict[str, int]:
    ref = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    names = _git(root, "ls-tree", "-r", "--name-only", ref).splitlines()
    if BASELINE in names:
        return load_baseline(_git(root, "show", f"{ref}:{BASELINE}"))
    scores = {}
    for name in names:
        path = Path(name)
        if path.suffix == ".py" and (len(path.parts) == 1 or path.parts[0] in {"src", "scripts", "tests"}):
            tree = ast.parse(_git(root, "show", f"{ref}:{name}"), filename=name)
            scores.update(complexity_scores(tree, name))
    return {name: score for name, score in scores.items() if score > LIMIT}


def baseline_violations(baseline: dict[str, int], reference: dict[str, int]) -> list[str]:
    return [f"{name}: baseline debt {score} exceeds reference {reference.get(name, LIMIT)}"
            for name, score in baseline.items() if score > reference.get(name, LIMIT)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base-ref", help="Git revision protecting the baseline against added debt")
    args = parser.parse_args(argv)
    try:
        baseline = load_baseline((args.root / BASELINE).read_text(encoding="utf-8"))
        scores, violations = scan(args.root)
        violations.extend(complexity_violations(scores, baseline))
        if args.base_ref:
            violations.extend(baseline_violations(baseline, reference_baseline(args.root, args.base_ref)))
    except (OSError, SyntaxError, ValueError, subprocess.CalledProcessError) as error:
        print(f"QUALITY ERROR: {error}")
        return 1
    print(f"Radon ceiling={LIMIT}; scope=root/*.py, src, scripts, tests")
    print(f"Legacy debt: {len(baseline)} functions above {LIMIT}; NOT compliant, ratchet exceptions only.")
    for name, score in sorted(baseline.items()):
        print(f"DEBT {name}={score}")
    for violation in violations:
        print(f"REJECT {violation}")
    print("QUALITY FAILED" if violations else "QUALITY PASSED (legacy debt remains)" if baseline else "QUALITY PASSED")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
