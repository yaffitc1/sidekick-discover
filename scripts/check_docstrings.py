#!/usr/bin/env python3
"""Fail if public modules/classes/functions lack docstrings.

Rules:
- Public = names not starting with underscore.
- Module docstring required for each file in `discovery/`.
- Class and function docstrings required for public ones.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Tuple


def find_python_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.py") if "/.venv/" not in str(p)]


def is_public(name: str) -> bool:
    return not name.startswith("_")


def check_file(path: Path) -> List[Tuple[str, int, str]]:
    errors: List[Tuple[str, int, str]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append((str(path), 0, f"parse error: {exc}"))
        return errors

    # module docstring
    if ast.get_docstring(tree) in (None, ""):
        errors.append((str(path), 1, "missing module docstring"))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if is_public(node.name) and not ast.get_docstring(node):
                errors.append((str(path), node.lineno, f"function '{node.name}' missing docstring"))
        elif isinstance(node, ast.AsyncFunctionDef):
            if is_public(node.name) and not ast.get_docstring(node):
                errors.append((str(path), node.lineno, f"async function '{node.name}' missing docstring"))
        elif isinstance(node, ast.ClassDef):
            if is_public(node.name) and not ast.get_docstring(node):
                errors.append((str(path), node.lineno, f"class '{node.name}' missing docstring"))

    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "discovery"
    files = find_python_files(root)
    all_errors: List[Tuple[str, int, str]] = []
    for f in files:
        all_errors.extend(check_file(f))
    if all_errors:
        for file, line, msg in all_errors:
            print(f"{file}:{line}: {msg}")
        print(f"Docstring check failed: {len(all_errors)} issue(s)")
        return 1
    print("Docstring check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







