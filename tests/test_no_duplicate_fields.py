"""
Tests against duplicate declarations (all modules).

`AgentConfig` declared four fields twice — `plan_first`, `resume_from`,
`skip_tests` and `yes`. A dataclass keeps the last declaration and discards the
earlier one silently, so the class imported, constructed and behaved correctly.

Nothing caught it. The tests pass, ruff does not flag it, and the import guard
is green, because a duplicate annotation is valid Python. mypy is what surfaced
it, with `Name "resume_from" already defined`.

This is the same merge damage that duplicated `_apply_unified_diff` in
`code_modifier.py`, in a form a function-and-class check does not reach.
"""

import ast
import collections
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKIP_DIRS = {"node_modules", "__pycache__", "logs", "backups", ".git"}


def python_files():
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(root) / name


def duplicates(path):
    """Names declared more than once at the same level in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    found = []

    top = collections.Counter(
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    found += [f"{name} ({count}x)" for name, count in top.items() if count > 1]

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        methods = collections.Counter(
            item.name for item in node.body if isinstance(item, ast.FunctionDef)
        )
        found += [
            f"{node.name}.{name} ({count}x)"
            for name, count in methods.items() if count > 1
        ]

        # Annotated class attributes -- dataclass fields among them. A duplicate
        # here is silent: the last declaration wins and the earlier one vanishes.
        attributes = collections.Counter(
            item.target.id for item in node.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        )
        found += [
            f"{node.name}.{name} ({count}x)"
            for name, count in attributes.items() if count > 1
        ]

    return found


# ── the check ─────────────────────────────────────────────────────────────


def test_nothing_is_declared_twice():
    offenders = {
        str(path.relative_to(ROOT)): found
        for path in python_files()
        if (found := duplicates(path))
    }

    assert offenders == {}, f"duplicate declarations: {offenders}"


def test_agent_config_declares_each_field_once():
    """The specific case: four fields were declared twice."""
    tree = ast.parse((ROOT / "modules" / "agent_loop.py").read_text(encoding="utf-8"))
    config = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentConfig"
    )
    names = [
        node.target.id for node in config.body if isinstance(node, ast.AnnAssign)
    ]

    assert len(names) == len(set(names))


# ── the detector works ────────────────────────────────────────────────────


def test_a_duplicate_dataclass_field_is_detected(tmp_path):
    """Otherwise the test above passes for the wrong reason."""
    path = tmp_path / "dupe.py"
    path.write_text("class C:\n    a: int = 1\n    b: str = 'x'\n    a: int = 2\n")

    assert duplicates(path) == ["C.a (2x)"]


def test_a_duplicate_function_is_detected(tmp_path):
    path = tmp_path / "dupe.py"
    path.write_text("def f():\n    pass\n\n\ndef f():\n    pass\n")

    assert duplicates(path) == ["f (2x)"]


def test_a_duplicate_method_is_detected(tmp_path):
    path = tmp_path / "dupe.py"
    path.write_text("class C:\n    def m(self):\n        pass\n\n    def m(self):\n        pass\n")

    assert duplicates(path) == ["C.m (2x)"]


def test_a_clean_file_reports_nothing(tmp_path):
    path = tmp_path / "clean.py"
    path.write_text("class C:\n    a: int = 1\n    b: str = 'x'\n\n    def m(self):\n        pass\n")

    assert duplicates(path) == []


def test_the_same_field_name_in_two_classes_is_fine(tmp_path):
    """Two dataclasses may both have a `path` field; that is not a duplicate."""
    path = tmp_path / "ok.py"
    path.write_text("class A:\n    path: str = ''\n\n\nclass B:\n    path: str = ''\n")

    assert duplicates(path) == []


def test_an_unparseable_file_is_skipped(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("class C(:\n")

    assert duplicates(path) == []


# ── behaviour is unchanged ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field,expected",
    [("plan_first", False), ("resume_from", None), ("skip_tests", False), ("yes", False)],
)
def test_the_deduplicated_fields_keep_their_defaults(field, expected):
    """
    A dataclass keeps the last declaration, so removing the earlier duplicates
    must leave every default exactly as it was.
    """
    from modules.agent_loop import AgentConfig

    assert getattr(AgentConfig(repo_root=".", task="t"), field) == expected


def test_no_field_was_lost():
    from dataclasses import fields

    from modules.agent_loop import AgentConfig

    # 49 on current main; rises as flags are added. Pinned so a field
    # cannot be dropped unnoticed by a merge, not to freeze the count.
    assert len(fields(AgentConfig)) == 49
