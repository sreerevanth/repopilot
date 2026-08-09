"""
Tests against duplicate definitions (all modules).

`_apply_unified_diff` was defined twice in `modules/code_modifier.py`, 56 lines
each, byte-identical. The second shadowed the first silently: the module
imported, every test passed, and nothing indicated a problem.

That is what a merge landing the same block twice looks like, and this repository
has had several. A shadowed definition is harmless only while the copies agree —
the moment one is edited, the edit has no effect and the reason is invisible.
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
        dirs[:] = [
            d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS
        ]
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(root) / name


def duplicates(path):
    """Top-level and per-class names defined more than once in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    found = []
    counts = collections.Counter(
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    found += [f"{name} ({count}x)" for name, count in counts.items() if count > 1]

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

    return found


# ── the check ─────────────────────────────────────────────────────────────


def test_no_module_defines_anything_twice():
    offenders = {
        str(path.relative_to(ROOT)): found
        for path in python_files()
        if (found := duplicates(path))
    }

    assert offenders == {}, f"duplicate definitions: {offenders}"


# ── the detector itself works ─────────────────────────────────────────────


def test_a_duplicate_function_is_detected(tmp_path):
    """Otherwise the test above passes for the wrong reason."""
    path = tmp_path / "dupe.py"
    path.write_text("def f():\n    pass\n\n\ndef f():\n    pass\n")

    assert duplicates(path) == ["f (2x)"]


def test_a_duplicate_method_is_detected(tmp_path):
    path = tmp_path / "dupe.py"
    path.write_text("class C:\n    def m(self):\n        pass\n\n    def m(self):\n        pass\n")

    assert duplicates(path) == ["C.m (2x)"]


def test_a_clean_file_reports_nothing(tmp_path):
    path = tmp_path / "clean.py"
    path.write_text("def f():\n    pass\n\n\ndef g():\n    pass\n")

    assert duplicates(path) == []


def test_the_same_name_in_different_classes_is_fine(tmp_path):
    """Two classes may both define run; that is not a duplicate."""
    path = tmp_path / "ok.py"
    path.write_text(
        "class A:\n    def run(self):\n        pass\n\n\n"
        "class B:\n    def run(self):\n        pass\n"
    )

    assert duplicates(path) == []


def test_an_unparseable_file_is_skipped(tmp_path):
    """A syntax error is someone else's problem, not a reason to fail here."""
    path = tmp_path / "broken.py"
    path.write_text("def f(:\n")

    assert duplicates(path) == []


# ── the specific case that prompted this ──────────────────────────────────


def test_the_patch_helper_is_defined_once():
    path = ROOT / "modules" / "code_modifier.py"

    assert "_apply_unified_diff" not in " ".join(duplicates(path))


@pytest.mark.parametrize(
    "original,patch,expected",
    [
        (
            "line one\nline two\nline three\n",
            "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n line one\n-line two\n+line TWO\n line three\n",
            "line one\nline TWO\nline three\n",
        ),
        (
            "",
            "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+first\n+second\n",
            "first\nsecond\n",
        ),
    ],
    ids=["modify", "new-file"],
)
def test_patching_still_works(original, patch, expected):
    """
    Including the `@@ -0,0` form a new file produces, since that is the case
    #151 suspected of an off-by-one.
    """
    from modules.code_modifier import _apply_unified_diff

    assert _apply_unified_diff(original, patch) == expected
