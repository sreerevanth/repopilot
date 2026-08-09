"""
Tests for the shape of the agent loop (modules/agent_loop.py).

`_run` was 551 lines with 52 branch points and six levels of nesting. The two
parts that genuinely separate are now methods:

- `_baseline_coverage` — measures coverage before the model touches anything.
  Reads the config, runs the suite once, returns one number.
- `_finalize` — push, PR, rollback, outcome message. Needs three values and
  nothing else.

The iteration body is deliberately left alone. It threads roughly forty locals
between phases — timings, the record, the last execution, the applied changes —
and extracting it means either a forty-parameter signature or a state object
that makes the data flow harder to follow rather than easier.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.agent_loop import AutonomousAgent  # noqa: E402

SOURCE = (ROOT / "modules" / "agent_loop.py").read_text(encoding="utf-8")


def method(name):
    tree = ast.parse(SOURCE)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AutonomousAgent"
    )
    return next(
        m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == name
    )


def nesting(node, depth=0):
    best = depth
    for child in ast.iter_child_nodes(node):
        deeper = depth + 1 if isinstance(
            child, (ast.If, ast.For, ast.While, ast.Try, ast.With)
        ) else depth
        best = max(best, nesting(child, deeper))
    return best


def length(name):
    node = method(name)
    return node.end_lineno - node.lineno + 1


# ── the extracted methods exist ───────────────────────────────────────────


@pytest.mark.parametrize("name", ["_baseline_coverage", "_finalize"])
def test_the_method_exists(name):
    assert callable(getattr(AutonomousAgent, name, None))


def test_the_baseline_returns_one_value():
    """
    A first attempt also extracted branch setup and resume state. That block
    initialises `last_exec`, `last_changes`, `outcome` and `iterations_used`,
    which the loop then reads — so extracting it left seven undefined names
    that every test still passed over. Narrowed to the part that genuinely
    stands alone.
    """
    assert "baseline_coverage = self._baseline_coverage()" in SOURCE


def test_finalize_takes_only_what_it_needs():
    node = method("_finalize")
    args = [a.arg for a in node.args.args]

    assert args == [
        "self", "outcome", "iterations_used", "last_apply_results",
        "all_apply_results",
    ]


# ── the loop got shorter ──────────────────────────────────────────────────


def test_the_loop_is_shorter_than_it_was():
    """551 lines before."""
    assert length("_run") < 500


def test_the_extracted_methods_are_readable_on_their_own():
    assert length("_baseline_coverage") < 40
    # 110 lines: push, PR, rollback, the change summary from #156 and the
    # outcome message. Long, but linear -- it is a sequence of end-of-run
    # steps rather than nested logic, and each reads on its own.
    assert length("_finalize") < 130


def test_the_call_sites_are_single_lines():
    assert "baseline_coverage = self._baseline_coverage()" in SOURCE
    assert "return self._finalize(" in SOURCE


# ── nothing was hidden rather than removed ────────────────────────────────


def test_the_extracted_work_still_happens():
    """
    Moving code into a method it is never called from would shorten `_run` and
    break the run. Both call sites are asserted above; these check the bodies
    came with them.
    """
    baseline = ast.unparse(method("_baseline_coverage"))
    finalize = ast.unparse(method("_finalize"))

    assert "parse_coverage_percent" in baseline
    assert "pytest-cov" in baseline
    assert "create_github_pr" in finalize
    assert "rollback" in finalize


def test_the_loop_still_iterates():
    assert "for iteration in range(start_iteration, cfg.max_iterations + 1):" in SOURCE


# ── the honest limit ──────────────────────────────────────────────────────


def test_the_iteration_body_was_not_extracted():
    """
    Recorded deliberately. The issue proposes `_iterate` alongside `_prepare`
    and `_finalize`; the iteration body threads too many locals between phases
    for that to be an improvement, and pretending otherwise in a later refactor
    would be a regression dressed as tidying.
    """
    assert not hasattr(AutonomousAgent, "_iterate")


def test_the_loop_is_still_the_complex_part():
    """
    Honest measurement rather than a claim of success. Nesting is unchanged --
    this reduced length, not depth.
    """
    assert nesting(method("_run")) >= nesting(method("_baseline_coverage"))
