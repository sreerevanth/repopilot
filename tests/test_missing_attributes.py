"""
Tests for two attributes that were called and never existed
(modules/agent_loop.py, modules/llm_client.py).

Both were found by mypy, and both are live crashes rather than style problems:

- `self._coverage_feedback(...)` — a run with `--coverage` raised AttributeError
  the moment a coverage percentage parsed
- `self.llm.usage.summary()` — a run that correctly stopped at its `--max-cost`
  limit then crashed while building the message explaining why

Neither shows up in normal use, because both sit on optional paths that the
test suite never exercised end to end.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import COVERAGE_TOLERANCE, AutonomousAgent  # noqa: E402
from modules.llm_client import BaseLLMClient  # noqa: E402


class Stub(BaseLLMClient):
    def _call(self, prompt):
        return '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'


@pytest.fixture
def feedback():
    """_coverage_feedback bound to a bare object; it uses no instance state."""
    holder = type("Holder", (), {})()
    return AutonomousAgent._coverage_feedback.__get__(holder)


# ── the attributes exist ──────────────────────────────────────────────────


def test_the_coverage_helper_exists():
    assert callable(getattr(AutonomousAgent, "_coverage_feedback", None))


def test_the_usage_summary_exists():
    assert callable(getattr(BaseLLMClient, "usage_summary", None))


# ── the coverage gate ─────────────────────────────────────────────────────


def test_a_real_drop_is_reported(feedback):
    message = feedback(80.0, 72.0)

    assert message is not None
    assert "80.0%" in message and "72.0%" in message


def test_unchanged_coverage_does_not_gate(feedback):
    assert feedback(80.0, 80.0) is None


def test_improved_coverage_does_not_gate(feedback):
    assert feedback(80.0, 85.0) is None


def test_rounding_noise_does_not_gate(feedback):
    """
    Coverage is reported to one decimal place and moves slightly between runs.
    Failing for 0.2% would be noise rather than a signal.
    """
    assert feedback(80.0, 79.8) is None


def test_a_drop_past_the_tolerance_does_gate(feedback):
    assert feedback(80.0, 80.0 - COVERAGE_TOLERANCE - 0.1) is not None


def test_the_tolerance_is_small_enough_to_be_useful():
    """Wide enough to absorb rounding, narrow enough to still catch a regression."""
    assert 0 < COVERAGE_TOLERANCE <= 1.0


@pytest.mark.parametrize(
    "baseline,current", [(None, 75.0), (80.0, None), (None, None)]
)
def test_an_impossible_comparison_does_not_gate(feedback, baseline, current):
    """
    Treating an unparseable figure as a drop would fail runs for a reporting
    quirk rather than for anything the model did.
    """
    assert feedback(baseline, current) is None


def test_the_message_names_the_failure_mode(feedback):
    """
    The wording is the pre-existing suite's, and it is sharper than a generic
    "add tests": the concern is a model deleting a failing test to make the
    run go green, which coverage is exactly what detects.
    """
    message = feedback(90.0, 70.0)

    assert "Restore the deleted tests" in message
    assert "deleting a failing test is not fixing it" in message


# ── the usage summary ─────────────────────────────────────────────────────


def test_it_reports_tokens_and_cost():
    client = Stub()
    client.initial_request("task", "context" * 200)

    summary = client.usage_summary()

    assert "input" in summary and "output" in summary and "$" in summary


def test_a_fresh_client_reports_zero():
    """Called when a run stops early, which can happen before any request."""
    summary = Stub().usage_summary()

    assert "0 input" in summary
    assert "$0.0000" in summary


def test_the_figures_grow_with_use():
    client = Stub()
    client.initial_request("task", "context")
    first = client.total_cost
    client.initial_request("task", "context")

    assert client.total_cost > first
    assert f"${client.total_cost:.4f}" in client.usage_summary()


def test_the_budget_message_can_be_built():
    """
    The exact expression that used to crash. The run stopping is correct; the
    crash was in explaining why.
    """
    client = Stub()
    client.initial_request("task", "context")

    assert f"Stopped after {client.usage_summary()}."


def test_the_loop_calls_the_method_that_exists():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "self.llm.usage_summary()" in source
    assert "self.llm.usage.summary()" not in source
