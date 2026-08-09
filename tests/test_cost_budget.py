"""
Tests for --max-cost (modules/llm_client.py).

`--max-cost` was registered in argparse and appeared in `--help`, but the value
was never read — `max_cost` existed nowhere else in the codebase. Someone
setting a spend limit got no protection at all, which is worse than the flag not
existing, because they believed they were covered.

An earlier version of this file tested a `UsageTracker`/`PRICING_PER_MTOK`
design that the multi-provider refactor replaced. Rewritten against the cost
tracking that actually shipped: `MODEL_PRICING` and `BaseLLMClient`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.llm_client import (  # noqa: E402
    MODEL_PRICING,
    BaseLLMClient,
    BudgetExceededError,
)

RESPONSE = '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'


class Stub(BaseLLMClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def _call(self, prompt):
        self.calls += 1
        if "Do NOT write any code" in prompt:
            return '{"plan":["a"],"files_to_change":[],"risks":[],"confidence":0.8}'
        return RESPONSE


# ── pricing ───────────────────────────────────────────────────────────────


def test_known_models_are_priced():
    assert MODEL_PRICING["claude-sonnet-4-20250514"] == (3.00, 15.00)


def test_cost_is_accumulated_across_calls():
    stub = Stub()
    stub.initial_request("task", "context")
    first = stub.total_cost
    stub.initial_request("task", "context")

    assert stub.total_cost > first > 0


def test_tokens_are_accumulated():
    stub = Stub()
    stub.initial_request("task", "context")

    assert stub.input_tokens_used > 0
    assert stub.output_tokens_used > 0


def test_an_unknown_model_costs_nothing_rather_than_raising():
    """
    A self-hosted or brand-new model should still run. Reporting zero is
    wrong-but-harmless; raising would block the run entirely.
    """
    stub = Stub(model="some-model-released-tomorrow")
    stub.initial_request("task", "context")

    assert stub.total_cost == 0.0


def test_a_larger_prompt_costs_more():
    small, large = Stub(), Stub()
    small.initial_request("task", "c")
    large.initial_request("task", "c" * 20_000)

    assert large.total_cost > small.total_cost


# ── enforcement ───────────────────────────────────────────────────────────


def test_no_limit_means_no_stopping():
    stub = Stub(max_cost=None)
    for _ in range(5):
        stub.initial_request("task", "context")

    assert stub.calls == 5


def test_the_limit_stops_the_run():
    stub = Stub(max_cost=0.001)

    with pytest.raises(BudgetExceededError):
        for _ in range(50):
            stub.initial_request("task", "c" * 4_000)


def test_the_error_reports_both_figures():
    """So the message says what was spent and what the limit was."""
    stub = Stub(max_cost=0.001)

    with pytest.raises(BudgetExceededError) as excinfo:
        for _ in range(50):
            stub.initial_request("task", "c" * 4_000)

    message = str(excinfo.value)
    assert "$" in message and "max-cost" in message


def test_the_check_happens_before_spending():
    """
    A limit checked after the call would report overspend rather than prevent
    it. Cost is only known once a response arrives, so a run can overshoot by
    at most one call — but not by two.
    """
    stub = Stub(max_cost=0.0001)

    with pytest.raises(BudgetExceededError):
        for _ in range(10):
            stub.initial_request("task", "c" * 4_000)

    assert stub.calls == 1


def test_a_generous_limit_does_not_interfere():
    stub = Stub(max_cost=1000.0)
    for _ in range(3):
        stub.initial_request("task", "context")

    assert stub.calls == 3


def test_retries_are_checked_too():
    """A retry is a paid call like any other."""
    stub = Stub(max_cost=0.001)

    with pytest.raises(BudgetExceededError):
        for _ in range(50):
            stub.retry_request("task", "c" * 4_000, [], "out", "err", 1)


# ── wiring ────────────────────────────────────────────────────────────────


def test_no_limit_by_default():
    assert AgentConfig(repo_root=".", task="t").max_cost is None


def test_the_loop_passes_the_limit_to_the_client():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "max_cost=cfg.max_cost" in source


def test_main_passes_the_flag_to_the_config():
    source = (
        Path(__file__).resolve().parents[1] / "main.py"
    ).read_text(encoding="utf-8")

    assert "max_cost=args.max_cost" in source


def test_the_facade_forwards_the_limit():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "llm_client.py"
    ).read_text(encoding="utf-8")

    assert "self.underlying_client.max_cost = max_cost" in source


# ── planning is a paid call too ───────────────────────────────────────────


def test_a_planning_call_is_checked_against_the_budget():
    """
    plan_request previously called _call directly, so --plan-first spent money
    the budget never saw. Raised in review of #204.
    """
    stub = Stub(max_cost=0.0001)
    stub.plan_request("task", "c" * 4_000)

    with pytest.raises(BudgetExceededError):
        stub.plan_request("task", "c" * 4_000)


def test_a_planning_call_counts_toward_the_total():
    """
    Without this the next execution call can exceed the limit by more than one
    request, because the planning spend is invisible to the check.
    """
    stub = Stub()
    stub.plan_request("task", "c" * 2_000)

    assert stub.total_cost > 0


def test_every_request_type_goes_through_one_accounted_path():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "llm_client.py"
    ).read_text(encoding="utf-8")

    assert "def _accounted_call(" in source
    assert source.count("self._accounted_call(prompt)") >= 3


def test_usage_is_not_double_counted():
    """
    _accounted_call records usage; _parse_response used to record it again.
    Two identical calls must cost exactly twice one call.
    """
    stub = Stub()
    stub.initial_request("task", "c" * 1_000)
    one = stub.total_cost
    stub.initial_request("task", "c" * 1_000)

    assert abs(stub.total_cost - 2 * one) < 1e-9
