"""
Tests for modules/token_tracker.py.

This module had no tests. It feeds cost reporting and, indirectly, the numbers
a user checks against --max-cost — so an error here is a wrong figure rather
than a crash, which is the kind that goes unnoticed.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.llm_client import MODEL  # noqa: E402
from modules.token_tracker import PRICING_TABLE, TokenTracker, TokenUsage  # noqa: E402

PRICED = MODEL  # the default model, which is in the table


@pytest.fixture
def tracker():
    return TokenTracker()


# ── accumulation ──────────────────────────────────────────────────────────


def test_a_fresh_tracker_is_empty(tracker):
    assert tracker.calls == 0
    assert tracker.total_usage.total_tokens == 0


def test_one_call_is_recorded(tracker):
    tracker.add_usage(PRICED, 100, 50)

    assert tracker.calls == 1
    assert tracker.total_usage.input_tokens == 100
    assert tracker.total_usage.output_tokens == 50


def test_calls_accumulate(tracker):
    """The property that matters: a summary reflects the whole run, not the last call."""
    tracker.add_usage(PRICED, 100, 50)
    tracker.add_usage(PRICED, 200, 75)

    assert tracker.calls == 2
    assert tracker.total_usage.input_tokens == 300
    assert tracker.total_usage.output_tokens == 125


def test_the_total_is_the_sum_of_both(tracker):
    tracker.add_usage(PRICED, 100, 50)

    assert tracker.total_usage.total_tokens == 150


def test_a_zero_token_call_still_counts(tracker):
    """A call that returned nothing was still a call, and still cost latency."""
    tracker.add_usage(PRICED, 0, 0)

    assert tracker.calls == 1


# ── cost ──────────────────────────────────────────────────────────────────


def test_a_priced_model_produces_a_cost(tracker):
    tracker.add_usage(PRICED, 1000, 1000)

    assert tracker.cost_usd > 0


def test_cost_accumulates_across_calls(tracker):
    tracker.add_usage(PRICED, 1000, 1000)
    first = tracker.cost_usd
    tracker.add_usage(PRICED, 1000, 1000)

    assert tracker.cost_usd == pytest.approx(first * 2)


def test_more_tokens_cost_more(tracker):
    small = TokenTracker()
    small.add_usage(PRICED, 100, 100)
    tracker.add_usage(PRICED, 10_000, 10_000)

    assert tracker.cost_usd > small.cost_usd


def test_an_unknown_model_does_not_raise(tracker):
    """
    A self-hosted or newly released model is not in the table. Reporting no
    cost is right; raising would fail a run over a reporting detail.
    """
    tracker.add_usage("some-local-model", 100, 50)

    assert tracker.calls == 1
    assert tracker.cost_usd == 0


def test_the_default_model_is_priced():
    """
    The one model that must be in the table. If the default falls out of it,
    every run reports Unknown and the cost figure silently becomes useless.
    """
    assert MODEL in PRICING_TABLE


# ── the summary ───────────────────────────────────────────────────────────


def test_the_summary_reports_the_totals(tracker):
    tracker.add_usage(PRICED, 300, 125)
    summary = tracker.get_summary()

    assert "300" in summary
    assert "125" in summary
    assert "425" in summary


def test_the_summary_reports_the_call_count(tracker):
    tracker.add_usage(PRICED, 10, 10)
    tracker.add_usage(PRICED, 10, 10)

    assert "2" in tracker.get_summary()


def test_an_unknown_model_says_so(tracker):
    """Rather than printing $0.0000, which reads as free."""
    tracker.add_usage("some-local-model", 100, 50)

    assert "Unknown" in tracker.get_summary()


def test_an_empty_tracker_summarises_without_error(tracker):
    assert isinstance(tracker.get_summary(), str)


# ── the pricing table ─────────────────────────────────────────────────────


def test_every_entry_has_both_rates():
    """(input, output) per 1K tokens. A one-element tuple would raise at use."""
    for model, pricing in PRICING_TABLE.items():
        assert len(pricing) == 2, f"{model} has {len(pricing)} rate(s)"
        assert all(isinstance(rate, (int, float)) and rate >= 0 for rate in pricing)


def test_usage_defaults_to_zero():
    usage = TokenUsage()

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (0, 0, 0)
