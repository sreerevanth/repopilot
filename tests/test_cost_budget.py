"""
Tests for the cost budget (modules/llm_client.py).

An agent that retries can loop, and every iteration is a paid API call. These
pin the accounting and the stop condition. Nothing here contacts the API — the
Anthropic client is replaced with a stub that reports usage.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.llm_client import (  # noqa: E402
    DEFAULT_PRICING,
    PRICING_PER_MTOK,
    BudgetExceededError,
    LLMClient,
    UsageTracker,
)

SONNET = "claude-sonnet-4-20250514"


def _agent_loop_source() -> str:
    """
    Read agent_loop.py as UTF-8.

    Path.read_text() defaults to the locale encoding, which is cp1252 on a
    default Windows install -- the module's box-drawing section comments then
    decode to mojibake and any search against them silently fails.
    """
    path = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    return path.read_text(encoding="utf-8")
VALID_JSON = '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'


def stub_response(in_tokens=100_000, out_tokens=20_000, usage=True):
    return SimpleNamespace(
        content=[SimpleNamespace(text=VALID_JSON)],
        usage=(
            SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens)
            if usage else None
        ),
    )


class StubMessages:
    def __init__(self, response_factory=stub_response):
        self.calls = 0
        self._factory = response_factory

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return self._factory()


def make_client(max_cost_usd=None, model=SONNET, response_factory=stub_response):
    """Build an LLMClient without needing an API key or the anthropic package."""
    client = object.__new__(LLMClient)
    client.client = SimpleNamespace(messages=StubMessages(response_factory))
    client.model = model
    client.max_cost_usd = max_cost_usd
    client.usage = UsageTracker(model=model)
    return client


# ── accounting ────────────────────────────────────────────────────────────


def test_cost_matches_published_rates():
    usage = UsageTracker(model=SONNET)
    usage.input_tokens = 1_000_000
    usage.output_tokens = 1_000_000
    # $3/Mtok in + $15/Mtok out
    assert usage.cost_usd == pytest.approx(18.00)


def test_input_and_output_are_priced_differently():
    only_in = UsageTracker(model=SONNET)
    only_in.input_tokens = 1_000_000
    only_out = UsageTracker(model=SONNET)
    only_out.output_tokens = 1_000_000

    assert only_out.cost_usd > only_in.cost_usd


@pytest.mark.parametrize("model", sorted(PRICING_PER_MTOK))
def test_every_priced_model_has_both_rates(model):
    assert PRICING_PER_MTOK[model]["input"] > 0
    assert PRICING_PER_MTOK[model]["output"] > 0


def test_unknown_model_falls_back_rather_than_costing_nothing():
    """A budget reporting $0.00 for an unrecognised model is worse than none."""
    usage = UsageTracker(model="claude-something-not-released-yet")
    usage.input_tokens = 1_000_000

    assert usage.pricing == DEFAULT_PRICING
    assert usage.cost_usd > 0


def test_zero_usage_costs_nothing():
    assert UsageTracker(model=SONNET).cost_usd == 0.0


def test_record_accumulates_across_calls():
    usage = UsageTracker(model=SONNET)
    usage.record(stub_response(1_000, 200))
    usage.record(stub_response(3_000, 400))

    assert usage.calls == 2
    assert usage.input_tokens == 4_000
    assert usage.output_tokens == 600


def test_record_tolerates_a_response_with_no_usage():
    """A stub or an older SDK should not crash the run."""
    usage = UsageTracker(model=SONNET)
    usage.record(stub_response(usage=False))

    assert usage.calls == 1
    assert usage.cost_usd == 0.0


def test_summary_reports_calls_tokens_and_cost():
    usage = UsageTracker(model=SONNET)
    usage.record(stub_response(1_000, 200))
    text = usage.summary()

    assert "1 call" in text
    assert "1,000" in text
    assert "$" in text


# ── the stop condition ────────────────────────────────────────────────────


def test_no_limit_means_no_stopping():
    client = make_client(max_cost_usd=None)
    for _ in range(10):
        client._call("prompt")

    assert client.client.messages.calls == 10


def test_calls_stop_once_the_limit_is_reached():
    client = make_client(max_cost_usd=1.00)  # each call costs $0.60

    made = 0
    with pytest.raises(BudgetExceededError):
        for _ in range(20):
            client._call("prompt")
            made += 1

    assert made == 2
    assert client.client.messages.calls == 2


def test_usage_is_recorded_on_every_call():
    client = make_client()
    client._call("prompt")
    client._call("prompt")

    assert client.usage.calls == 2
    assert client.usage.input_tokens == 200_000


def test_overshoot_is_at_most_one_call():
    """
    Cost is only known after a call returns, so the limit is a stop condition
    rather than a pre-authorisation. Documented, and pinned here.
    """
    client = make_client(max_cost_usd=1.00)
    with pytest.raises(BudgetExceededError):
        for _ in range(20):
            client._call("prompt")

    per_call = 0.60
    assert client.usage.cost_usd <= 1.00 + per_call


def test_the_error_says_what_was_spent():
    client = make_client(max_cost_usd=0.50)
    with pytest.raises(BudgetExceededError) as excinfo:
        for _ in range(5):
            client._call("prompt")

    message = str(excinfo.value)
    assert "0.50" in message
    assert "call(s)" in message


def test_a_zero_limit_blocks_the_first_call():
    client = make_client(max_cost_usd=0.0)
    with pytest.raises(BudgetExceededError):
        client._call("prompt")

    assert client.client.messages.calls == 0


def test_budget_error_is_catchable_as_runtime_error():
    """agent_loop distinguishes it from a generic failure; both are catchable."""
    assert issubclass(BudgetExceededError, RuntimeError)


# ── plumbing ──────────────────────────────────────────────────────────────


def test_the_configured_model_is_the_one_called():
    client = make_client(model="claude-3-5-haiku-20241022")
    client._call("prompt")

    assert client.client.messages.last_kwargs["model"] == "claude-3-5-haiku-20241022"


def test_pricing_follows_the_configured_model():
    client = make_client(model="claude-opus-4-20250514")
    client._call("prompt")
    sonnet = make_client(model=SONNET)
    sonnet._call("prompt")

    assert client.usage.cost_usd > sonnet.usage.cost_usd


def test_loop_treats_the_budget_as_a_stop_not_an_error():
    """
    agent_loop must catch BudgetExceededError before the generic handler, or a
    deliberate stop is reported as "the API broke".
    """
    source = _agent_loop_source()

    # Scoped to the LLM call block: there is an unrelated `outcome = "error"`
    # earlier in run() for context building. Anchored on code rather than on a
    # section comment, since comments are decorative and get reworded.
    start = source.index("self.llm.initial_request")
    block = source[start:start + 1800]

    assert "except BudgetExceededError" in block
    assert block.index("except BudgetExceededError") < block.index("except Exception")


def test_budget_stop_rolls_back_applied_changes():
    """The reported message promises this, so it needs to be true."""
    source = _agent_loop_source()

    rollback_line = source.index("Rolling back file changes due to failed run")
    condition = source[source.rindex("if (", 0, rollback_line):rollback_line]

    assert "budget_exceeded" in condition
