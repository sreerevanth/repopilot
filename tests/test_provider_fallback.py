"""
Tests for --fallback-provider (modules/llm_client.py).

When the primary provider returns a 500 or reports being overloaded, the
request itself was fine and the service was not — so another provider may well
answer it.

The distinction that makes this safe is the same one #154 used for push
retries: an authentication failure or a malformed request fails identically on
the fallback, so trying anyway spends money to produce the same error twice.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.llm_client import (  # noqa: E402
    BaseLLMClient,
    BudgetExceededError,
    LLMClient,
    is_transient_api_error,
)

RESPONSE = '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'


class Failing(BaseLLMClient):
    def __init__(self, error):
        super().__init__()
        self.error = error
        self.calls = 0

    def _call(self, prompt):
        self.calls += 1
        raise self.error


class Working(BaseLLMClient):
    def __init__(self, cost=0.004):
        super().__init__()
        self.calls = 0
        self.cost = cost

    def _call(self, prompt):
        self.calls += 1
        self.total_cost = self.cost
        self.input_tokens_used = 100
        self.output_tokens_used = 50
        return RESPONSE


def facade(primary, fallback=None):
    client = object.__new__(LLMClient)
    BaseLLMClient.__init__(client)
    client.provider = "anthropic"
    client.underlying_client = primary
    client.fallback_provider = "openai" if fallback else None
    client._fallback_client = fallback
    client._fallback_args = (None, "model", None)
    client.used_fallback = False
    return client


# ── what counts as worth failing over ─────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    ["Error 529: Overloaded", "503 Service Unavailable", "500 Internal Server Error",
     "429 Too Many Requests", "Connection reset by peer", "Request timed out",
     "502 Bad Gateway"],
)
def test_service_failures_are_transient(message):
    assert is_transient_api_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    ["401 Authentication failed", "invalid request: unknown model",
     "permission denied", "context length exceeded"],
)
def test_request_failures_are_not_transient(message):
    """These fail identically on any provider."""
    assert is_transient_api_error(RuntimeError(message)) is False


def test_a_budget_stop_is_not_transient():
    """
    Failing over would spend the money the limit exists to prevent, on a
    provider the limit also applies to.
    """
    assert is_transient_api_error(BudgetExceededError("limit reached")) is False


# ── failover ──────────────────────────────────────────────────────────────


def test_a_transient_failure_uses_the_fallback():
    backup = Working()
    client = facade(Failing(RuntimeError("Error 529: Overloaded")), backup)

    assert client._call("prompt") == RESPONSE
    assert backup.calls == 1


def test_it_records_that_the_fallback_was_used():
    """So a surprising answer can be traced to a different model."""
    client = facade(Failing(RuntimeError("503 Service Unavailable")), Working())
    client._call("prompt")

    assert client.used_fallback is True


def test_a_non_transient_failure_is_raised_untouched():
    backup = Working()
    client = facade(Failing(RuntimeError("401 Authentication failed")), backup)

    with pytest.raises(RuntimeError, match="Authentication"):
        client._call("prompt")

    assert backup.calls == 0


def test_no_fallback_configured_means_the_error_propagates():
    client = facade(Failing(RuntimeError("Error 529: Overloaded")))

    with pytest.raises(RuntimeError):
        client._call("prompt")


def test_a_working_primary_never_touches_the_fallback():
    backup = Working()
    client = facade(Working(), backup)

    client._call("prompt")

    assert backup.calls == 0


# ── accounting survives the failover ──────────────────────────────────────


def test_the_fallback_spend_is_counted():
    """
    Spend on the fallback is real spend. Not folding it in would let --max-cost
    be exceeded silently whenever a failover happened.
    """
    client = facade(Failing(RuntimeError("Error 529: Overloaded")), Working(cost=0.004))
    client._call("prompt")

    assert client.total_cost == pytest.approx(0.004)


def test_the_fallback_tokens_are_counted():
    client = facade(Failing(RuntimeError("Error 529: Overloaded")), Working())
    client._call("prompt")

    assert client.input_tokens_used == 100
    assert client.output_tokens_used == 50


# ── wiring ────────────────────────────────────────────────────────────────


def test_no_fallback_by_default():
    assert AgentConfig(repo_root=".", task="t").fallback_provider is None


def test_the_fallback_client_is_built_lazily():
    """
    Building it eagerly would demand a second SDK and a second key from
    everyone, including the majority who never fail over.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "llm_client.py"
    ).read_text(encoding="utf-8")

    assert "self._fallback_client = None" in source
    assert "def _fallback(self)" in source


def test_main_passes_both_flags():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

    assert "fallback_provider=args.fallback_provider" in source
    assert "fallback_api_key=args.fallback_api_key" in source


def test_the_fallback_key_defaults_to_the_primary_key():
    """Some people use one key across providers; requiring two would be noise."""
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "llm_client.py"
    ).read_text(encoding="utf-8")

    assert "fallback_api_key or api_key" in source
