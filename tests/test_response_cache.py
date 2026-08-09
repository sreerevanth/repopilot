"""
Tests for --cache (modules/response_cache.py).

Stores responses so an identical request need not be paid for twice — a dry run
followed by a real run against an unchanged repository being the case that
prompted it.

Opt-in on purpose. Requests go out at `temperature: 0.2`, so the provider may
return different text for the same prompt; serving a cached answer is a
behaviour change, not only an optimisation.
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.llm_client import BaseLLMClient  # noqa: E402
from modules.response_cache import (  # noqa: E402
    CACHE_DIRNAME,
    MAX_ENTRY_BYTES,
    ResponseCache,
    cache_key,
)

RESPONSE = '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'
KEY_PARTS = ("claude-sonnet-4", "SYSTEM PROMPT", "the prompt")


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(str(tmp_path), enabled=True)


class Counting(BaseLLMClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def _call(self, prompt):
        self.calls += 1
        return RESPONSE


# ── the key ───────────────────────────────────────────────────────────────


def test_the_same_request_gives_the_same_key():
    assert cache_key(*KEY_PARTS) == cache_key(*KEY_PARTS)


def test_a_different_model_gives_a_different_key():
    assert cache_key("gpt-4o", *KEY_PARTS[1:]) != cache_key(*KEY_PARTS)


def test_a_different_system_prompt_gives_a_different_key():
    """
    A run with --system-prompt must not be served an entry written by the
    default persona; the persona is what shapes the answer.
    """
    assert cache_key(KEY_PARTS[0], "OTHER", KEY_PARTS[2]) != cache_key(*KEY_PARTS)


def test_a_different_prompt_gives_a_different_key():
    """The prompt carries the repository context, so a changed repo misses."""
    assert cache_key(*KEY_PARTS[:2], "different") != cache_key(*KEY_PARTS)


def test_the_parts_cannot_run_together():
    """Without a separator ("ab", "c") and ("a", "bc") would collide."""
    assert cache_key("a", "b", "c") != cache_key("ab", "", "c")


# ── storage ───────────────────────────────────────────────────────────────


def test_a_miss_returns_nothing(cache):
    assert cache.get(cache_key(*KEY_PARTS)) is None


def test_a_stored_response_is_returned(cache):
    key = cache_key(*KEY_PARTS)
    cache.put(key, RESPONSE)

    assert cache.get(key) == RESPONSE


def test_entries_survive_a_new_instance(tmp_path):
    """The point is reuse across runs, not within one."""
    key = cache_key(*KEY_PARTS)
    ResponseCache(str(tmp_path), enabled=True).put(key, RESPONSE)

    assert ResponseCache(str(tmp_path), enabled=True).get(key) == RESPONSE


def test_entries_live_under_one_directory(tmp_path, cache):
    cache.put(cache_key(*KEY_PARTS), RESPONSE)

    assert (tmp_path / CACHE_DIRNAME).is_dir()


def test_hits_and_misses_are_counted(cache):
    key = cache_key(*KEY_PARTS)
    cache.get(key)
    cache.put(key, RESPONSE)
    cache.get(key)

    assert (cache.stats.hits, cache.stats.misses) == (1, 1)


# ── it stays out of the way when off ──────────────────────────────────────


def test_a_disabled_cache_stores_nothing(tmp_path):
    disabled = ResponseCache(str(tmp_path), enabled=False)
    key = cache_key(*KEY_PARTS)
    disabled.put(key, RESPONSE)

    assert ResponseCache(str(tmp_path), enabled=True).get(key) is None


def test_a_disabled_cache_never_hits(tmp_path):
    key = cache_key(*KEY_PARTS)
    ResponseCache(str(tmp_path), enabled=True).put(key, RESPONSE)

    assert ResponseCache(str(tmp_path), enabled=False).get(key) is None


# ── failing softly ────────────────────────────────────────────────────────


def test_a_corrupt_entry_is_a_miss(tmp_path, cache):
    key = cache_key(*KEY_PARTS)
    cache.put(key, RESPONSE)
    path = next((tmp_path / CACHE_DIRNAME).rglob("*.json"))
    path.write_text("not json at all")

    assert cache.get(key) is None


def test_an_unwritable_cache_does_not_raise(tmp_path, monkeypatch):
    """A cache that cannot be written should not fail the run."""
    def explode(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("os.makedirs", explode)

    ResponseCache(str(tmp_path), enabled=True).put(cache_key(*KEY_PARTS), RESPONSE)


def test_an_oversized_response_is_not_stored(cache):
    key = cache_key(*KEY_PARTS)
    cache.put(key, "x" * (MAX_ENTRY_BYTES + 1))

    assert cache.get(key) is None


def test_an_expired_entry_is_a_miss(tmp_path):
    key = cache_key(*KEY_PARTS)
    cache = ResponseCache(str(tmp_path), enabled=True, ttl_seconds=1)
    cache.put(key, RESPONSE)
    path = next((tmp_path / CACHE_DIRNAME).rglob("*.json"))
    entry = json.loads(path.read_text())
    entry["written_at"] = time.time() - 10
    path.write_text(json.dumps(entry))

    assert cache.get(key) is None


# ── through the client ────────────────────────────────────────────────────


def test_an_identical_request_makes_no_second_call(tmp_path):
    cache = ResponseCache(str(tmp_path), enabled=True)
    first = Counting(cache=cache)
    first.initial_request("fix parser", "context")
    second = Counting(cache=cache)
    second.initial_request("fix parser", "context")

    assert (first.calls, second.calls) == (1, 0)


def test_a_cached_run_costs_nothing(tmp_path):
    cache = ResponseCache(str(tmp_path), enabled=True)
    Counting(cache=cache).initial_request("fix parser", "context")
    second = Counting(cache=cache)
    second.initial_request("fix parser", "context")

    assert second.total_cost == 0.0


def test_a_cached_response_still_parses(tmp_path):
    cache = ResponseCache(str(tmp_path), enabled=True)
    Counting(cache=cache).initial_request("fix parser", "context")

    result = Counting(cache=cache).initial_request("fix parser", "context")

    assert result.parse_error is None
    assert result.done is True


def test_no_cache_means_every_request_is_made(tmp_path):
    client = Counting(cache=None)
    client.initial_request("fix parser", "context")
    client.initial_request("fix parser", "context")

    assert client.calls == 2


def test_a_hit_is_served_even_past_the_budget(tmp_path):
    """
    A cache hit costs nothing, so a run at its limit can still be served rather
    than stopping on a request it is not going to pay for.
    """
    cache = ResponseCache(str(tmp_path), enabled=True)
    Counting(cache=cache).initial_request("fix parser", "context")

    broke = Counting(cache=cache, max_cost=0.0000001)
    broke.total_cost = 999.0

    assert broke.initial_request("fix parser", "context").parse_error is None
