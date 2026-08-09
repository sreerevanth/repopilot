"""
Tests for streamed LLM responses (modules/llm_client.py).

`AnthropicClient._call` streams the response via `messages.stream()` and falls
back to a blocking `messages.create()` if streaming raises. Neither path had
coverage.

An earlier version of this file tested `_call_streaming`/`_call_blocking` on a
single-client design that the multi-provider refactor replaced. Rewritten
against what actually shipped.

No test contacts the API.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.llm_client import SYSTEM_PROMPT, AnthropicClient  # noqa: E402

RESPONSE = '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'


class FakeStream:
    def __init__(self, chunks):
        self.text_stream = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def client(chunks=None, stream_raises=None, blocking=RESPONSE):
    """An AnthropicClient with a stubbed SDK, built without a key or the package."""
    obj = object.__new__(AnthropicClient)
    obj.model = "claude-sonnet-4-20250514"
    obj.verbose = False
    obj.system_prompt = SYSTEM_PROMPT
    obj.cache = None
    obj.max_cost = None
    obj.input_tokens_used = obj.output_tokens_used = 0
    obj.total_cost = 0.0

    calls = {"stream": 0, "create": 0}

    def _stream(**kwargs):
        calls["stream"] += 1
        if stream_raises is not None:
            raise stream_raises
        return FakeStream(chunks if chunks is not None else [RESPONSE])

    def _create(**kwargs):
        calls["create"] += 1
        return SimpleNamespace(content=[SimpleNamespace(text=blocking)])

    obj.client = SimpleNamespace(messages=SimpleNamespace(stream=_stream, create=_create))
    obj.calls = calls
    return obj


# ── the streaming path ────────────────────────────────────────────────────


def test_a_streamed_response_is_returned_whole():
    assert client()._call("prompt") == RESPONSE


def test_streaming_is_preferred_over_blocking():
    stub = client()
    stub._call("prompt")

    assert stub.calls["stream"] == 1
    assert stub.calls["create"] == 0


@pytest.mark.parametrize("size", [1, 3, 7, 13, 500])
def test_chunk_boundaries_do_not_corrupt_the_json(size):
    """
    Deltas split wherever the API decides, including mid-token and mid-escape.
    Reassembling before parsing is what makes that safe.
    """
    import json

    chunks = [RESPONSE[i:i + size] for i in range(0, len(RESPONSE), size)]

    assert json.loads(client(chunks=chunks)._call("prompt"))["confidence"] == 0.9


def test_unicode_survives_reassembly():
    payload = '{"analysis":"café — naïve","changes":[],"confidence":0.5,"done":false}'
    chunks = [payload[i:i + 3] for i in range(0, len(payload), 3)]

    assert client(chunks=chunks)._call("prompt") == payload


def test_an_empty_stream_returns_an_empty_string():
    """Left to the existing parse-error path rather than raising here."""
    assert client(chunks=[])._call("prompt") == ""


# ── the fallback ──────────────────────────────────────────────────────────


def test_a_streaming_failure_falls_back_to_a_blocking_call():
    """
    A provider that does not support streaming, or a mid-stream disconnect,
    should not fail the iteration when a blocking call would work.
    """
    stub = client(stream_raises=RuntimeError("streaming unavailable"))

    assert stub._call("prompt") == RESPONSE
    assert stub.calls["create"] == 1


def test_the_fallback_returns_the_blocking_content():
    stub = client(stream_raises=RuntimeError("boom"), blocking='{"done":true}')

    assert stub._call("prompt") == '{"done":true}'


def test_streaming_is_attempted_before_the_fallback():
    stub = client(stream_raises=RuntimeError("boom"))
    stub._call("prompt")

    assert stub.calls["stream"] == 1


# ── the model and prompt are passed through ───────────────────────────────


def test_the_configured_model_is_used():
    seen = {}
    stub = client()

    def _stream(**kwargs):
        seen.update(kwargs)
        return FakeStream([RESPONSE])

    stub.client.messages.stream = _stream
    stub._call("prompt")

    assert seen["model"] == "claude-sonnet-4-20250514"


def test_the_prompt_is_sent_as_the_user_message():
    seen = {}
    stub = client()

    def _stream(**kwargs):
        seen.update(kwargs)
        return FakeStream([RESPONSE])

    stub.client.messages.stream = _stream
    stub._call("a distinctive prompt")

    assert seen["messages"][0]["content"] == "a distinctive prompt"
    assert seen["messages"][0]["role"] == "user"


def test_the_system_prompt_is_sent():
    seen = {}
    stub = client()

    def _stream(**kwargs):
        seen.update(kwargs)
        return FakeStream([RESPONSE])

    stub.client.messages.stream = _stream
    stub._call("prompt")

    assert seen["system"]


# ── it parses ─────────────────────────────────────────────────────────────


def test_a_streamed_response_parses_into_an_LLMResponse():
    stub = client()
    result = stub.initial_request("task", "context")

    assert result.confidence == 0.9
    assert result.done is True
