"""
Tests for streamed LLM responses (modules/llm_client.py).

A blocking call sits silent for 20-30 seconds. Streaming shows progress while
the response arrives. The response is still parsed as one JSON object at the
end — see the note in the PR about why the deltas are not parsed incrementally.

No test contacts the API.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.llm_client import LLMClient  # noqa: E402

RESPONSE = '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'


class FakeStream:
    def __init__(self, chunks):
        self.text_stream = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_client(stream=True, streaming_available=True, chunks=None, raises=None):
    """Build a client without an API key or the anthropic package."""
    client = object.__new__(LLMClient)
    messages = SimpleNamespace()

    calls = {"stream": 0, "create": 0}

    if streaming_available:
        def _stream(**kwargs):
            calls["stream"] += 1
            if raises is not None:
                raise raises
            return FakeStream(chunks if chunks is not None else [RESPONSE])
        messages.stream = _stream

    def _create(**kwargs):
        calls["create"] += 1
        return SimpleNamespace(content=[SimpleNamespace(text=RESPONSE)])
    messages.create = _create

    client.client = SimpleNamespace(messages=messages)
    client.stream = stream
    client.calls = calls
    return client


# ── the streamed text is reassembled correctly ────────────────────────────


def test_streaming_returns_the_whole_response():
    assert make_client()._call("prompt") == RESPONSE


@pytest.mark.parametrize("size", [1, 3, 7, 13, 500])
def test_chunk_boundaries_do_not_corrupt_the_json(size):
    """
    Deltas split wherever the API decides, including mid-token and mid-escape.
    Reassembling before parsing is what makes that safe.
    """
    import json

    chunks = [RESPONSE[i:i + size] for i in range(0, len(RESPONSE), size)]
    parsed = json.loads(make_client(chunks=chunks)._call("prompt"))

    assert parsed["confidence"] == 0.9


def test_an_empty_stream_returns_an_empty_string():
    """Handled by the existing parse-error path rather than raising here."""
    assert make_client(chunks=[])._call("prompt") == ""


def test_unicode_survives_reassembly():
    payload = '{"analysis":"café — naïve","changes":[],"confidence":0.5,"done":false}'
    chunks = [payload[i:i + 3] for i in range(0, len(payload), 3)]
    client = make_client(chunks=chunks)

    assert client._call("prompt") == payload


# ── choosing a path ───────────────────────────────────────────────────────


def test_streaming_is_used_by_default():
    client = make_client()
    client._call("prompt")

    assert client.calls["stream"] == 1
    assert client.calls["create"] == 0


def test_streaming_can_be_disabled():
    client = make_client(stream=False)
    client._call("prompt")

    assert client.calls["create"] == 1
    assert client.calls["stream"] == 0


def test_an_sdk_without_streaming_falls_back():
    """A progress indicator is not worth failing a run over."""
    client = make_client(streaming_available=False)

    assert client._call("prompt") == RESPONSE
    assert client.calls["create"] == 1


def test_the_fallback_is_remembered():
    """Otherwise every call would re-attempt streaming and fail again."""
    client = make_client(streaming_available=False)
    client._call("prompt")

    assert client.stream is False


def test_both_paths_return_the_same_text():
    assert make_client()._call("p") == make_client(stream=False)._call("p")


# ── errors still retry ────────────────────────────────────────────────────


def test_a_streaming_error_is_retried():
    client = make_client(raises=RuntimeError("connection reset"))

    with pytest.raises(RuntimeError):
        client._call("prompt", retries=3)

    assert client.calls["stream"] == 3


# ── progress output ───────────────────────────────────────────────────────


def test_progress_is_silent_when_stderr_is_redirected(monkeypatch, capsys):
    """
    Progress uses a carriage return to overwrite one line. In a log file or a CI
    job that produces a wall of partial lines, so it is suppressed off a tty.
    """
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    make_client(chunks=["a", "b", "c"])._call("prompt")

    assert capsys.readouterr().err == ""
