"""
Tests for reply block selection (modules/llm_client.py).

`response.content[0].text` assumed index 0 is a TextBlock. It is a union —
thinking, tool-use and tool-result blocks are all possible, and none of them
has a `text` attribute. A reply opening with one raised AttributeError and
killed the run.

#176's provider fallback would not have caught it either: an AttributeError is
not a transient API error, so it would have been re-raised rather than retried.

There was also a bare `print(response.content[0].text)` on the line above,
writing the full reply to stdout outside the logger.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.llm_client import _first_text_block  # noqa: E402


def text(value):
    return SimpleNamespace(type="text", text=value)


def thinking():
    """No `text` attribute — this is what used to raise."""
    return SimpleNamespace(type="thinking", thinking="reasoning")


def tool_use():
    return SimpleNamespace(type="tool_use", name="search", input={})


# ── selection ─────────────────────────────────────────────────────────────


def test_a_text_block_is_returned():
    assert _first_text_block([text("hello")]) == "hello"


def test_a_leading_thinking_block_is_skipped():
    """The case that used to raise AttributeError."""
    assert _first_text_block([thinking(), text("hello")]) == "hello"


def test_several_non_text_blocks_are_skipped():
    assert _first_text_block([tool_use(), thinking(), text("hi")]) == "hi"


def test_the_first_text_block_wins():
    assert _first_text_block([text("first"), text("second")]) == "first"


def test_an_empty_string_block_is_still_text():
    """`getattr(..., None)` rather than truthiness: "" is a real reply."""
    assert _first_text_block([text(""), text("later")]) == ""


# ── nothing usable ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content,label",
    [([thinking()], "only thinking"), ([], "empty"), (None, "none")],
)
def test_no_text_returns_empty_rather_than_raising(content, label):
    """
    A parse failure, not a crash. The caller already handles an unparseable
    response and retries; raising here would end the run instead.
    """
    assert _first_text_block(content) == ""


def test_no_text_is_logged(caplog):
    """Silence would make an empty reply look like an empty model answer."""
    with caplog.at_level("WARNING"):
        _first_text_block([thinking()])

    assert "no text block" in caplog.text


# ── the print is gone ─────────────────────────────────────────────────────


def test_the_reply_is_not_printed_to_stdout():
    """
    It bypassed the logger entirely: unaffected by --quiet and unmasked by the
    secret redaction that --verbose applies, so anyone piping output got the
    raw reply interleaved with their own.
    """
    source = (ROOT / "modules" / "llm_client.py").read_text(encoding="utf-8")

    assert "print(response.content[0].text)" not in source


def test_no_provider_prints_the_reply_before_returning():
    """
    The same three lines appeared in the OpenAI, Gemini and Ollama paths --
    print the reply, print a blank line, return it. All four are removed.

    The streaming prints are deliberate and stay: those are the streamed
    output itself, which is the point of streaming.
    """
    source = (ROOT / "modules" / "llm_client.py").read_text(encoding="utf-8")

    assert "print(text)\n                print" not in source
    assert source.count("print(text)") == 0


def test_index_zero_is_no_longer_assumed():
    source = (ROOT / "modules" / "llm_client.py").read_text(encoding="utf-8")

    assert "response.content[0].text" not in source
