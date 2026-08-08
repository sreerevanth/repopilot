"""
Tests for --verbose (modules/llm_client.py).

Prints the exact prompt sent to the model and the raw text returned, for
debugging a response that parsed wrongly or a context that selected the wrong
files.

The prompt is repository file contents, so the property that matters most is
that a key committed somewhere in the repo is not echoed to the terminal.
"""

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.llm_client import (  # noqa: E402
    BaseLLMClient,
    _dump_payload,
    _redact_secrets,
)

OK = '{"analysis":"ok","changes":[],"confidence":0.9,"done":true}'


class Stub(BaseLLMClient):
    def __init__(self, verbose=False, raw=OK):
        super().__init__(verbose=verbose)
        self.raw = raw
        self.prompts = []

    def _call(self, prompt):
        self.prompts.append(prompt)
        return self.raw


def capture(client, method="initial", **kwargs):
    err = io.StringIO()
    with redirect_stderr(err):
        if method == "initial":
            client.initial_request(kwargs.get("task", "t"), kwargs.get("context", "c"))
        else:
            client.retry_request(
                kwargs.get("task", "t"), kwargs.get("context", "c"),
                [], "out", "err", 1,
            )
    return err.getvalue()


# ── off by default ────────────────────────────────────────────────────────


def test_verbose_is_off_by_default():
    assert Stub().verbose is False
    assert AgentConfig(repo_root=".", task="t").verbose_payloads is False


def test_nothing_is_printed_when_off():
    assert capture(Stub(verbose=False)) == ""


def test_the_response_still_parses_when_off():
    assert Stub(verbose=False).initial_request("t", "c").confidence == 0.9


# ── what gets dumped ──────────────────────────────────────────────────────


@pytest.fixture
def dumped():
    return capture(Stub(verbose=True))


def test_the_system_prompt_is_dumped(dumped):
    assert "system prompt" in dumped


def test_the_request_is_dumped(dumped):
    assert "request" in dumped


def test_the_response_is_dumped(dumped):
    assert "response" in dumped
    assert '"confidence":0.9' in dumped


def test_the_context_appears_in_the_request():
    """The point of the flag: seeing which files were actually sent."""
    assert "utils.py contents here" in capture(
        Stub(verbose=True), context="utils.py contents here"
    )


def test_retries_are_dumped_too():
    """A retry prompt carries the failure output and is worth seeing."""
    assert "[verbose]" in capture(Stub(verbose=True), method="retry")


def test_character_counts_are_reported(dumped):
    assert "chars" in dumped


# ── it goes to stderr ─────────────────────────────────────────────────────


def test_output_goes_to_stderr_not_stdout():
    """
    stdout carries --context-only and the dry-run manifest, which are meant to
    be piped. Debug output on stdout would corrupt them.
    """
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        Stub(verbose=True).initial_request("t", "c")

    assert out.getvalue() == ""
    assert err.getvalue() != ""


# ── secrets are masked ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_1234567890123456789012345678901234ab",
        "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ],
)
def test_known_secrets_are_masked_in_the_dump(secret):
    """
    The prompt is repository content. A key committed anywhere in the repo
    would otherwise be echoed to the terminal and into whatever captures it.
    """
    dumped = capture(Stub(verbose=True), context=f"config = '{secret}'")

    assert secret not in dumped
    assert "[REDACTED]" in dumped


def test_redaction_keeps_a_prefix_so_the_finding_is_identifiable():
    assert _redact_secrets("AKIAIOSFODNN7EXAMPLE").startswith("AKI")


def test_ordinary_text_is_untouched():
    text = "def add(a, b):\n    return a + b\n"
    assert _redact_secrets(text) == text


def test_redaction_can_be_turned_off_for_callers_that_need_raw():
    err = io.StringIO()
    with redirect_stderr(err):
        _dump_payload("raw", "AKIAIOSFODNN7EXAMPLE", redact=False)

    assert "AKIAIOSFODNN7EXAMPLE" in err.getvalue()


def test_redaction_reuses_the_scanner_patterns():
    """Two copies of the pattern list would drift apart."""
    source = (Path(__file__).resolve().parents[1] / "modules" / "llm_client.py")
    assert "SECRET_PATTERNS" in source.read_text(encoding="utf-8")


# ── wiring ────────────────────────────────────────────────────────────────


def test_the_flag_reaches_the_client():
    """
    A config field that never reaches the client would make --verbose a no-op
    that looks wired up.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "verbose=cfg.verbose_payloads" in source


def test_the_facade_forwards_to_the_chosen_provider():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "llm_client.py"
    ).read_text(encoding="utf-8")

    assert "self.underlying_client.verbose = verbose" in source
