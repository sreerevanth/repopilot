"""
Tests for missing-credential reporting (modules/llm_client.py).

Every provider raised a bare exception, so the two most common first-run
mistakes produced a Python traceback into code the user did not write:

    ValueError: ANTHROPIC_API_KEY not set
    RuntimeError: anthropic package not installed. Run: pip install anthropic

`ConfigurationError` was already imported at the top of that file and is an
`AgentError`, so it reaches the handler added in #260 and prints a message with
a remedy. These sites were simply not converted when that landed.

Exercised against the client classes directly rather than through the CLI. An
earlier version of this file ran `main.py --repo .` without `--no-git`, which
made the agent create and check out a branch in the repository under test —
the tests mutated the working tree they were running in.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import llm_client  # noqa: E402
from modules.errors import AgentError, ConfigurationError  # noqa: E402

KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]


@pytest.fixture
def no_keys(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


# ── a missing key ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cls_name,variable",
    [
        ("AnthropicClient", "ANTHROPIC_API_KEY"),
        ("OpenAIClient", "OPENAI_API_KEY"),
        ("GeminiClient", "GEMINI_API_KEY"),
    ],
)
def test_a_missing_key_raises_configuration_error(no_keys, monkeypatch, cls_name, variable):
    """
    A bare ValueError escaped the top-level handler and became a traceback.
    ConfigurationError is an AgentError, so it is reported with a remedy.
    """
    cls = getattr(llm_client, cls_name, None)
    if cls is None:
        pytest.skip(f"{cls_name} not present in this build")

    # The anthropic client checks for its SDK before it checks for the key, so
    # the availability flag is forced open here to reach the key check. On a
    # machine without the SDK installed, the SDK error is what the user sees
    # first -- which is why it is now a ConfigurationError too.
    monkeypatch.setattr(llm_client, "_ANTHROPIC_AVAILABLE", True, raising=False)

    with pytest.raises(ConfigurationError) as caught:
        cls()

    assert variable in str(caught.value)


@pytest.mark.parametrize("cls_name", ["AnthropicClient", "OpenAIClient", "GeminiClient"])
def test_the_message_offers_a_remedy(no_keys, monkeypatch, cls_name):
    """
    The remedy matters more than the class change — someone hitting this has
    usually just cloned the repository.
    """
    cls = getattr(llm_client, cls_name, None)
    if cls is None:
        pytest.skip(f"{cls_name} not present in this build")
    monkeypatch.setattr(llm_client, "_ANTHROPIC_AVAILABLE", True, raising=False)

    with pytest.raises(ConfigurationError) as caught:
        cls()

    assert "Export it" in str(caught.value)


def test_the_message_points_at_the_documentation(no_keys, monkeypatch):
    monkeypatch.setattr(llm_client, "_ANTHROPIC_AVAILABLE", True, raising=False)

    with pytest.raises(ConfigurationError) as caught:
        llm_client.AnthropicClient()

    assert "README" in str(caught.value)


# ── a missing SDK ─────────────────────────────────────────────────────────


def test_a_missing_sdk_raises_configuration_error(monkeypatch):
    """
    This fires before the key check, so on a machine without the SDK it was the
    error the user actually saw — and it was a bare RuntimeError.
    """
    monkeypatch.setattr(llm_client, "_ANTHROPIC_AVAILABLE", False, raising=False)

    with pytest.raises(ConfigurationError) as caught:
        llm_client.AnthropicClient(api_key="x")

    assert "not installed" in str(caught.value)


def test_the_missing_sdk_message_names_the_install_command(monkeypatch):
    monkeypatch.setattr(llm_client, "_ANTHROPIC_AVAILABLE", False, raising=False)

    with pytest.raises(ConfigurationError) as caught:
        llm_client.AnthropicClient(api_key="x")

    assert "requirements.txt" in str(caught.value)


def test_the_missing_sdk_message_names_the_alternative(monkeypatch):
    """The other three providers use urllib and need no SDK at all."""
    monkeypatch.setattr(llm_client, "_ANTHROPIC_AVAILABLE", False, raising=False)

    with pytest.raises(ConfigurationError) as caught:
        llm_client.AnthropicClient(api_key="x")

    assert "--provider" in str(caught.value)


# ── the class and the source ──────────────────────────────────────────────


def test_configuration_error_is_an_agent_error():
    """This is what routes it to the clean handler rather than a traceback."""
    assert issubclass(ConfigurationError, AgentError)


def test_no_bare_credential_error_remains():
    """
    The constructs being replaced. A future provider copying either pattern
    would reintroduce the traceback.
    """
    source = (ROOT / "modules" / "llm_client.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    for key in KEYS:
        assert f'ValueError("{key} not set")' not in code
    assert 'RuntimeError("anthropic package not installed' not in code


def test_every_credential_check_uses_the_same_class():
    source = (ROOT / "modules" / "llm_client.py").read_text(encoding="utf-8")

    assert source.count("raise ConfigurationError(") >= 4
