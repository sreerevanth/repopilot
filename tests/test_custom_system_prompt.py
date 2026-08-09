"""
Tests for --system-prompt (modules/llm_client.py).

Replaces the default agent persona with the contents of a file.

The risk worth guarding is quiet: the system prompt carries the JSON output
contract, so a replacement that drops it produces responses the parser cannot
read — and the run fails on every iteration with a parse error that says
nothing about the real cause.
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.llm_client import (  # noqa: E402
    REQUIRED_SCHEMA_KEYS,
    SYSTEM_PROMPT,
    BaseLLMClient,
    SystemPromptError,
    load_system_prompt,
)

COMPLETE = (
    "You are a code agent.\n"
    "Return JSON with analysis, changes, confidence and done.\n"
)


class Stub(BaseLLMClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts = []

    def _call(self, prompt):
        self.prompts.append(prompt)
        return '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'


# ── the default is unchanged ──────────────────────────────────────────────


def test_no_override_uses_the_module_prompt():
    assert Stub().system_prompt == SYSTEM_PROMPT


def test_the_flag_is_off_by_default():
    assert AgentConfig(repo_root=".", task="t").system_prompt_file is None


def test_an_override_replaces_it_entirely():
    """The issue asks for replacement, not augmentation."""
    client = Stub(system_prompt="be terse")

    assert client.system_prompt == "be terse"
    assert SYSTEM_PROMPT not in client.system_prompt


# ── loading ───────────────────────────────────────────────────────────────


def test_a_file_is_read(tmp_path):
    path = tmp_path / "persona.txt"
    path.write_text(COMPLETE)

    assert load_system_prompt(str(path)) == COMPLETE


def test_utf8_is_read_correctly(tmp_path):
    """The default encoding is cp1252 on Windows and would mangle this."""
    path = tmp_path / "persona.txt"
    path.write_text(COMPLETE + "Répondez précisément — no waffle\n", encoding="utf-8")

    assert "précisément" in load_system_prompt(str(path))


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(SystemPromptError) as excinfo:
        load_system_prompt(str(tmp_path / "nope.txt"))

    assert "Could not read" in str(excinfo.value)


def test_an_empty_file_is_refused(tmp_path):
    """
    An empty system prompt is accepted by the API and produces confident
    nonsense, which is far harder to diagnose than a refusal.
    """
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n")

    with pytest.raises(SystemPromptError) as excinfo:
        load_system_prompt(str(path))

    assert "empty" in str(excinfo.value)


# ── the schema warning ────────────────────────────────────────────────────


def test_a_prompt_missing_the_schema_warns(tmp_path, caplog):
    path = tmp_path / "poet.txt"
    path.write_text("You are a helpful poet.\n")

    with caplog.at_level(logging.WARNING, logger="agent.llm_client"):
        load_system_prompt(str(path))

    assert any("does not mention" in r.message for r in caplog.records)


def test_the_warning_names_the_missing_keys(tmp_path, caplog):
    path = tmp_path / "partial.txt"
    path.write_text("Return JSON with analysis and changes.\n")

    with caplog.at_level(logging.WARNING, logger="agent.llm_client"):
        load_system_prompt(str(path))

    message = " ".join(r.message for r in caplog.records)
    assert "confidence" in message and "done" in message


def test_a_complete_prompt_does_not_warn(tmp_path, caplog):
    path = tmp_path / "complete.txt"
    path.write_text(COMPLETE)

    with caplog.at_level(logging.WARNING, logger="agent.llm_client"):
        load_system_prompt(str(path))

    assert not any("does not mention" in r.message for r in caplog.records)


def test_a_missing_schema_warns_rather_than_refuses(tmp_path):
    """
    The flag exists to let people experiment with the persona. A hard rejection
    would block a legitimate rewrite that phrases the contract differently.
    """
    path = tmp_path / "poet.txt"
    path.write_text("You are a helpful poet.\n")

    assert load_system_prompt(str(path))  # returns, does not raise


@pytest.mark.parametrize("key", REQUIRED_SCHEMA_KEYS)
def test_the_default_prompt_satisfies_its_own_check(key):
    """If the shipped prompt failed this check, the check would be wrong."""
    assert key in SYSTEM_PROMPT


# ── it reaches the request ────────────────────────────────────────────────


def test_the_override_is_used_for_token_estimation():
    """Cost is estimated from prompt + system prompt; the wrong one skews it."""
    short, long = Stub(system_prompt="x"), Stub(system_prompt="x" * 4000)

    assert long._estimate_tokens("p" + long.system_prompt) > \
        short._estimate_tokens("p" + short.system_prompt)


def test_the_override_is_dumped_by_verbose(capsys):
    Stub(system_prompt="CUSTOM PERSONA HERE", verbose=True).initial_request("t", "c")

    assert "CUSTOM PERSONA HERE" in capsys.readouterr().err


def test_the_facade_forwards_the_override():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "llm_client.py"
    ).read_text(encoding="utf-8")

    assert "self.underlying_client.system_prompt = self.system_prompt" in source


def test_the_loop_loads_the_file():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "load_system_prompt(cfg.system_prompt_file)" in source


# ── it reaches the accounted path ─────────────────────────────────────────


def test_the_override_is_what_gets_costed():
    """
    _accounted_call estimates input tokens from prompt + system prompt, and
    --max-cost is enforced from that number. Costing the default while sending
    an override would misreport spend.
    """
    short, long = Stub(system_prompt="x"), Stub(system_prompt="x" * 8_000)

    short.initial_request("task", "context")
    long.initial_request("task", "context")

    assert long.total_cost > short.total_cost


def test_the_prompt_is_read_once_per_client_not_per_call():
    """
    Reading the file on every request would hit disk each iteration and let the
    persona change mid-run.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "llm_client.py"
    ).read_text(encoding="utf-8")

    assert "self.system_prompt = system_prompt or SYSTEM_PROMPT" in source
    assert source.count("load_system_prompt(cfg.system_prompt_file)") <= 1
