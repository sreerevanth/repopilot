"""
Tests for --quiet and --api-base-url (modules/agent_loop.py, modules/llm_client.py).

Both flags were registered in argparse and appeared in `--help`, and neither
value was ever read. `--quiet` had no effect because AgentLogger was constructed
with `verbose=True` unconditionally; `--api-base-url` had none because the
Ollama endpoint was hardcoded to localhost.

A flag that silently does nothing is worse than a missing flag — nobody reports
it, because the tool appears to have accepted the setting.
"""

import dataclasses
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.llm_client import DEFAULT_BASE_URLS, OllamaClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def source(name):
    return (ROOT / name).read_text(encoding="utf-8")


# ── every flag is read ────────────────────────────────────────────────────


def test_no_flag_is_registered_but_unread():
    """
    The check that would have caught both of these. --max-cost had the same
    problem and is fixed separately.
    """
    help_text = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--help"],
        capture_output=True, text=True, cwd=str(ROOT),
    ).stdout

    code = source("main.py") + "".join(
        source(f"modules/{m}")
        for m in ("agent_loop.py", "llm_client.py", "sandbox.py",
                  "context_builder.py", "code_modifier.py")
    )

    dead = []
    for flag in sorted(set(re.findall(r"--([a-z][a-z0-9-]+)", help_text))):
        if flag in ("help", "cov", "cov-report", "no-install", "select"):
            continue
        match = re.search(
            rf'add_argument\(\s*"--{re.escape(flag)}"[^)]*?dest="(\w+)"', code, re.S
        )
        dest = match.group(1) if match else flag.replace("-", "_")
        if not re.search(rf"args\.{dest}\b|cfg\.{dest}\b", code):
            dead.append(flag)

    # --max-cost has the same problem and is fixed in a separate PR; this
    # allowance should be removed once that merges, at which point the check
    # becomes unconditional.
    dead = [flag for flag in dead if flag != "max-cost"]

    assert dead == [], f"flags registered but never read: {dead}"


# ── --quiet ───────────────────────────────────────────────────────────────


def test_quiet_is_a_config_field():
    assert "quiet" in {f.name for f in dataclasses.fields(AgentConfig)}


def test_quiet_is_off_by_default():
    assert AgentConfig(repo_root=".", task="t").quiet is False


def test_the_logger_is_no_longer_pinned_to_verbose():
    """It was `verbose=True` regardless of the flag."""
    text = source("modules/agent_loop.py")

    assert "AgentLogger(self.log_dir, self.run_id, verbose=True)" not in text
    assert "verbose=not cfg.quiet" in text


def test_main_passes_the_flag():
    assert "quiet=args.quiet" in source("main.py")


@pytest.mark.parametrize("quiet,expected", [(False, True), (True, False)])
def test_quiet_controls_logger_verbosity(tmp_path, quiet, expected):
    from modules.logger import AgentLogger

    logger = AgentLogger(str(tmp_path), "run_x", verbose=not quiet)

    assert logger.verbose is expected


# ── --api-base-url ────────────────────────────────────────────────────────


def test_api_base_url_is_a_config_field():
    assert "api_base_url" in {f.name for f in dataclasses.fields(AgentConfig)}


def test_it_is_unset_by_default():
    assert AgentConfig(repo_root=".", task="t").api_base_url is None


def test_ollama_defaults_to_localhost():
    """Unchanged behaviour for anyone not passing the flag."""
    assert OllamaClient().api_base_url == DEFAULT_BASE_URLS["ollama"]


def test_an_override_is_used():
    """
    Ollama is self-hosted, so the address is a per-user setting — the whole
    reason this flag exists.
    """
    assert OllamaClient(api_base_url="http://gpu-box:11434").api_base_url == \
        "http://gpu-box:11434"


def test_a_trailing_slash_is_normalised():
    """Otherwise the request path becomes //api/chat."""
    assert OllamaClient(api_base_url="http://x:1234/").api_base_url == "http://x:1234"


def test_the_endpoint_is_built_from_the_base():
    text = source("modules/llm_client.py")

    assert '"http://localhost:11434/api/chat"' not in text
    assert 'f"{self.api_base_url}/api/chat"' in text


def test_the_error_message_names_the_configured_address():
    """A connection error pointing at localhost is misleading once overridden."""
    text = source("modules/llm_client.py")

    assert "is the Ollama server running at {self.api_base_url}?" in text


def test_the_facade_forwards_it():
    assert "OllamaClient(model, api_base_url)" in source("modules/llm_client.py")


def test_main_passes_the_base_url():
    assert "api_base_url=args.api_base_url" in source("main.py")
