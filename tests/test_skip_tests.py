"""
Tests for --skip-tests (modules/agent_loop.py).

The agent normally decides success from a test run. --skip-tests removes that
evidence and accepts the model's own verdict instead, which is reasonable for a
refactor or a comment pass and unreasonable for a behaviour change.

The risk worth guarding is a log that reads like tests passed when none ran.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig, AutonomousAgent  # noqa: E402

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"


def source() -> str:
    # Explicit encoding: the default is locale-dependent and mangles this
    # module's box-drawing comments on a default Windows install.
    return AGENT_LOOP.read_text(encoding="utf-8")


def make_agent(skip_tests=False):
    agent = object.__new__(AutonomousAgent)
    agent.config = AgentConfig(repo_root=".", task="t", skip_tests=skip_tests)
    agent.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return agent


# ── config surface ────────────────────────────────────────────────────────


def test_skip_tests_defaults_to_off():
    assert AgentConfig(repo_root=".", task="t").skip_tests is False


# ── the stand-in result ───────────────────────────────────────────────────


@pytest.fixture
def result():
    return make_agent(skip_tests=True)._skipped_execution(0.92)


def test_the_result_counts_as_success(result):
    """Otherwise the loop would retry until max_iterations with nothing to fix."""
    assert result.success is True
    assert result.exit_code == 0


def test_it_does_not_claim_tests_ran(result):
    assert "skipped" in result.command
    assert "no test suite was run" in result.command
    assert "Tests were not run" in result.stdout


def test_it_records_the_confidence_the_decision_rested_on(result):
    assert "0.92" in result.stdout


def test_it_says_nothing_verified_the_change(result):
    """A reader of the log should not have to infer this."""
    assert "No suite verified this change" in result.stdout


def test_it_is_not_a_timeout(result):
    assert result.timed_out is False


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_confidence_is_rendered_for_any_value(confidence):
    stdout = make_agent(skip_tests=True)._skipped_execution(confidence).stdout
    assert f"{confidence:.2f}" in stdout


# ── wiring ────────────────────────────────────────────────────────────────


def test_the_sandbox_is_bypassed_when_skipping():
    """The point of the flag: no sandbox call at all."""
    text = source()
    start = text.index("if cfg.skip_tests:")
    block = text[start:start + 200]

    assert "_skipped_execution" in block
    assert block.index("_skipped_execution") < block.index("_run_execution")


def test_normal_runs_still_execute():
    text = source()
    start = text.index("if cfg.skip_tests:")
    assert "self._run_execution()" in text[start:start + 200]


def test_success_is_logged_differently_when_skipping():
    """"Tests passed" would be a false statement under --skip-tests."""
    text = source()
    start = text.index("Accepting iteration")
    block = text[max(0, start - 200):start + 200]

    assert "--skip-tests" in block
    assert "Tests passed" in text  # still used on the normal path
