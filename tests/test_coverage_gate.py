"""
Tests for the coverage gate (modules/sandbox.py, modules/agent_loop.py).

`--coverage` measures a baseline before the model touches anything, then fails
any iteration that lowers it and feeds the drop back as error output.

The design choice worth pinning is *drop-only*. An absolute threshold would fail
every run on a repository that already sits below it, for reasons the model did
not cause and cannot fix.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig, AutonomousAgent  # noqa: E402
from modules.sandbox import coverage_args, parse_coverage_percent  # noqa: E402

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"


def source() -> str:
    return AGENT_LOOP.read_text(encoding="utf-8")


REPORT = """\
Name              Stmts   Miss  Cover
-------------------------------------
modules/utils.py     40      5    88%
-------------------------------------
TOTAL               120     16    87%
"""


# ── parsing ───────────────────────────────────────────────────────────────


def test_a_total_is_parsed():
    assert parse_coverage_percent(REPORT) == 87.0


def test_a_decimal_total_is_parsed():
    assert parse_coverage_percent("TOTAL   120    16   86.5%") == 86.5


@pytest.mark.parametrize("value", ["4 passed in 0.1s", "", None])
def test_no_report_yields_none(value):
    """pytest-cov not installed, or a runner that does not report coverage."""
    assert parse_coverage_percent(value) is None


def test_the_last_total_wins():
    """A rerun or a nested report should not leave a stale figure."""
    assert parse_coverage_percent("TOTAL 1 1 50%\nTOTAL 1 1 91%") == 91.0


def test_per_file_lines_are_not_mistaken_for_the_total():
    assert parse_coverage_percent(REPORT) != 88.0


def test_coverage_args_target_the_source_dir():
    assert coverage_args("modules") == ["--cov=modules", "--cov-report=term"]


def test_coverage_args_request_a_terminal_report():
    """The gate parses stdout, so a non-terminal report would yield nothing."""
    assert any("term" in a for a in coverage_args())


# ── the drop rule ─────────────────────────────────────────────────────────


def agent():
    obj = object.__new__(AutonomousAgent)
    obj.config = AgentConfig(repo_root=".", task="t", coverage=True)
    return obj


def test_a_drop_is_reported():
    message = agent()._coverage_feedback(87.0, 80.0)

    assert message is not None
    assert "87.0" in message and "80.0" in message


def test_the_message_tells_the_model_what_to_do():
    message = agent()._coverage_feedback(87.0, 80.0)

    assert "Restore the deleted tests" in message


def test_holding_steady_is_not_a_drop():
    assert agent()._coverage_feedback(87.0, 87.0) is None


def test_an_increase_is_not_a_drop():
    assert agent()._coverage_feedback(87.0, 91.0) is None


def test_a_low_but_stable_repo_is_not_penalised():
    """An absolute threshold would fail this run; a drop rule does not."""
    assert agent()._coverage_feedback(12.0, 12.0) is None


@pytest.mark.parametrize("before,after", [(None, 80.0), (87.0, None), (None, None)])
def test_a_missing_measurement_is_not_a_drop(before, after):
    """Without pytest-cov there is no signal; guessing would be worse."""
    assert agent()._coverage_feedback(before, after) is None


# ── wiring ────────────────────────────────────────────────────────────────


def test_coverage_is_off_by_default():
    config = AgentConfig(repo_root=".", task="t")
    assert config.coverage is False
    assert config.coverage_source == "."


def test_a_baseline_is_measured_before_the_loop():
    """A first-iteration regression is only visible against a prior figure."""
    text = source()
    assert text.index("baseline_coverage") < text.index("for iteration in range(")


def test_a_missing_baseline_is_a_warning_not_a_failure():
    text = source()
    block = text[text.index("baseline_coverage"):][:900]

    assert "Is " in block and "pytest-cov" in block
    assert "Continuing without the gate" in block


def test_a_drop_turns_a_passing_run_into_a_failure():
    """Otherwise the agent commits the regression and stops."""
    text = source()
    block = text[text.index("if cfg.coverage and exec_result.success:"):][:1200]

    assert "exit_code=1" in block
    assert "coverage gate" in block


def test_the_drop_message_reaches_the_model():
    """It is appended to stderr, which the retry prompt feeds back."""
    block = source()[source().index("if cfg.coverage and exec_result.success:"):][:1200]

    assert "stderr=" in block and "drop" in block


def test_run_file_mode_does_not_get_coverage_args():
    """--run-file runs one script; --cov would report on the wrong thing."""
    block = source()[source().index("if cfg.coverage and not cfg.run_file:"):][:300]

    assert "coverage_args" in block
