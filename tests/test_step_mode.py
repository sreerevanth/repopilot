"""
Tests for --step (modules/agent_loop.py).

The issue asks for a keypress to pause the loop at any moment. That is not
viable here and this does something narrower instead — see the PR for the
reasoning; in short, non-blocking keypress capture needs raw terminal mode,
which is unavailable whenever output is piped or CI is running, and both are
the normal case for this tool.

`--step` pauses at the iteration boundary, which is the only point where there
is anything coherent to inspect: the changes are applied and the tests have
reported. Mid-iteration the agent is inside an API call or a test run, and the
working tree is either untouched or half-written.
"""

import builtins
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig, AutonomousAgent  # noqa: E402

PASSED = SimpleNamespace(success=True)
FAILED = SimpleNamespace(success=False)


def agent(**config):
    instance = object.__new__(AutonomousAgent)
    instance.config = AgentConfig(repo_root=".", task="t", **config)
    instance.logger = SimpleNamespace(
        warning=lambda message: None, info=lambda message: None
    )
    return instance


@pytest.fixture
def terminal(monkeypatch):
    """Pretend stdin is a terminal, and script the answer."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def answer(text):
        monkeypatch.setattr(builtins, "input", lambda _: text)

    return answer


def paused(instance, iteration=1, result=PASSED):
    with redirect_stdout(io.StringIO()):
        return instance._pause_after_iteration(iteration, result)


# ── off unless asked for ──────────────────────────────────────────────────


def test_it_is_off_by_default():
    assert AgentConfig(repo_root=".", task="t").step is False


def test_no_step_flag_never_pauses():
    assert paused(agent()) is True


def test_yes_overrides_it(terminal):
    """
    --yes means unattended, and main.py sets it automatically under CI. A run
    told not to ask questions must not stop on one.
    """
    terminal("s")

    assert paused(agent(step=True, yes=True)) is True


# ── it needs a terminal ───────────────────────────────────────────────────


def test_without_a_terminal_it_continues(monkeypatch):
    """
    Piped output is the normal case -- most commands in this repository's own
    documentation pipe. Blocking on a prompt nobody can answer would hang.
    """
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    assert paused(agent(step=True)) is True


def test_the_warning_is_printed_once(monkeypatch):
    """Once per run, not once per iteration."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    warnings = []
    instance = agent(step=True)
    instance.logger = SimpleNamespace(warning=warnings.append, info=lambda m: None)

    for iteration in (1, 2, 3):
        with redirect_stdout(io.StringIO()):
            instance._pause_after_iteration(iteration, PASSED)

    assert len(warnings) == 1


# ── the answers ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("answer", ["", "   ", "y", "anything"])
def test_continuing_is_the_default(terminal, answer):
    """Enter is the common case and must not need a word typed."""
    terminal(answer)

    assert paused(agent(step=True)) is True


@pytest.mark.parametrize("answer", ["s", "S", "stop", "q", "quit"])
def test_stopping_is_accepted_several_ways(terminal, answer):
    terminal(answer)

    assert paused(agent(step=True)) is False


@pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
def test_an_interrupt_at_the_prompt_stops(terminal, monkeypatch, interrupt):
    """
    Ctrl+D and Ctrl+C mean stop. Treating them as "continue" would carry on
    spending money after someone tried to bail out.
    """
    def raising(_):
        raise interrupt()

    monkeypatch.setattr(builtins, "input", raising)

    assert paused(agent(step=True)) is False


# ── what it shows ─────────────────────────────────────────────────────────


def test_the_banner_names_the_iteration(terminal):
    terminal("")
    output = io.StringIO()
    with redirect_stdout(output):
        agent(step=True)._pause_after_iteration(3, PASSED)

    assert "iteration 3" in output.getvalue()


@pytest.mark.parametrize(
    "result,expected", [(PASSED, "tests passed"), (FAILED, "tests did not pass")]
)
def test_the_banner_reflects_the_test_result(terminal, result, expected):
    """Whether to keep going usually depends on this."""
    terminal("")
    output = io.StringIO()
    with redirect_stdout(output):
        agent(step=True)._pause_after_iteration(1, result)

    assert expected in output.getvalue()


def test_it_says_what_can_be_inspected(terminal):
    terminal("")
    output = io.StringIO()
    with redirect_stdout(output):
        agent(step=True)._pause_after_iteration(1, PASSED)

    assert "git diff" in output.getvalue()


def test_both_choices_are_shown(terminal):
    terminal("")
    output = io.StringIO()
    with redirect_stdout(output):
        agent(step=True)._pause_after_iteration(1, PASSED)

    text = output.getvalue()
    assert "continue" in text and "stop here" in text


# ── wiring ────────────────────────────────────────────────────────────────


def test_the_pause_is_at_the_iteration_boundary():
    """
    Not mid-iteration: there the agent is inside an API call or a test run and
    the working tree is half-written, so there is nothing coherent to look at.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert source.index("_pause_after_iteration(iteration, last_exec)") > \
        source.index("# ── More iterations needed ──")


def test_stopping_ends_the_run_cleanly():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("if not self._pause_after_iteration"):][:200]

    assert 'outcome = "stopped"' in block
    assert "break" in block


def test_the_outcome_explains_where_the_changes_are():
    """
    A stopped run leaves changes in the tree; saying so is the difference
    between a deliberate stop and an apparent failure.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "Stopped at your request between iterations" in source
    assert "--rollback" in source


def test_main_registers_and_passes_the_flag():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

    assert '"--step"' in source
    assert "step=args.step" in source
