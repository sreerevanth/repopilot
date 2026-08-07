"""
Tests for rollback on Ctrl+C (modules/agent_loop.py).

`_run` rolls back applied files when the loop exits with a failure outcome. A
KeyboardInterrupt does not exit the loop — it unwinds past that code entirely,
which left half-applied changes in the working tree. `run()` now wraps `_run`
and applies the same rollback on the way out.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig, AgentRunResult, AutonomousAgent  # noqa: E402

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"


class RecordingModifier:
    """Stands in for CodeModificationEngine."""

    def __init__(self, restored=None, explode=False):
        self.calls: list = []
        self._restored = restored if restored is not None else []
        self._explode = explode

    def rollback(self, results):
        self.calls.append(results)
        if self._explode:
            raise OSError("disk gone")
        return self._restored


def make_agent(applied=None, modifier=None):
    """Build an agent without running __init__ side effects."""
    agent = object.__new__(AutonomousAgent)
    agent.config = AgentConfig(repo_root=".", task="t")
    agent.run_id = "run_test"
    agent.branch_name = "agent/test"
    agent._applied = applied if applied is not None else []
    agent.modifier = modifier or RecordingModifier()
    agent.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        finish_run=lambda *a, **k: None,
    )
    return agent


# ── the wrapper catches the interrupt ─────────────────────────────────────


def test_interrupt_is_caught_and_reported(monkeypatch):
    agent = make_agent(applied=["change"])

    def boom(self):
        raise KeyboardInterrupt()

    monkeypatch.setattr(AutonomousAgent, "_run", boom)

    result = agent.run()
    assert isinstance(result, AgentRunResult)
    assert result.outcome == "aborted"


def test_interrupt_triggers_rollback(monkeypatch):
    modifier = RecordingModifier(restored=["a.py", "b.py"])
    agent = make_agent(applied=["change"], modifier=modifier)

    monkeypatch.setattr(
        AutonomousAgent, "_run", lambda self: (_ for _ in ()).throw(KeyboardInterrupt())
    )

    agent.run()
    assert modifier.calls == [["change"]]


def test_normal_run_is_passed_through(monkeypatch):
    """The wrapper must not change behaviour when nothing is interrupted."""
    sentinel = AgentRunResult(
        run_id="r", outcome="success", branch_name="b",
        pr_url=None, iterations_used=2, final_message="done",
    )
    agent = make_agent()
    monkeypatch.setattr(AutonomousAgent, "_run", lambda self: sentinel)

    assert agent.run() is sentinel


def test_other_exceptions_still_propagate(monkeypatch):
    """Only KeyboardInterrupt is handled; real errors must not be swallowed."""
    agent = make_agent()
    monkeypatch.setattr(
        AutonomousAgent, "_run", lambda self: (_ for _ in ()).throw(ValueError("boom"))
    )

    with pytest.raises(ValueError):
        agent.run()


# ── rollback behaviour ────────────────────────────────────────────────────


def test_nothing_applied_means_nothing_rolled_back():
    modifier = RecordingModifier()
    agent = make_agent(applied=[], modifier=modifier)

    result = agent._handle_interrupt()
    assert modifier.calls == []
    assert "No file changes" in result.final_message


def test_restored_count_is_reported():
    modifier = RecordingModifier(restored=["a.py", "b.py"])
    agent = make_agent(applied=["c"], modifier=modifier)
    result = agent._handle_interrupt()
    assert "2 file(s)" in result.final_message


def test_applied_state_is_cleared_after_rollback():
    """A second interrupt must not roll the same changes back twice."""
    modifier = RecordingModifier(restored=["a.py"])
    agent = make_agent(applied=["c"], modifier=modifier)

    agent._handle_interrupt()
    agent._handle_interrupt()

    assert len(modifier.calls) == 1
    assert agent._applied == []


def test_rollback_failure_does_not_mask_the_abort():
    """A failing rollback is reported, not raised over the interrupt."""
    agent = make_agent(applied=["c"], modifier=RecordingModifier(explode=True))

    result = agent._handle_interrupt()  # must not raise
    assert result.outcome == "aborted"


def test_logging_failure_does_not_mask_the_abort():
    agent = make_agent(applied=["c"])

    def explode(*a, **k):
        raise RuntimeError("log sink gone")

    agent.logger.finish_run = explode
    assert agent._handle_interrupt().outcome == "aborted"


# ── the result is well formed ─────────────────────────────────────────────


def test_result_carries_run_id_and_branch():
    result = make_agent(applied=["c"])._handle_interrupt()
    assert result.run_id == "run_test"
    assert result.branch_name == "agent/test"


def test_no_pr_is_claimed_for_an_aborted_run():
    result = make_agent(applied=["c"])._handle_interrupt()
    assert result.pr_url is None


# ── structure ─────────────────────────────────────────────────────────────


def test_apply_results_are_mirrored_onto_the_instance():
    """
    _handle_interrupt reads self._applied because the loop's local is gone by
    the time the exception reaches the wrapper. If the mirror is dropped, the
    rollback silently becomes a no-op.
    """
    source = AGENT_LOOP.read_text()
    assert "self._applied = apply_results" in source


def test_run_delegates_to_run_underscore():
    source = AGENT_LOOP.read_text()
    assert "return self._run()" in source
    assert "except KeyboardInterrupt:" in source
