"""
Tests for --interactive commit confirmation (modules/agent_loop.py).

The agent commits automatically once the suite goes green. --interactive adds a
review gate between "tests passed" and "git commit", which is the point the
issue asks for: after changes are applied, before anything is committed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig, AutonomousAgent  # noqa: E402
from modules.dry_run import ask_confirmation  # noqa: E402


class FakeGit:
    """Minimal stand-in for GitIntegration."""

    def __init__(self, diff=""):
        self._diff = diff

    def diff_unstaged(self, stat_only=False):
        return self._diff


def make_agent(monkeypatch, *, interactive, yes, git=None, answer="y"):
    """Build an AutonomousAgent without running __init__ side effects."""
    agent = object.__new__(AutonomousAgent)
    agent.config = AgentConfig(
        repo_root=".",
        task="t",
        interactive=interactive,
        yes=yes,
    )
    agent.git = git
    agent.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    monkeypatch.setattr("builtins.input", lambda *a, **k: answer)
    return agent


# ── gating ────────────────────────────────────────────────────────────────


def test_no_prompt_when_interactive_is_off(monkeypatch):
    """Default behaviour is unchanged: commit proceeds with no prompt."""

    def explode(*a, **k):
        raise AssertionError("input() must not be called without --interactive")

    monkeypatch.setattr("builtins.input", explode)
    agent = make_agent(monkeypatch, interactive=False, yes=False)
    monkeypatch.setattr("builtins.input", explode)

    assert agent._confirm_commit(["a.py"]) is True


def test_yes_bypasses_the_prompt(monkeypatch):
    """--yes wins over --interactive so CI never blocks on stdin."""

    def explode(*a, **k):
        raise AssertionError("input() must not be called when --yes is set")

    agent = make_agent(monkeypatch, interactive=True, yes=True)
    monkeypatch.setattr("builtins.input", explode)

    assert agent._confirm_commit(["a.py"]) is True


# ── answers ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "", "  "])
def test_affirmative_answers_approve(monkeypatch, answer):
    agent = make_agent(monkeypatch, interactive=True, yes=False, answer=answer)
    assert agent._confirm_commit(["a.py"]) is True


@pytest.mark.parametrize("answer", ["n", "N", "no", "No"])
def test_negative_answers_decline(monkeypatch, answer):
    agent = make_agent(monkeypatch, interactive=True, yes=False, answer=answer)
    assert agent._confirm_commit(["a.py"]) is False


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
def test_interrupt_declines_rather_than_crashing(monkeypatch, exc):
    agent = make_agent(monkeypatch, interactive=True, yes=False)

    def raise_it(*a, **k):
        raise exc()

    monkeypatch.setattr("builtins.input", raise_it)
    assert agent._confirm_commit(["a.py"]) is False


# ── what the reviewer is shown ────────────────────────────────────────────


def test_diff_is_printed_for_review(monkeypatch, capsys):
    diff = "diff --git a/x.py b/x.py\n+added line\n"
    agent = make_agent(monkeypatch, interactive=True, yes=False, git=FakeGit(diff))

    agent._confirm_commit(["x.py"])
    out = capsys.readouterr().out
    assert "REVIEW BEFORE COMMIT" in out
    assert "+added line" in out


def test_long_diff_is_truncated(monkeypatch, capsys):
    diff = "\n".join(f"+line {i}" for i in range(1000))
    agent = make_agent(monkeypatch, interactive=True, yes=False, git=FakeGit(diff))

    agent._confirm_commit(["x.py"])
    out = capsys.readouterr().out
    assert "diff truncated" in out
    assert "+line 0" in out
    assert "+line 999" not in out


def test_falls_back_to_file_list_without_git(monkeypatch, capsys):
    agent = make_agent(monkeypatch, interactive=True, yes=False, git=None)

    agent._confirm_commit(["alpha.py", "beta.py"])
    out = capsys.readouterr().out
    assert "Git is disabled" in out
    assert "alpha.py" in out
    assert "beta.py" in out


def test_falls_back_to_file_list_when_diff_is_empty(monkeypatch, capsys):
    agent = make_agent(monkeypatch, interactive=True, yes=False, git=FakeGit(""))

    agent._confirm_commit(["alpha.py"])
    out = capsys.readouterr().out
    assert "Git reported no diff" in out
    assert "alpha.py" in out


# ── the shared prompt helper ──────────────────────────────────────────────


def test_default_prompt_wording_is_unchanged(monkeypatch, capsys):
    """The pre-apply prompt must read exactly as it did before."""
    monkeypatch.setattr("builtins.input", lambda prompt="": print(prompt) or "y")
    ask_confirmation(3)
    assert "Apply these 3 change(s)? [Y/n]:" in capsys.readouterr().out


def test_commit_prompt_uses_the_commit_verb(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": print(prompt) or "y")
    ask_confirmation(2, action="Commit")
    assert "Commit these 2 change(s)? [Y/n]:" in capsys.readouterr().out


# ── config surface ────────────────────────────────────────────────────────


def test_interactive_defaults_to_off():
    assert AgentConfig(repo_root=".", task="t").interactive is False


AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"


def test_aborted_outcome_has_a_real_message():
    """`aborted` must not fall through to the 'Unknown outcome' default."""
    source = AGENT_LOOP.read_text()
    assert '"aborted": (' in source
    assert "still in the working tree" in source


def test_every_commit_path_is_gated():
    """
    run() commits from two places: the normal tests-passed exit and the
    success_on_llm_done early exit. Both must go through _confirm_commit, or
    --interactive silently does nothing on one of them.
    """
    lines = AGENT_LOOP.read_text().splitlines()
    call_sites = [
        i for i, line in enumerate(lines)
        if "self._commit_changes(" in line and "def " not in line
    ]

    assert len(call_sites) == 2, (
        f"expected 2 commit call sites, found {len(call_sites)}"
    )

    for i in call_sites:
            # 20 rather than 12: the checkpoint added for #203 sits between the
            # confirmation and the commit. The guard is unchanged -- confirmation
            # must still precede every commit -- only the gap is wider.
        preceding = "\n".join(lines[max(0, i - 20):i])
        assert "_confirm_commit" in preceding, (
            f"_commit_changes at line {i + 1} is not gated by _confirm_commit"
        )
