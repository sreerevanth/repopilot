"""
Tests for --undo (modules/undo.py).

`--rollback` pops the pre-apply git stash and restores the working tree. That
helps during a run, or right after one that never committed. It does nothing
about a branch and commits that already exist, which is exactly what a
successful run leaves behind — so undoing one meant `git reset --hard HEAD~1`
and `git branch -D` by hand.

Most of what follows is about refusing. Everything here removes work, and
deleting the wrong branch is not recoverable from this tool.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.git_integration import GitIntegration  # noqa: E402
from modules.undo import (  # noqa: E402
    RunSummary,
    UndoError,
    describe,
    find_run,
    find_runs,
    undo_run,
)


def write_summary(log_dir, run_id, branch="agent/fix-it", outcome="success"):
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{run_id}_summary.json").write_text(json.dumps({
        "run_id": run_id, "branch_name": branch,
        "outcome": outcome, "task": "fix the parser",
    }))


@pytest.fixture
def repo(tmp_path):
    """A repo on main, with an agent branch holding one commit."""
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], capture_output=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    run("add", "-A")
    run("commit", "-qm", "init")
    run("checkout", "-qb", "agent/fix-it")
    (tmp_path / "a.py").write_text("x = 2\n")
    run("add", "-A")
    run("commit", "-qm", "agent work")
    return tmp_path


def branches(repo):
    out = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list"], capture_output=True, text=True
    ).stdout
    return {line.strip("* ").strip() for line in out.splitlines() if line.strip()}


# ── finding a run ─────────────────────────────────────────────────────────


def test_the_most_recent_run_is_the_default(tmp_path):
    logs = tmp_path / "logs"
    write_summary(logs, "agent_20260101_000000_aaaaaa")
    write_summary(logs, "agent_20260202_000000_bbbbbb")

    assert find_run(str(logs)).run_id == "agent_20260202_000000_bbbbbb"


def test_a_named_run_is_found(tmp_path):
    logs = tmp_path / "logs"
    write_summary(logs, "agent_20260101_000000_aaaaaa")
    write_summary(logs, "agent_20260202_000000_bbbbbb")

    assert find_run(str(logs), "agent_20260101_000000_aaaaaa").run_id.endswith("aaaaaa")


def test_an_unknown_run_is_refused(tmp_path):
    """
    Not "fall back to the latest". Undoing a different run than the one asked
    for is the worst outcome available here.
    """
    logs = tmp_path / "logs"
    write_summary(logs, "agent_20260101_000000_aaaaaa")

    with pytest.raises(UndoError) as excinfo:
        find_run(str(logs), "agent_typo")

    assert "No run named" in str(excinfo.value)


def test_the_error_lists_what_is_available(tmp_path):
    logs = tmp_path / "logs"
    write_summary(logs, "agent_20260101_000000_aaaaaa")

    with pytest.raises(UndoError) as excinfo:
        find_run(str(logs), "wrong")

    assert "aaaaaa" in str(excinfo.value)


def test_no_runs_at_all_is_refused(tmp_path):
    with pytest.raises(UndoError, match="Nothing to undo"):
        find_run(str(tmp_path))


def test_a_corrupt_summary_is_skipped(tmp_path):
    logs = tmp_path / "logs"
    write_summary(logs, "agent_20260101_000000_good")
    (logs / "agent_20260202_000000_bad_summary.json").write_text("not json")

    assert len(find_runs(str(logs))) == 1


# ── what it would do ──────────────────────────────────────────────────────


def test_the_description_names_the_branch():
    text = describe(RunSummary("r1", "agent/fix-it", "success", "fix the parser", ""))

    assert "agent/fix-it" in text
    assert "will be deleted" in text


def test_a_run_without_a_branch_says_so():
    text = describe(RunSummary("r1", None, "success", "t", ""))

    assert "none recorded" in text


# ── undoing ───────────────────────────────────────────────────────────────


def test_the_branch_is_deleted(repo):
    undo_run(GitIntegration(str(repo)), RunSummary("r1", "agent/fix-it", "success", "t", ""))

    assert "agent/fix-it" not in branches(repo)


def test_it_leaves_the_branch_first(repo):
    """Git refuses to delete the branch you are standing on."""
    git = GitIntegration(str(repo))
    assert git.current_branch() == "agent/fix-it"

    undo_run(git, RunSummary("r1", "agent/fix-it", "success", "t", ""))

    assert git.current_branch() == "main"


def test_the_commit_goes_with_the_branch(repo):
    undo_run(GitIntegration(str(repo)), RunSummary("r1", "agent/fix-it", "success", "t", ""))

    assert (repo / "a.py").read_text() == "x = 1\n"


def test_it_reports_what_it_did(repo):
    done = undo_run(
        GitIntegration(str(repo)), RunSummary("r1", "agent/fix-it", "success", "t", "")
    )

    assert any("deleted" in step for step in done)


def test_undoing_from_another_branch_works(repo):
    """The common case: you already moved on before deciding to undo."""
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "main"], check=True)

    undo_run(GitIntegration(str(repo)), RunSummary("r1", "agent/fix-it", "success", "t", ""))

    assert "agent/fix-it" not in branches(repo)


# ── refusals ──────────────────────────────────────────────────────────────


def test_a_branch_we_did_not_create_is_refused(repo):
    """
    The refusal that matters. A branch this tool did not make may hold work
    nobody else has a copy of.
    """
    subprocess.run(["git", "-C", str(repo), "branch", "my-own-work"], check=True)

    with pytest.raises(UndoError) as excinfo:
        undo_run(
            GitIntegration(str(repo)),
            RunSummary("r1", "my-own-work", "success", "t", ""),
        )

    assert "not created by this tool" in str(excinfo.value)
    assert "my-own-work" in branches(repo)


def test_a_custom_prefix_is_honoured(repo):
    """A run with --branch-prefix set records that prefix; the check reads it."""
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "main"], check=True)
    subprocess.run(["git", "-C", str(repo), "branch", "bot/work"], check=True)

    undo_run(
        GitIntegration(str(repo)),
        RunSummary("r1", "bot/work", "success", "t", ""),
        branch_prefix="bot",
    )

    assert "bot/work" not in branches(repo)


def test_a_run_with_no_branch_is_refused():
    with pytest.raises(UndoError) as excinfo:
        undo_run(
            SimpleNamespace(), RunSummary("r1", None, "success", "t", "")
        )

    assert "--rollback" in str(excinfo.value)


def test_a_failed_checkout_stops_before_deleting():
    """
    Uncommitted work would be lost by a forced switch, so git refuses, and this
    must not delete the branch anyway.
    """
    git = SimpleNamespace(
        current_branch=lambda: "agent/fix-it",
        checkout=lambda b: SimpleNamespace(success=False, error="local changes"),
        delete_branch=lambda b, force=False: pytest.fail("should not be reached"),
    )

    with pytest.raises(UndoError, match="Commit or stash"):
        undo_run(git, RunSummary("r1", "agent/fix-it", "success", "t", ""))


# ── wiring ────────────────────────────────────────────────────────────────


def test_it_asks_before_deleting():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

    assert 'ask_confirmation("Delete this branch?")' in source


def test_yes_skips_the_prompt():
    """Matching every other confirmation, so CI never waits on stdin."""
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

    assert "skip_prompt = args.yes" in source


def test_the_delete_is_forced():
    """
    An agent branch is unmerged by definition when someone decides against it,
    so -d would refuse exactly the case this exists for.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "git_integration.py"
    ).read_text(encoding="utf-8")

    assert 'flag = "-D" if force else "-d"' in source


def test_main_registers_both_flags():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

    assert '"--undo"' in source
    assert '"--list-runs"' in source
