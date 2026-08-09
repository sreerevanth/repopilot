"""
Tests for concurrent tasks (modules/parallel_tasks.py).

The issue asks to run tasks in parallel "if they don't affect the same files".
That condition cannot be evaluated in advance — which files a task touches is
decided by the model, and is not known until after the call that would already
have happened.

Isolation removes the question instead of answering it: each task gets its own
git worktree, so two tasks may edit the same file because they are editing
different copies. Most of what follows checks that isolation actually holds.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.parallel_tasks import (  # noqa: E402
    MAX_PARALLEL_TASKS,
    TaskOutcome,
    WorktreeError,
    create_worktrees,
    is_git_repo,
    render_summary,
    run_tasks,
    slugify,
)


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "shared.py").write_text("x = 0\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    return tmp_path


def writer(content_from_task=True):
    """A fake run that edits the same file in whichever worktree it is given."""

    def _run(task, worktree, branch):
        value = task if content_from_task else "fixed"
        Path(worktree, "shared.py").write_text(f"x = {value!r}\n")
        subprocess.run(["git", "-C", worktree, "add", "-A"], check=True)
        subprocess.run(["git", "-C", worktree, "commit", "-qm", task], check=True)
        return SimpleNamespace(outcome="success", final_message=f"did {task}")

    return _run


def show(repo, branch, path="shared.py"):
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{branch}:{path}"],
        capture_output=True, text=True,
    ).stdout.strip()


# ── isolation ─────────────────────────────────────────────────────────────


def test_tasks_touching_the_same_file_do_not_interfere(repo):
    """
    The case the issue says to avoid. With a worktree each it is not a problem,
    which is why nothing has to be predicted about file overlap.
    """
    tasks = ["fix the parser", "add caching", "update docs"]

    outcomes = run_tasks(str(repo), tasks, writer())

    assert all(o.ok for o in outcomes)
    for outcome in outcomes:
        assert show(repo, outcome.branch) == f"x = {outcome.task!r}"


def test_the_original_checkout_is_untouched(repo):
    run_tasks(str(repo), ["one", "two"], writer())

    assert (repo / "shared.py").read_text() == "x = 0\n"


def test_each_task_gets_its_own_branch(repo):
    outcomes = run_tasks(str(repo), ["alpha", "beta"], writer())

    branches = {o.branch for o in outcomes}
    assert len(branches) == 2


def test_branches_survive_the_run(repo):
    """
    They are the output. Cleaning them up with the worktrees would throw the
    work away.
    """
    outcomes = run_tasks(str(repo), ["alpha"], writer())
    listed = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", outcomes[0].branch],
        capture_output=True, text=True,
    ).stdout

    assert outcomes[0].branch in listed


def test_worktrees_are_removed(repo):
    """Leaving them behind would fill the disk over repeated runs."""
    outcomes = run_tasks(str(repo), ["alpha", "beta"], writer())

    for outcome in outcomes:
        assert not os.path.exists(outcome.worktree)


# ── failure handling ──────────────────────────────────────────────────────


def test_one_failing_task_does_not_stop_the_others(repo):
    def sometimes(task, worktree, branch):
        if task == "bad":
            raise RuntimeError("boom")
        return writer()(task, worktree, branch)

    outcomes = run_tasks(str(repo), ["good", "bad", "also good"], sometimes)

    assert sum(1 for o in outcomes if o.ok) == 2
    assert any(o.error and "boom" in o.error for o in outcomes)


def test_a_failure_is_reported_against_its_own_task(repo):
    def fail_one(task, worktree, branch):
        if task == "bad":
            raise ValueError("nope")
        return writer()(task, worktree, branch)

    outcomes = run_tasks(str(repo), ["good", "bad"], fail_one)
    failed = [o for o in outcomes if o.error]

    assert len(failed) == 1
    assert failed[0].task == "bad"


def test_worktrees_are_removed_even_when_a_task_raises(repo):
    def always_fail(task, worktree, branch):
        raise RuntimeError("boom")

    outcomes = run_tasks(str(repo), ["a", "b"], always_fail)

    for outcome in outcomes:
        assert not os.path.exists(outcome.worktree)


def test_a_non_git_directory_is_refused(tmp_path):
    """
    Running several agents against one shared checkout would have them
    overwrite each other. Refusing is better than that.
    """
    with pytest.raises(WorktreeError) as excinfo:
        create_worktrees(str(tmp_path), ["a", "b"])

    assert "not a git repository" in str(excinfo.value)


def test_no_tasks_is_a_no_op(repo):
    assert run_tasks(str(repo), [], writer()) == []


# ── naming ────────────────────────────────────────────────────────────────


def test_branch_names_are_derived_from_the_task():
    assert slugify("Fix the parser TypeError", 1) == "01-fix-the-parser-typeerror"


def test_branch_names_are_git_safe():
    """Spaces, slashes and quotes in a task must not reach a ref name."""
    name = slugify("fix: the parser's ~weird~ bug/thing", 3)

    for bad in (" ", "~", "'", ":", "/"):
        assert bad not in name


def test_the_index_keeps_similar_tasks_distinct():
    assert slugify("fix bug", 1) != slugify("fix bug", 2)


def test_an_unnameable_task_still_yields_a_branch():
    assert slugify("!!! ???", 7) == "07-task"


# ── concurrency bound ─────────────────────────────────────────────────────


def test_the_worker_count_is_bounded():
    """
    Each worktree is a checkout and each task runs a test suite; unbounded
    means many copies of the repo and many suites fighting for the same cores.
    """
    assert 0 < MAX_PARALLEL_TASKS <= 8


def test_more_workers_than_tasks_is_harmless(repo):
    outcomes = run_tasks(str(repo), ["only one"], writer(), max_workers=16)

    assert len(outcomes) == 1


# ── reporting ─────────────────────────────────────────────────────────────


def test_the_summary_counts_both_outcomes():
    outcomes = [
        TaskOutcome(task="a", branch="agent/01-a", outcome="success"),
        TaskOutcome(task="b", branch="agent/02-b", outcome="failed"),
    ]

    summary = render_summary(outcomes)

    assert "1 succeeded" in summary
    assert "1 did not" in summary


def test_the_summary_names_each_branch():
    summary = render_summary(
        [TaskOutcome(task="a", branch="agent/01-a", outcome="success")]
    )

    assert "agent/01-a" in summary


def test_the_summary_says_where_the_work_is():
    """Otherwise it is not obvious the branches are the deliverable."""
    summary = render_summary(
        [TaskOutcome(task="a", branch="agent/01-a", outcome="success")]
    )

    assert "branches remain" in summary


def test_an_empty_summary_says_so():
    assert "No tasks" in render_summary([])


def test_the_repo_root_is_recognised(repo):
    assert is_git_repo(str(repo)) is True


def test_a_plain_directory_is_not_a_repo(tmp_path):
    assert is_git_repo(str(tmp_path)) is False


def test_a_subdirectory_is_not_treated_as_the_repo(repo):
    """
    `--is-inside-work-tree` walks up, so a nested directory answers yes and
    worktrees would be created for the parent repository instead. Same trap the
    --update flag hit.
    """
    nested = repo / "src" / "deep"
    nested.mkdir(parents=True)

    assert is_git_repo(str(nested)) is False


# ── outcome pairing (#267) ────────────────────────────────────────────────


def test_the_outcome_zip_is_strict():
    """
    Without strict=True, a worktree that failed to create makes paths shorter
    than tasks, zip stops at the shortest, and the extra tasks disappear from
    the outcomes — the run reports success for what it produced and says
    nothing about what it dropped.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "parallel_tasks.py"
    ).read_text(encoding="utf-8")

    assert "zip(tasks, branches, worktrees.paths, strict=True)" in source


def test_mismatched_lengths_raise_rather_than_truncate():
    """The behaviour the flag buys, stated directly."""
    tasks, branches, paths = ["a", "b", "c"], ["x", "y", "z"], ["p", "q"]

    assert len(list(zip(tasks, branches, paths))) == 2

    with pytest.raises(ValueError):
        list(zip(tasks, branches, paths, strict=True))
