"""
Module: Parallel tasks.

Runs several tasks at once, each in its own git worktree.

The issue this implements asks to "run in parallel if they don't affect the same
files". That condition cannot be evaluated in advance: which files a task
touches is decided by the model, and is not known until after the call that
would already have happened. Any upfront guess is either so conservative that
nothing runs in parallel, or wrong in a way that corrupts a run.

Isolation sidesteps the question. `git worktree` gives each task a separate
checkout on its own branch, sharing one object store. Two tasks may edit the
same file because they are editing different copies of it, and the result is one
branch per task for a human to merge or discard. Nothing needs to be predicted.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional
from modules.errors import ExecutionError

_LOG = logging.getLogger("agent.parallel_tasks")

# Worktrees are checkouts; too many at once means many copies of the repository
# on disk and many test suites competing for the same cores.
MAX_PARALLEL_TASKS = 4

GIT_TIMEOUT = 120


class WorktreeError(ExecutionError, RuntimeError):
    """Raised when an isolated checkout cannot be prepared."""


@dataclass
class TaskOutcome:
    task: str
    branch: Optional[str] = None
    worktree: Optional[str] = None
    outcome: str = "not_run"
    message: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.outcome == "success"


@dataclass
class WorktreeSet:
    """Worktrees created for one batch, so they can all be removed together."""

    repo_root: str
    root: str
    paths: list = field(default_factory=list)

    def cleanup(self) -> None:
        for path in self.paths:
            _remove_worktree(self.repo_root, path)
        shutil.rmtree(self.root, ignore_errors=True)


def _git(repo_root: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo_root, *args],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=GIT_TIMEOUT,
    )


def _remove_worktree(repo_root: str, path: str) -> None:
    """Remove one worktree. Never raises: cleanup must not mask a run's result."""
    try:
        _git(repo_root, "worktree", "remove", "--force", path)
    except Exception as exc:
        _LOG.warning("could not remove worktree %s: %s", path, exc)
    shutil.rmtree(path, ignore_errors=True)


def slugify(task: str, index: int) -> str:
    """A short branch-safe name from a task description."""
    words = [w for w in "".join(
        c if c.isalnum() or c.isspace() else " " for c in task.lower()
    ).split()][:4]
    return f"{index:02d}-" + ("-".join(words) or "task")


def is_git_repo(repo_root: str) -> bool:
    """
    True only when repo_root is itself the top of a work tree.

    `--is-inside-work-tree` walks up, so any directory nested inside a
    repository answers yes -- and pointing at a subdirectory would then create
    worktrees for the parent repository, which is not what was asked for.
    Comparing against `--show-toplevel` is the same check modules/updater.py
    already makes.
    """
    result = _git(repo_root, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return False

    def norm(path: str) -> str:
        return os.path.normcase(os.path.realpath(path))

    return norm(result.stdout.strip()) == norm(repo_root)


def create_worktrees(
    repo_root: str,
    tasks: list,
    branch_prefix: str = "agent",
    base_branch: Optional[str] = None,
) -> WorktreeSet:
    """
    One worktree per task, each on a fresh branch.

    Raises rather than falling back to a shared checkout: running several agents
    against one working tree would have them overwrite each other's edits, which
    is worse than refusing.
    """
    if not is_git_repo(repo_root):
        raise WorktreeError(
            f"{repo_root} is not a git repository. Parallel tasks need git "
            f"worktrees for isolation; run tasks one at a time instead."
        )

    root = tempfile.mkdtemp(prefix="repopilot-tasks-")
    created = WorktreeSet(repo_root=repo_root, root=root)

    for index, task in enumerate(tasks, 1):
        branch = f"{branch_prefix}/{slugify(task, index)}"
        path = os.path.join(root, f"task-{index:02d}")
        args = ["worktree", "add", "-b", branch, path]
        if base_branch:
            args.append(base_branch)

        result = _git(repo_root, *args)
        if result.returncode != 0:
            created.cleanup()
            raise WorktreeError(
                f"Could not create a worktree for task {index}: "
                f"{result.stderr.strip()}"
            )
        created.paths.append(path)

    return created


def run_tasks(
    repo_root: str,
    tasks: list,
    run_one: Callable,
    max_workers: int = MAX_PARALLEL_TASKS,
    branch_prefix: str = "agent",
    base_branch: Optional[str] = None,
) -> list:
    """
    Run every task in its own worktree, concurrently.

    `run_one(task, worktree_path, branch)` performs a single run and returns
    something with `.outcome` and `.final_message`; it is injected so this module
    does not import the agent loop, and so tests can drive it without an API key.

    Worktrees are removed afterwards. The branches are not — they are the point,
    and deleting them would throw the work away.
    """
    if not tasks:
        return []

    worktrees = create_worktrees(repo_root, tasks, branch_prefix, base_branch)
    branches = [f"{branch_prefix}/{slugify(t, i)}" for i, t in enumerate(tasks, 1)]
    outcomes = [
        TaskOutcome(task=t, branch=b, worktree=p)
        for t, b, p in zip(tasks, branches, worktrees.paths)
    ]

    try:
        workers = max(1, min(max_workers, len(tasks)))
        _LOG.info("Running %d task(s) across %d worker(s).", len(tasks), workers)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one, o.task, o.worktree, o.branch): o
                for o in outcomes
            }
            for future in as_completed(futures):
                outcome = futures[future]
                try:
                    result = future.result()
                    outcome.outcome = getattr(result, "outcome", "unknown")
                    outcome.message = getattr(result, "final_message", "") or ""
                except Exception as exc:
                    # One task failing must not take the others down with it.
                    outcome.outcome = "error"
                    outcome.error = f"{type(exc).__name__}: {exc}"
                    _LOG.error("task failed: %s -- %s", outcome.task[:60], exc)
    finally:
        worktrees.cleanup()

    return outcomes


def render_summary(outcomes: list) -> str:
    """A table of what each task did, and the branch its work is on."""
    if not outcomes:
        return "No tasks were run."

    succeeded = sum(1 for o in outcomes if o.ok)
    lines = [
        "=" * 60,
        f"{len(outcomes)} task(s): {succeeded} succeeded, "
        f"{len(outcomes) - succeeded} did not",
        "=" * 60,
    ]
    for outcome in outcomes:
        mark = "OK  " if outcome.ok else "FAIL"
        lines.append(f"{mark} {outcome.task[:56]}")
        lines.append(f"     branch: {outcome.branch}  outcome: {outcome.outcome}")
        if outcome.error:
            lines.append(f"     error : {outcome.error}")
    lines.append("")
    lines.append("Worktrees have been removed. The branches remain -- review and")
    lines.append("merge the ones you want.")
    return "\n".join(lines)
