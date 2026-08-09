"""
Module: Undo.

Reverses a previous run: returns to the base branch, and removes the branch the
agent created along with the commits on it.

Distinct from `--rollback`, which pops the pre-apply git stash and restores the
working tree. That helps while a run is in progress or immediately after one
that never committed. It does nothing about a branch and commits that already
exist, which is the state a successful run leaves behind.

Written to refuse rather than guess. Everything here removes work, and the
failure mode -- deleting a branch someone else made, or discarding a commit the
agent did not write -- is not recoverable from the tool.
"""

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

_LOG = logging.getLogger("agent.undo")

# Only branches the agent creates are eligible. A run with --branch-prefix set
# to something else records that prefix in its summary, so the check reads the
# recorded name rather than assuming.
DEFAULT_BRANCH_PREFIX = "agent"


class UndoError(RuntimeError):
    """Raised when a run cannot be undone safely."""


@dataclass
class RunSummary:
    run_id: str
    branch_name: Optional[str]
    outcome: str
    task: str
    path: str


def find_runs(log_dir: str) -> list:
    """Every run summary on disk, newest first."""
    runs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*_summary.json")), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(
            RunSummary(
                run_id=data.get("run_id") or os.path.basename(path),
                branch_name=data.get("branch_name"),
                outcome=data.get("outcome", "unknown"),
                task=data.get("task", ""),
                path=path,
            )
        )
    return runs


def find_run(log_dir: str, run_id: Optional[str] = None) -> RunSummary:
    """
    One run: the named one, or the most recent.

    Raises rather than falling back when a named run is missing. Undoing a
    different run than the one asked for is the worst possible outcome here.
    """
    runs = find_runs(log_dir)
    if not runs:
        raise UndoError(
            f"No run summaries found in {log_dir}. Nothing to undo."
        )

    if run_id is None:
        return runs[0]

    for run in runs:
        if run.run_id == run_id:
            return run

    known = ", ".join(r.run_id for r in runs[:5])
    raise UndoError(f"No run named {run_id}. Recent runs: {known}")


def describe(run: RunSummary) -> str:
    """What undoing this run would do, for a confirmation prompt."""
    lines = [
        f"  Run    : {run.run_id}",
        f"  Task   : {run.task[:68]}",
        f"  Outcome: {run.outcome}",
    ]
    if run.branch_name:
        lines.append(f"  Branch : {run.branch_name}  (will be deleted)")
    else:
        lines.append("  Branch : none recorded (nothing to delete)")
    return "\n".join(lines)


def undo_run(
    git,
    run: RunSummary,
    base_branch: str = "main",
    branch_prefix: str = DEFAULT_BRANCH_PREFIX,
) -> list:
    """
    Remove the run's branch, returning a description of what was done.

    `git` is any object with the `GitIntegration` surface, injected so this can
    be tested without a repository and so the module does not import it.
    """
    done = []

    if not run.branch_name:
        raise UndoError(
            f"Run {run.run_id} recorded no branch -- it ran with --no-git, or "
            f"git was unavailable. There is nothing for --undo to remove; if "
            f"changes are still in the working tree, use --rollback."
        )

    # Refused rather than checked-and-warned: a branch this tool did not create
    # may hold work nobody else has a copy of.
    if not run.branch_name.startswith(f"{branch_prefix}/"):
        raise UndoError(
            f"{run.branch_name} does not start with '{branch_prefix}/', so it "
            f"was not created by this tool. Delete it yourself if that is what "
            f"you want."
        )

    current = git.current_branch()
    if current == run.branch_name:
        result = git.checkout(base_branch)
        if not result.success:
            raise UndoError(
                f"Could not leave {run.branch_name} for {base_branch}: "
                f"{result.error.strip()}. Commit or stash your changes first."
            )
        done.append(f"switched to {base_branch}")

    result = git.delete_branch(run.branch_name, force=True)
    if not result.success:
        raise UndoError(
            f"Could not delete {run.branch_name}: {result.error.strip()}"
        )
    done.append(f"deleted {run.branch_name}")

    _LOG.info("Undid run %s: %s", run.run_id, "; ".join(done))
    return done
