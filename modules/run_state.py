"""
Module: Run state.

A run that dies at iteration 4 — a dropped connection, a rate limit, Ctrl+C —
throws away three iterations of paid API calls. This persists the little state
`run()` actually needs to pick up where it left off.

Only four things matter for resuming: which iteration was reached, the changes
the model last proposed, the execution output those changes produced, and the
branch being worked on. Everything else (the repo, the context) is re-derived
from disk on the next iteration anyway, which is what makes this small.

The file is written after every iteration and removed when a run finishes, so
its presence means "this run did not complete".
"""

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Optional
from modules.errors import StateError

_LOG = logging.getLogger("agent.run_state")

# Bumped when the shape changes. A state file from an older version is ignored
# rather than half-understood -- resuming from a misread state would apply
# changes the model never proposed.
STATE_VERSION = 1


class ResumeError(StateError, RuntimeError):
    """Raised when a run cannot be resumed from the requested state."""


@dataclass
class RunState:
    run_id: str
    task: str
    repo_root: str
    iteration: int                      # the last iteration that completed
    branch_name: Optional[str] = None
    last_changes: list = field(default_factory=list)   # serialised FileChange
    last_exit_code: Optional[int] = None
    last_stdout: str = ""
    last_stderr: str = ""
    version: int = STATE_VERSION


def state_path(log_dir: str, run_id: str) -> str:
    return os.path.join(log_dir, f"{run_id}_state.json")


def save_state(state: RunState, log_dir: str) -> Optional[str]:
    """
    Write the state file atomically.

    Written to a temporary file and renamed, because this is saved after every
    iteration and a crash mid-write is exactly the case it exists to survive --
    a half-written file would be worse than none.

    Never raises: failing to checkpoint must not fail a run that is otherwise
    going fine.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        path = state_path(log_dir, state.run_id)
        handle, tmp = tempfile.mkstemp(dir=log_dir, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(asdict(state), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path
    except Exception as exc:
        _LOG.warning("could not save run state: %s", exc)
        return None


def clear_state(log_dir: str, run_id: str) -> None:
    """Remove the state file. A finished run has nothing to resume."""
    try:
        os.remove(state_path(log_dir, run_id))
    except OSError:
        pass


def load_state(log_dir: str, run_id: str) -> RunState:
    """Read a state file, or raise ResumeError explaining why it cannot be used."""
    path = state_path(log_dir, run_id)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ResumeError(
            f"No resumable state for run '{run_id}'. Either the run completed, "
            f"or it never wrote a checkpoint. Looked in {path}."
        ) from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(f"Could not read {path}: {exc}") from exc

    version = data.get("version")
    if version != STATE_VERSION:
        raise ResumeError(
            f"State file {path} is version {version}, this build expects "
            f"{STATE_VERSION}. Start a fresh run rather than resuming."
        )

    known = set(RunState.__dataclass_fields__)

    # Unknown keys were already filtered, but the values that remain were not
    # checked. Dataclass annotations are not enforced at runtime, so a file with
    # the right keys and wrong values loaded cleanly and failed later, somewhere
    # else, as a type the caller was not expecting.
    try:
        state = RunState(**{k: v for k, v in data.items() if k in known})
    except TypeError as exc:
        # A missing required field raised TypeError straight out of __init__.
        # ResumeError is an AgentError, so it reaches the handler that prints a
        # remedy; a bare TypeError escaped as an unhandled traceback instead.
        raise ResumeError(
            f"State file {path} is missing a required field ({exc}). "
            f"Start a fresh run rather than resuming."
        ) from exc

    # iteration is the one that failed silently. A negative value made the loop
    # start below zero and run more iterations than --max-iter allows, each one
    # a paid API call, with nothing reported. A string raised deep inside the
    # loop instead, far from the file that caused it.
    if not isinstance(state.iteration, int) or isinstance(state.iteration, bool):
        raise ResumeError(
            f"State file {path} has iteration={state.iteration!r}, which is not "
            f"an integer. The file is corrupt; start a fresh run."
        )
    if state.iteration < 0:
        raise ResumeError(
            f"State file {path} has iteration={state.iteration}, which is "
            f"negative. Resuming would run past --max-iter. Start a fresh run."
        )

    for field, value in (("run_id", state.run_id), ("task", state.task),
                         ("repo_root", state.repo_root)):
        if not isinstance(value, str):
            raise ResumeError(
                f"State file {path} has {field}={value!r}, expected a string. "
                f"The file is corrupt; start a fresh run."
            )

    if not isinstance(state.last_changes, list):
        raise ResumeError(
            f"State file {path} has last_changes={state.last_changes!r}, "
            f"expected a list. The file is corrupt; start a fresh run."
        )

    return state


def list_resumable(log_dir: str) -> list[str]:
    """Run ids with a state file present, newest first."""
    try:
        entries = [
            name for name in os.listdir(log_dir) if name.endswith("_state.json")
        ]
    except OSError:
        return []

    entries.sort(
        key=lambda name: os.path.getmtime(os.path.join(log_dir, name)),
        reverse=True,
    )
    return [name[: -len("_state.json")] for name in entries]


def check_resumable(
    state: RunState, repo_root: str, task: Optional[str] = None
) -> None:
    """
    Refuse to resume into a different repository or a different task.

    Both would silently apply one run's half-finished changes to another run's
    problem, which is worse than starting over.
    """
    if os.path.realpath(state.repo_root) != os.path.realpath(repo_root):
        raise ResumeError(
            f"Run '{state.run_id}' was against {state.repo_root}, not {repo_root}. "
            f"Resume from the same repository, or start a fresh run."
        )
    if task and task.strip() and task.strip() != state.task.strip():
        raise ResumeError(
            f"Run '{state.run_id}' was working on a different task. Resume without "
            f"--task to continue the original, or start a fresh run."
        )
