"""
Module: Self-update.

Updates RepoPilot's own checkout to the latest upstream commit.

The design is deliberately conservative, because "fetch the latest code and run
it" is remote code execution by definition:

- It updates a **git checkout** rather than downloading and unpacking an
  archive, so every change is signed by the transport, attributable to a commit,
  and reversible with `git reset`.
- It **fast-forwards only**. A diverged local branch is reported, never
  overwritten.
- It **refuses on a dirty tree**, so nothing uncommitted is lost.
- It **shows the incoming commits and asks** before moving anything.
- It **prints the pip command rather than running it**. Installing packages as a
  side effect of `--update` is the kind of surprise that erodes trust in a tool.
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field

_LOG = logging.getLogger("agent.updater")

# The directory this file lives in, i.e. RepoPilot's own checkout -- not the
# repository the agent is being pointed at.
INSTALL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "main"
GIT_TIMEOUT = 60


@dataclass
class UpdateResult:
    ok: bool
    message: str
    commits: list[str] = field(default_factory=list)
    requirements_changed: bool = False


def _same_path(a: str, b: str) -> bool:
    """Compare two paths, tolerating symlinks and Windows case differences."""
    return (
        os.path.normcase(os.path.realpath(a))
        == os.path.normcase(os.path.realpath(b))
    )


def _git(*args: str, cwd: str = INSTALL_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GIT_TIMEOUT,
    )


def is_git_checkout(root: str = INSTALL_ROOT) -> bool:
    """
    True only if `root` is itself the top of a git checkout.

    `git rev-parse --is-inside-work-tree` walks *up* the directory tree, so it
    answers "true" for any directory nested inside a repository. Relying on it
    would let --update fast-forward an ancestor repo -- someone's home directory,
    or a parent project -- instead of RepoPilot. Comparing the discovered
    toplevel against `root` is what makes the refusal mean anything.
    """
    try:
        result = _git("rev-parse", "--show-toplevel", cwd=root)
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False

    toplevel = result.stdout.strip()
    if not toplevel:
        return False
    return _same_path(toplevel, root)


def has_local_changes(root: str = INSTALL_ROOT) -> bool:
    result = _git("status", "--porcelain", cwd=root)
    return bool(result.stdout.strip())


def current_branch(root: str = INSTALL_ROOT) -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).stdout.strip()


def check_for_updates(
    remote: str = DEFAULT_REMOTE,
    branch: str = DEFAULT_BRANCH,
    root: str = INSTALL_ROOT,
) -> UpdateResult:
    """Fetch and report what would change, without touching the working tree."""
    if not is_git_checkout(root):
        return UpdateResult(
            False,
            f"{root} is not a git checkout, so it cannot be updated this way. "
            f"Reinstall by cloning the repository.",
        )

    fetch = _git("fetch", remote, branch, cwd=root)
    if fetch.returncode != 0:
        return UpdateResult(
            False, f"Could not fetch {remote}/{branch}: {fetch.stderr.strip()}"
        )

    target = f"{remote}/{branch}"
    log = _git("log", "--oneline", f"HEAD..{target}", cwd=root)
    if log.returncode != 0:
        return UpdateResult(
            False, f"Could not compare against {target}: {log.stderr.strip()}"
        )

    commits = [line for line in log.stdout.splitlines() if line.strip()]
    if not commits:
        return UpdateResult(True, f"Already up to date with {target}.")

    # Only a fast-forward is safe. A diverged branch means local commits that a
    # merge or reset would discard or entangle.
    ahead = _git("rev-list", "--count", f"{target}..HEAD", cwd=root)
    if ahead.returncode == 0 and ahead.stdout.strip() not in ("", "0"):
        return UpdateResult(
            False,
            f"Your checkout has {ahead.stdout.strip()} commit(s) not on {target}. "
            f"Refusing to update automatically -- rebase or reset by hand.",
            commits=commits,
        )

    changed = _git("diff", "--name-only", f"HEAD..{target}", cwd=root)
    requirements_changed = any(
        line.strip().startswith("requirements")
        for line in changed.stdout.splitlines()
    )

    return UpdateResult(
        True,
        f"{len(commits)} new commit(s) available on {target}.",
        commits=commits,
        requirements_changed=requirements_changed,
    )


def apply_update(
    remote: str = DEFAULT_REMOTE,
    branch: str = DEFAULT_BRANCH,
    root: str = INSTALL_ROOT,
) -> UpdateResult:
    """Fast-forward to the fetched commit. Refuses anything else."""
    if not is_git_checkout(root):
        return UpdateResult(False, f"{root} is not the top of a git checkout.")

    if has_local_changes(root):
        return UpdateResult(
            False,
            "You have uncommitted changes in the RepoPilot checkout. "
            "Commit or stash them first -- an update will not discard them.",
        )

    merge = _git("merge", "--ff-only", f"{remote}/{branch}", cwd=root)
    if merge.returncode != 0:
        return UpdateResult(
            False,
            f"Fast-forward failed: {merge.stderr.strip() or merge.stdout.strip()}",
        )

    head = _git("rev-parse", "--short", "HEAD", cwd=root).stdout.strip()
    return UpdateResult(True, f"Updated to {head}.")


def run_update(
    remote: str = DEFAULT_REMOTE,
    branch: str = DEFAULT_BRANCH,
    assume_yes: bool = False,
    root: str = INSTALL_ROOT,
    confirm=input,
) -> int:
    """Drive the whole flow. Returns a process exit code."""
    print(f"Checking {root} for updates...")

    check = check_for_updates(remote, branch, root)
    print(f"  {check.message}")
    if not check.ok:
        return 1
    if not check.commits:
        return 0

    print()
    for line in check.commits[:20]:
        print(f"    {line}")
    if len(check.commits) > 20:
        print(f"    ... and {len(check.commits) - 20} more")
    print()

    if has_local_changes(root):
        print(
            "  Uncommitted changes in the checkout. Commit or stash them first.",
            file=sys.stderr,
        )
        return 1

    if not assume_yes:
        try:
            prompt = (
                f"Fast-forward {current_branch(root)} to {remote}/{branch}? [y/N]: "
            )
            answer = confirm(prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 1
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    result = apply_update(remote, branch, root)
    print(f"  {result.message}")
    if not result.ok:
        return 1

    if check.requirements_changed:
        # Printed, not executed. Installing packages as a side effect of an
        # update is a surprise nobody asked for.
        print()
        print("  Dependencies changed. Update them with:")
        print(f"    pip install -r {os.path.join(root, 'requirements.txt')}")

    return 0
