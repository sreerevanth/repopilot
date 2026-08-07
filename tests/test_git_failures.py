"""
Tests for git failure handling (modules/git_integration.py).

Two real failures are covered. `create_branch` used to fall back to a plain
checkout when `checkout -b` failed, which reported success while putting the
agent on a pre-existing branch that did not contain the base — it then worked
against a stale tree. And a rejected push returned git's raw stderr with no
diagnosis and no recovery.

The integration tests drive real repositories; they skip if git is unavailable.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.git_integration import (  # noqa: E402
    PUSH_REMEDIES,
    GitIntegration,
    classify_push_failure,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def run(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True
    )


@pytest.fixture
def remote(tmp_path):
    path = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(path)], check=True)
    return path


@pytest.fixture
def repo(tmp_path, remote):
    """A clone with one commit on `main`, pushed."""
    path = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(remote), str(path)], check=True)
    run(path, "config", "user.email", "t@example.com")
    run(path, "config", "user.name", "Test")
    run(path, "checkout", "-qb", "main")
    (path / "f.txt").write_text("v1\n")
    run(path, "add", "-A")
    run(path, "commit", "-qm", "init")
    run(path, "push", "-q", "-u", "origin", "main")
    return path


@pytest.fixture
def git(repo):
    return GitIntegration(str(repo))


def commit(git_obj, repo_path, name, content):
    (repo_path / name).write_text(content)
    git_obj.stage_all()
    git_obj.commit(f"add {name}")


# ── classifying push failures ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("! [rejected] main -> main (fetch first)", "non-fast-forward"),
        ("Updates were rejected because the remote contains work", "non-fast-forward"),
        ("remote: error: GH006: Protected branch update failed", "protected"),
        ("'origin' does not appear to be a git repository", "no-remote"),
        ("Permission denied (publickey).", "auth"),
        ("fatal: Authentication failed for 'https://github.com/x'", "auth"),
        ("could not read Username for 'https://github.com'", "auth"),
        ("ssh: Could not resolve hostname github.com", "network"),
        ("something nobody has seen before", "unknown"),
        ("", "unknown"),
    ],
)
def test_push_failures_are_classified(stderr, expected):
    assert classify_push_failure(stderr) == expected


def test_every_reason_has_a_remedy():
    """A classification with no advice attached is no better than raw stderr."""
    for reason in set(PUSH_REMEDIES):
        assert PUSH_REMEDIES[reason].strip()


# ── create_branch ─────────────────────────────────────────────────────────


def test_creates_a_branch_off_the_base(git):
    assert git.create_branch("agent/a", "main").success is True
    assert git.current_branch() == "agent/a"


def test_stale_existing_branch_is_refused(git, repo):
    """
    The regression this PR exists for: reusing the name silently put the agent
    on a branch missing the base branch's newer commits.
    """
    git.create_branch("agent/a", "main")
    commit(git, repo, "old.txt", "work from an earlier run\n")

    run(repo, "checkout", "-q", "main")
    commit(git, repo, "new.txt", "main has moved on\n")

    result = git.create_branch("agent/a", "main")

    assert result.success is False
    assert "stale" in result.error
    assert git.current_branch() == "main"  # not switched onto the stale branch


def test_up_to_date_existing_branch_is_reused(git):
    """Reuse is only dangerous when the branch is behind the base."""
    git.create_branch("agent/a", "main")
    run(git.repo_root, "checkout", "-q", "main")

    result = git.create_branch("agent/a", "main")

    assert result.success is True
    assert git.current_branch() == "agent/a"


def test_missing_base_branch_falls_back(git):
    """Pre-existing behaviour: fall back rather than fail outright."""
    assert git.create_branch("agent/b", "does-not-exist").success is True


def test_branch_exists_reports_accurately(git):
    assert git.branch_exists("agent/c") is False
    git.create_branch("agent/c", "main")
    assert git.branch_exists("agent/c") is True


# ── push ──────────────────────────────────────────────────────────────────


def test_pushing_a_new_branch_succeeds(git, repo):
    """
    An advanced base branch does not break a push of a new branch — worth
    pinning, since the issue assumed otherwise.
    """
    git.create_branch("agent/p", "main")
    commit(git, repo, "a.txt", "work\n")
    assert git.push("agent/p").success is True


def test_push_sets_upstream(git, repo):
    git.create_branch("agent/p", "main")
    commit(git, repo, "a.txt", "work\n")
    git.push("agent/p")

    tracking = run(repo, "rev-parse", "--abbrev-ref", "agent/p@{upstream}")
    assert tracking.stdout.strip() == "origin/agent/p"


def diverge_remote(tmp_path, remote, branch, filename, content):
    """Land a commit on `branch` from a second clone."""
    other = tmp_path / f"other-{filename}"
    subprocess.run(["git", "clone", "-q", str(remote), str(other)], check=True)
    run(other, "config", "user.email", "o@example.com")
    run(other, "config", "user.name", "Other")
    run(other, "checkout", "-q", branch)
    (other / filename).write_text(content)
    run(other, "add", "-A")
    run(other, "commit", "-qm", f"other {filename}")
    run(other, "push", "-q", "origin", branch)


def test_diverged_remote_is_rebased_and_retried(git, repo, tmp_path, remote):
    git.create_branch("agent/p", "main")
    commit(git, repo, "ours.txt", "ours\n")
    git.push("agent/p")

    diverge_remote(tmp_path, remote, "agent/p", "theirs.txt", "theirs\n")

    commit(git, repo, "ours2.txt", "ours again\n")
    result = git.push("agent/p")

    assert result.success is True
    assert (repo / "theirs.txt").exists()  # their commit was rebased in


def test_conflicting_rebase_is_aborted(git, repo, tmp_path, remote):
    """
    An unattended agent cannot resolve conflicts. Leaving the repository
    mid-rebase would be a worse outcome than a failed push.
    """
    git.create_branch("agent/p", "main")
    commit(git, repo, "shared.txt", "ours\n")
    git.push("agent/p")

    diverge_remote(tmp_path, remote, "agent/p", "shared.txt", "theirs\n")

    (repo / "shared.txt").write_text("ours changed\n")
    git.stage_all()
    git.commit("ours conflicting")

    result = git.push("agent/p")

    assert result.success is False
    assert "aborted" in result.error
    assert not (repo / ".git" / "rebase-merge").exists()
    assert not (repo / ".git" / "rebase-apply").exists()
    assert (repo / "shared.txt").read_text() == "ours changed\n"


def test_rebase_retry_can_be_disabled(git, repo, tmp_path, remote):
    git.create_branch("agent/p", "main")
    commit(git, repo, "ours.txt", "ours\n")
    git.push("agent/p")

    diverge_remote(tmp_path, remote, "agent/p", "theirs.txt", "theirs\n")

    commit(git, repo, "ours2.txt", "ours again\n")
    result = git.push("agent/p", retry_with_rebase=False)

    assert result.success is False
    assert not (repo / "theirs.txt").exists()  # no rebase happened


def test_failed_push_error_carries_a_remedy(git, repo, tmp_path, remote):
    git.create_branch("agent/p", "main")
    commit(git, repo, "ours.txt", "ours\n")
    git.push("agent/p")

    diverge_remote(tmp_path, remote, "agent/p", "theirs.txt", "theirs\n")

    commit(git, repo, "ours2.txt", "ours again\n")
    result = git.push("agent/p", retry_with_rebase=False)

    assert PUSH_REMEDIES["non-fast-forward"] in result.error


def test_missing_remote_is_diagnosed(git, repo):
    git.create_branch("agent/p", "main")
    commit(git, repo, "a.txt", "work\n")

    result = git.push("agent/p", remote="nope")

    assert result.success is False
    assert PUSH_REMEDIES["no-remote"] in result.error


def test_the_commit_survives_a_failed_push(git, repo):
    """The wording of every remedy promises this; it should be true."""
    git.create_branch("agent/p", "main")
    commit(git, repo, "a.txt", "work\n")
    head_before = run(repo, "rev-parse", "HEAD").stdout.strip()

    git.push("agent/p", remote="nope")

    assert run(repo, "rev-parse", "HEAD").stdout.strip() == head_before


# ── rebase_onto_remote ────────────────────────────────────────────────────


def test_rebase_reports_a_missing_remote_branch(git, repo):
    git.create_branch("agent/p", "main")
    commit(git, repo, "a.txt", "work\n")

    assert git.rebase_onto_remote("agent/never-pushed").success is False
