"""
Tests for self-update (modules/updater.py).

"Fetch the latest code and run it" is remote code execution by definition, so
most of what follows is about the refusals rather than the happy path. These
drive real git repositories; they skip if git is unavailable.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.updater import (  # noqa: E402
    apply_update,
    check_for_updates,
    current_branch,
    has_local_changes,
    is_git_checkout,
    run_update,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True
    )


@pytest.fixture
def upstream(tmp_path):
    """A bare remote with one commit on main."""
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q")
    git(seed, "config", "user.email", "t@example.com")
    git(seed, "config", "user.name", "Test")
    git(seed, "checkout", "-qb", "main")
    (seed / "version.txt").write_text("1\n")
    (seed / "requirements.txt").write_text("anthropic\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "initial")

    bare = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(bare)], check=True)
    return bare


@pytest.fixture
def install(tmp_path, upstream):
    """A clone standing in for RepoPilot's own checkout."""
    path = tmp_path / "install"
    subprocess.run(["git", "clone", "-q", str(upstream), str(path)], check=True)
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "Test")
    return path


def push_upstream(tmp_path, upstream, filename, content, message="upstream change"):
    """Land a commit on the remote from a separate clone."""
    other = tmp_path / f"other-{filename}"
    if not other.exists():
        subprocess.run(["git", "clone", "-q", str(upstream), str(other)], check=True)
        git(other, "config", "user.email", "o@example.com")
        git(other, "config", "user.name", "Other")
    (other / filename).write_text(content)
    git(other, "add", "-A")
    git(other, "commit", "-qm", message)
    git(other, "push", "-q", "origin", "main")


# ── detection ─────────────────────────────────────────────────────────────


def test_a_checkout_is_recognised(install):
    assert is_git_checkout(str(install)) is True


def test_a_plain_directory_is_not(tmp_path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    assert is_git_checkout(str(plain)) is False


def test_a_non_checkout_is_refused(tmp_path):
    """Downloading a tarball over it is not an acceptable fallback."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    result = check_for_updates(root=str(plain))

    assert result.ok is False
    assert "not a git checkout" in result.message


def test_local_changes_are_detected(install):
    assert has_local_changes(str(install)) is False
    (install / "version.txt").write_text("edited\n")
    assert has_local_changes(str(install)) is True


def test_current_branch_is_reported(install):
    assert current_branch(str(install)) == "main"


# ── checking ──────────────────────────────────────────────────────────────


def test_up_to_date_is_reported(install):
    result = check_for_updates(root=str(install))

    assert result.ok is True
    assert result.commits == []
    assert "up to date" in result.message


def test_new_commits_are_listed(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "version.txt", "2\n", "bump version")
    result = check_for_updates(root=str(install))

    assert result.ok is True
    assert len(result.commits) == 1
    assert "bump version" in result.commits[0]


def test_checking_does_not_touch_the_working_tree(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "version.txt", "2\n")
    check_for_updates(root=str(install))

    assert (install / "version.txt").read_text() == "1\n"


def test_a_requirements_change_is_flagged(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "requirements.txt", "anthropic\npathspec\n")
    assert check_for_updates(root=str(install)).requirements_changed is True


def test_an_unrelated_change_is_not_flagged(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "version.txt", "2\n")
    assert check_for_updates(root=str(install)).requirements_changed is False


def test_an_unreachable_remote_is_reported(install):
    result = check_for_updates(remote="nope", root=str(install))

    assert result.ok is False
    assert "Could not fetch" in result.message


# ── refusals ──────────────────────────────────────────────────────────────


def test_a_diverged_checkout_is_refused(tmp_path, upstream, install):
    """
    Local commits mean a merge or reset would entangle or discard work. Only a
    fast-forward is safe without asking a human.
    """
    push_upstream(tmp_path, upstream, "version.txt", "2\n")
    (install / "local.txt").write_text("my own work\n")
    git(install, "add", "-A")
    git(install, "commit", "-qm", "local work")

    result = check_for_updates(root=str(install))

    assert result.ok is False
    assert "not on" in result.message


def test_a_dirty_tree_is_refused(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "version.txt", "2\n")
    check_for_updates(root=str(install))
    (install / "scratch.txt").write_text("uncommitted\n")

    result = apply_update(root=str(install))

    assert result.ok is False
    assert "uncommitted" in result.message
    assert (install / "scratch.txt").exists()  # not discarded


def test_apply_is_fast_forward_only(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "version.txt", "2\n")
    git(install, "fetch", "-q", "origin", "main")
    (install / "local.txt").write_text("mine\n")
    git(install, "add", "-A")
    git(install, "commit", "-qm", "local")

    result = apply_update(root=str(install))

    assert result.ok is False
    assert "Fast-forward failed" in result.message


# ── applying ──────────────────────────────────────────────────────────────


def test_a_clean_fast_forward_succeeds(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "version.txt", "2\n")
    check_for_updates(root=str(install))

    result = apply_update(root=str(install))

    assert result.ok is True
    assert (install / "version.txt").read_text() == "2\n"


# ── the flow ──────────────────────────────────────────────────────────────


def test_confirmation_is_required(tmp_path, upstream, install, capsys):
    push_upstream(tmp_path, upstream, "version.txt", "2\n")

    code = run_update(root=str(install), confirm=lambda prompt: "n")

    assert code == 1
    assert (install / "version.txt").read_text() == "1\n"  # unchanged


def test_yes_skips_the_prompt(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "version.txt", "2\n")

    def explode(prompt):
        raise AssertionError("should not prompt when assume_yes is set")

    assert run_update(root=str(install), assume_yes=True, confirm=explode) == 0
    assert (install / "version.txt").read_text() == "2\n"


def test_up_to_date_exits_cleanly_without_prompting(install):
    def explode(prompt):
        raise AssertionError("nothing to confirm")

    assert run_update(root=str(install), confirm=explode) == 0


def test_incoming_commits_are_shown_before_asking(tmp_path, upstream, install, capsys):
    push_upstream(tmp_path, upstream, "version.txt", "2\n", "a visible commit message")
    run_update(root=str(install), confirm=lambda prompt: "n")

    assert "a visible commit message" in capsys.readouterr().out


def test_dependencies_are_printed_not_installed(tmp_path, upstream, install, capsys):
    """
    Running pip as a side effect of --update is a surprise nobody asked for.
    """
    push_upstream(tmp_path, upstream, "requirements.txt", "anthropic\npathspec\n")
    run_update(root=str(install), assume_yes=True)

    out = capsys.readouterr().out
    assert "pip install -r" in out
    assert "Dependencies changed" in out


def test_an_interrupt_at_the_prompt_aborts(tmp_path, upstream, install):
    push_upstream(tmp_path, upstream, "version.txt", "2\n")

    def interrupt(prompt):
        raise KeyboardInterrupt()

    assert run_update(root=str(install), confirm=interrupt) == 1
    assert (install / "version.txt").read_text() == "1\n"
