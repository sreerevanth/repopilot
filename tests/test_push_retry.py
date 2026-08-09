"""
Tests for push retries (modules/git_integration.py).

A push that fails because the network blipped is worth retrying; one that fails
because the token is wrong is not. `classify_push_failure` already told those
apart for the purpose of printing a remedy — this uses the same classification
to decide what to retry.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.git_integration import (  # noqa: E402
    PUSH_BACKOFF_SECONDS,
    PUSH_RETRIES,
    RETRYABLE_PUSH_REASONS,
    GitIntegration,
    GitResult,
)

NETWORK = "fatal: unable to access: Could not resolve host: github.com"
AUTH = "fatal: Authentication failed for 'https://github.com/x'"
PROTECTED = "remote: error: GH006: Protected branch update failed"
REJECTED = "! [rejected] main -> main (non-fast-forward)"


class FakeGit(GitIntegration):
    """A GitIntegration whose _run replays a scripted list of outcomes."""

    def __init__(self, outcomes):
        self.repo_root = "/tmp"
        self.outcomes = list(outcomes)
        self.attempts = 0

    def _run(self, cmd, **kwargs):
        self.attempts += 1
        success, error = self.outcomes.pop(0)
        return GitResult(command="git push", success=success, output="", error=error)


@pytest.fixture(autouse=True)
def no_real_waiting(monkeypatch):
    """Backoff is real time; the tests assert the arithmetic, not the sleeping."""
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


# ── what gets retried ─────────────────────────────────────────────────────


def test_only_network_failures_are_retryable():
    """
    Retrying an auth failure or a protected-branch refusal just delays the same
    error, and a non-fast-forward is handled by rebasing rather than waiting.
    """
    assert RETRYABLE_PUSH_REASONS == {"network"}


def test_a_transient_failure_recovers():
    git = FakeGit([(False, NETWORK), (True, "")])

    assert git.push("branch").success is True
    assert git.attempts == 2


def test_a_persistent_failure_gives_up():
    git = FakeGit([(False, NETWORK)] * 10)

    assert git.push("branch").success is False
    assert git.attempts == PUSH_RETRIES


def test_a_success_does_not_retry():
    git = FakeGit([(True, "")])

    git.push("branch")

    assert git.attempts == 1


@pytest.mark.parametrize("error", [AUTH, PROTECTED])
def test_non_transient_failures_are_not_retried(error):
    git = FakeGit([(False, error)] * 10)

    git.push("branch")

    assert git.attempts == 1


def test_the_final_error_is_preserved():
    """The caller still needs to see why it failed after the retries."""
    git = FakeGit([(False, NETWORK)] * 10)

    assert "Could not resolve host" in git.push("branch").error


# ── backoff ───────────────────────────────────────────────────────────────


def test_backoff_is_exponential(no_real_waiting):
    FakeGit([(False, NETWORK)] * 10).push("branch")

    assert no_real_waiting == [PUSH_BACKOFF_SECONDS, PUSH_BACKOFF_SECONDS * 2]


def test_there_is_no_wait_before_the_first_attempt(no_real_waiting):
    FakeGit([(True, "")]).push("branch")

    assert no_real_waiting == []


def test_no_wait_after_the_last_attempt(no_real_waiting):
    """Sleeping after the final failure delays the error for no purpose."""
    FakeGit([(False, NETWORK)] * 10).push("branch")

    assert len(no_real_waiting) == PUSH_RETRIES - 1


def test_the_retry_count_is_what_the_issue_asked_for():
    assert PUSH_RETRIES == 3


def test_the_backoff_is_short_enough_to_watch():
    """A person is waiting on this; a flake that has not cleared in a few
    seconds usually is not a flake."""
    total = sum(PUSH_BACKOFF_SECONDS * (2 ** i) for i in range(PUSH_RETRIES - 1))

    assert total <= 10
