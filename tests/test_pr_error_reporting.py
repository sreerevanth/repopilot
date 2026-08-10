"""
Tests for pull request failure reporting (modules/git_integration.py).

`create_github_pr` returned `None` for a 401, a 403, a 404, a network failure
and every other HTTPError alike, and `agent_loop` did not report the `None`
either. A run finished, showed no PR URL, and gave no way to tell an expired
token from a missing scope from a repository the token cannot see.

The likeliest cause by a distance is a fine-grained PAT without
`pull_request: write` — a twenty-second fix once you know that is what it is.
"""

import io
import logging
import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.git_integration import GitIntegration  # noqa: E402


@pytest.fixture
def git(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return GitIntegration(str(tmp_path))


@pytest.fixture
def captured():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    log = logging.getLogger("agent.git")
    log.addHandler(handler)
    previous = log.level
    log.setLevel(logging.ERROR)
    yield stream
    log.removeHandler(handler)
    log.setLevel(previous)


def http_error(code, body):
    return urllib.error.HTTPError(
        "https://api.github.com", code, "", {}, io.BytesIO(body.encode())
    )


def attempt(git, error):
    with patch("urllib.request.urlopen", side_effect=error), \
         patch.dict("os.environ", {"GITHUB_TOKEN": "x"}), \
         patch.object(GitIntegration, "get_remote_url",
                      return_value="https://github.com/owner/repo.git"):
        return git.create_github_pr(
            title="t", body="b", head_branch="h", base_branch="main"
        )


# ── the reason is reported ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "code,body",
    [
        (401, '{"message":"Bad credentials"}'),
        (403, '{"message":"Resource not accessible by personal access token"}'),
        (404, '{"message":"Not Found"}'),
        (500, '{"message":"Server Error"}'),
    ],
)
def test_the_status_code_is_logged(git, captured, code, body):
    attempt(git, http_error(code, body))

    assert str(code) in captured.getvalue()


def test_the_response_body_is_passed_through(git, captured):
    """
    GitHub's bodies are specific, and the scope message is the one that turns
    an unexplained failure into a twenty-second fix.
    """
    attempt(git, http_error(403, '{"message":"Resource not accessible by personal access token"}'))

    assert "not accessible by personal access token" in captured.getvalue()


def test_a_network_failure_is_logged(git, captured):
    attempt(git, OSError("connection refused"))

    assert "connection refused" in captured.getvalue()


def test_every_failure_still_returns_none(git, captured):
    """The contract does not change; only the silence does."""
    assert attempt(git, http_error(401, "{}")) is None


# ── the case that already worked ──────────────────────────────────────────


def test_an_existing_pr_still_reports_itself(git, captured):
    """
    The 422 path already returned a useful message. It is what showed the
    machinery for reporting a reason existed and was used exactly once.
    """
    result = attempt(git, http_error(422, '{"message":"A pull request already exists"}'))

    assert result is not None
    assert "already exists" in result


def test_the_existing_pr_case_is_not_logged_as_an_error(git, captured):
    attempt(git, http_error(422, '{"message":"A pull request already exists"}'))

    assert captured.getvalue() == ""


# ── the caller ────────────────────────────────────────────────────────────


def test_the_loop_warns_when_no_pr_was_created():
    """
    Silence read as "the feature did not run" rather than "it ran and failed",
    and the branch had already been pushed.
    """
    source = (ROOT / "modules" / "agent_loop.py").read_text(encoding="utf-8")

    assert "--pr was requested but no pull request was" in source


def test_the_warning_says_the_branch_was_pushed():
    """The useful next step: the work is not lost, open one by hand."""
    source = (ROOT / "modules" / "agent_loop.py").read_text(encoding="utf-8")

    assert "open one by hand" in source


def test_no_bare_swallow_remains():
    source = (ROOT / "modules" / "git_integration.py").read_text(encoding="utf-8")
    start = source.index("def create_github_pr")

    assert "except Exception:\n            return None" not in source[start:]
