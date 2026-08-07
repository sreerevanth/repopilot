"""
Tests for resolving a task from a GitHub issue URL (modules/task_source.py).

A failure here is fatal by design. Passing an unresolved URL through as the
literal task would send the agent off to "implement
https://github.com/o/r/issues/1", which is worse than stopping.

No test contacts the network; urlopen is replaced throughout.
"""

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import task_source  # noqa: E402
from modules.task_source import (  # noqa: E402
    MAX_BODY_CHARS,
    MAX_COMMENTS,
    TaskResolutionError,
    compose_task,
    fetch_issue,
    looks_like_issue_url,
    parse_issue_url,
    resolve_task,
)

ISSUE = {
    "number": 75,
    "title": "Extract context from GitHub issues",
    "body": "Users should be able to pass a URL instead of a task string.",
    "comments": 2,
}
COMMENTS = [
    {"user": {"login": "maintainer"}, "body": "Agreed, fetch title and body."},
    {"user": {"login": "someone"}, "body": "And the comments."},
]


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def responder(mapping, raises=None):
    """Return a urlopen stub serving `mapping` of url-substring -> payload."""

    def _open(request, timeout=None):
        if raises is not None:
            raise raises
        url = request.full_url if hasattr(request, "full_url") else str(request)
        for fragment, payload in mapping.items():
            if fragment in url:
                return _Response(json.dumps(payload).encode("utf-8"))
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    return _open


@pytest.fixture(autouse=True)
def no_ambient_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)


# ── recognising a URL ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,owner,repo,number",
    [
        (
            "https://github.com/sreerevanth/repopilot/issues/75",
            "sreerevanth", "repopilot", 75,
        ),
        ("http://github.com/o/r/issues/1", "o", "r", 1),
        ("https://www.github.com/o/r/issues/42#issuecomment-1", "o", "r", 42),
        ("https://github.com/o/r/issues/9?foo=bar", "o", "r", 9),
        ("  https://github.com/o/r/issues/3  ", "o", "r", 3),
    ],
)
def test_issue_urls_are_parsed(url, owner, repo, number):
    parts = parse_issue_url(url)
    assert (parts["owner"], parts["repo"], parts["number"]) == (owner, repo, number)


@pytest.mark.parametrize(
    "value",
    [
        "Fix the parser TypeError",
        "https://github.com/o/r/pull/75",          # a PR, not an issue
        "https://gitlab.com/o/r/issues/1",         # not GitHub
        "https://github.com/o/r/issues/abc",       # not a number
        "https://github.com/o/r/issues",           # no number
        "",
        None,
    ],
)
def test_non_issue_values_are_not_matched(value):
    assert parse_issue_url(value) is None
    assert looks_like_issue_url(value) is False


def test_a_plain_description_passes_through_untouched():
    assert resolve_task("Fix the parser TypeError") == "Fix the parser TypeError"


# ── composing the task ────────────────────────────────────────────────────


def test_task_leads_with_number_and_title():
    assert compose_task(ISSUE).splitlines()[0] == (
        "GitHub issue #75: Extract context from GitHub issues"
    )


def test_body_is_included():
    assert ISSUE["body"] in compose_task(ISSUE)


def test_comments_are_attributed():
    text = compose_task(ISSUE, COMMENTS)
    assert "--- comment by maintainer ---" in text
    assert "And the comments." in text


def test_empty_comments_are_skipped():
    text = compose_task(ISSUE, [{"user": {"login": "x"}, "body": "   "}])
    assert "comment by x" not in text


def test_a_missing_body_still_yields_a_task():
    assert "only a title" in compose_task({"number": 1, "title": "only a title"})


def test_long_bodies_are_truncated():
    """The issue text competes with source files for the context budget."""
    text = compose_task({"number": 1, "title": "t", "body": "x" * (MAX_BODY_CHARS * 3)})
    assert "truncated" in text
    assert len(text) < MAX_BODY_CHARS * 2


def test_comment_count_is_capped():
    many = [
        {"user": {"login": f"u{i}"}, "body": f"c{i}"}
        for i in range(MAX_COMMENTS + 10)
    ]
    text = compose_task(ISSUE, many)
    assert f"u{MAX_COMMENTS + 5}" not in text


# ── fetching ──────────────────────────────────────────────────────────────


def test_issue_and_comments_are_fetched(monkeypatch):
    monkeypatch.setattr(
        task_source.urllib.request, "urlopen",
        responder({"/comments": COMMENTS, "/issues/75": ISSUE}),
    )
    result = fetch_issue("o", "r", 75)

    assert result["issue"]["title"] == ISSUE["title"]
    assert len(result["comments"]) == 2


def test_comment_failure_degrades_rather_than_aborting(monkeypatch):
    """Title and body alone are a usable task."""

    def _open(request, timeout=None):
        if "/comments" in request.full_url:
            raise TimeoutError("slow")
        return _Response(json.dumps(ISSUE).encode("utf-8"))

    monkeypatch.setattr(task_source.urllib.request, "urlopen", _open)
    result = fetch_issue("o", "r", 75)

    assert result["issue"]["title"] == ISSUE["title"]
    assert result["comments"] == []


@pytest.mark.parametrize(
    "code,expected",
    [
        (404, "not found"),
        (403, "rate limit"),
        (401, "rate limit"),
        (500, "HTTP 500"),
    ],
)
def test_http_errors_explain_themselves(monkeypatch, code, expected):
    error = urllib.error.HTTPError("u", code, "err", {}, None)
    monkeypatch.setattr(
        task_source.urllib.request, "urlopen", responder({}, raises=error)
    )

    with pytest.raises(TaskResolutionError) as excinfo:
        fetch_issue("o", "r", 1)

    assert expected.lower() in str(excinfo.value).lower()


def test_network_failure_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(
        task_source.urllib.request, "urlopen",
        responder({}, raises=OSError("no route to host")),
    )
    with pytest.raises(TaskResolutionError):
        fetch_issue("o", "r", 1)


def test_a_token_is_sent_when_available(monkeypatch):
    seen = {}

    def _open(request, timeout=None):
        seen["auth"] = request.headers.get("Authorization")
        return _Response(json.dumps(ISSUE).encode("utf-8"))

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setattr(task_source.urllib.request, "urlopen", _open)
    fetch_issue("o", "r", 1, include_comments=False)

    assert seen["auth"] == "Bearer ghp_secret"


def test_no_auth_header_without_a_token(monkeypatch):
    """Public repositories work unauthenticated."""
    seen = {}

    def _open(request, timeout=None):
        seen["auth"] = request.headers.get("Authorization")
        return _Response(json.dumps(ISSUE).encode("utf-8"))

    monkeypatch.setattr(task_source.urllib.request, "urlopen", _open)
    fetch_issue("o", "r", 1, include_comments=False)

    assert seen["auth"] is None


# ── end to end ────────────────────────────────────────────────────────────


def test_resolve_task_returns_the_composed_issue(monkeypatch):
    monkeypatch.setattr(
        task_source.urllib.request, "urlopen",
        responder({"/comments": COMMENTS, "/issues/75": ISSUE}),
    )
    task = resolve_task("https://github.com/o/r/issues/75")

    assert task.startswith("GitHub issue #75:")
    assert "maintainer" in task


def test_an_unresolvable_url_raises_rather_than_passing_through(monkeypatch):
    """Handing the model a bare URL as its task would be worse than stopping."""
    error = urllib.error.HTTPError("u", 404, "nf", {}, None)
    monkeypatch.setattr(
        task_source.urllib.request, "urlopen", responder({}, raises=error)
    )

    with pytest.raises(TaskResolutionError):
        resolve_task("https://github.com/o/r/issues/75")


def test_an_empty_issue_raises(monkeypatch):
    monkeypatch.setattr(
        task_source.urllib.request, "urlopen",
        responder({"/issues/": {"number": 1, "title": "", "body": ""}}),
    )
    with pytest.raises(TaskResolutionError):
        resolve_task("https://github.com/o/r/issues/1")
