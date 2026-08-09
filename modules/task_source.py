"""
Module: Task sources.

`--task` normally carries a description. It can instead carry a GitHub issue
URL, in which case the title, body and comments are fetched and composed into
the task the agent actually works from.

Unlike notifications, a failure here is fatal by design. If the URL cannot be
resolved, passing it through as a literal task would send the agent off to
"implement https://github.com/o/r/issues/1", which is worse than stopping.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Optional
from modules.errors import ConfigurationError

_LOG = logging.getLogger("agent.task_source")

GITHUB_ISSUE_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/"
    r"(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/issues/(?P<number>\d+)"
    r"(?:[/?#].*)?$",
    re.IGNORECASE,
)

API_ROOT = "https://api.github.com"
DEFAULT_TIMEOUT = 20

# The issue body and its comments go into the prompt, where they compete with
# source files for the context budget. Long threads get trimmed rather than
# crowding the code out.
MAX_BODY_CHARS = 6000
MAX_COMMENT_CHARS = 1500
MAX_COMMENTS = 10


class TaskResolutionError(ConfigurationError, RuntimeError):
    """Raised when a task URL is recognised but cannot be turned into a task."""


def parse_issue_url(value: str) -> Optional[dict]:
    """Return {owner, repo, number} for a GitHub issue URL, else None."""
    match = GITHUB_ISSUE_URL.match((value or "").strip())
    if not match:
        return None
    parts = match.groupdict()
    parts["number"] = int(parts["number"])
    return parts


def looks_like_issue_url(value: str) -> bool:
    return parse_issue_url(value) is not None


def _get_json(url: str, token: Optional[str], timeout: int):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "repopilot-agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n[... truncated, {len(text) - limit} more chars]"


def fetch_issue(
    owner: str,
    repo: str,
    number: int,
    token: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    include_comments: bool = True,
) -> dict:
    """
    Fetch one issue and its comments.

    Works unauthenticated against public repositories; GITHUB_TOKEN raises the
    rate limit and is required for private ones.
    """
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    base = f"{API_ROOT}/repos/{owner}/{repo}/issues/{number}"

    try:
        issue = _get_json(base, token, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise TaskResolutionError(
                f"Issue {owner}/{repo}#{number} was not found. It may be private "
                f"or deleted; set GITHUB_TOKEN if the repository is private."
            ) from exc
        if exc.code in (401, 403):
            raise TaskResolutionError(
                f"GitHub refused the request for {owner}/{repo}#{number} "
                f"(HTTP {exc.code}). This is usually a rate limit or a missing "
                f"GITHUB_TOKEN."
            ) from exc
        raise TaskResolutionError(
            f"GitHub returned HTTP {exc.code} for {owner}/{repo}#{number}."
        ) from exc
    except Exception as exc:
        raise TaskResolutionError(
            f"Could not reach GitHub for {owner}/{repo}#{number}: {exc}"
        ) from exc

    comments: list[dict] = []
    if include_comments and issue.get("comments"):
        try:
            fetched = _get_json(f"{base}/comments", token, timeout)
            comments = fetched if isinstance(fetched, list) else []
        except Exception as exc:
            # The title and body alone are a usable task; losing the discussion
            # is a degradation, not a reason to abort.
            _LOG.warning("could not fetch comments for #%s: %s", number, exc)

    return {"issue": issue, "comments": comments}


def compose_task(issue: dict, comments: Optional[list] = None) -> str:
    """Render an issue and its comments as a task description."""
    number = issue.get("number", "?")
    title = (issue.get("title") or "").strip()
    body = _truncate(issue.get("body") or "", MAX_BODY_CHARS)

    if not title and not body:
        # "GitHub issue #1:" is a non-empty string but not a task. Returning it
        # would send the agent off with nothing to work from.
        return ""

    lines = [f"GitHub issue #{number}: {title}"]
    if body:
        lines += ["", body]

    for comment in (comments or [])[:MAX_COMMENTS]:
        author = (comment.get("user") or {}).get("login", "someone")
        text = _truncate(comment.get("body") or "", MAX_COMMENT_CHARS)
        if text:
            lines += ["", f"--- comment by {author} ---", text]

    return "\n".join(lines).strip()


def resolve_task(
    value: str,
    token: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Return the task to work from.

    A plain description passes through untouched. A GitHub issue URL is fetched
    and composed. Anything that looks like an issue URL but cannot be fetched
    raises, rather than being handed to the model as a literal URL.
    """
    parts = parse_issue_url(value)
    if not parts:
        return value

    _LOG.info(
        "Resolving task from %s/%s#%s",
        parts["owner"], parts["repo"], parts["number"],
    )
    fetched = fetch_issue(
        parts["owner"], parts["repo"], parts["number"], token=token, timeout=timeout
    )
    task = compose_task(fetched["issue"], fetched["comments"])

    if not task:
        raise TaskResolutionError(
            f"Issue {parts['owner']}/{parts['repo']}#{parts['number']} has no "
            f"title or body to use as a task."
        )
    return task
