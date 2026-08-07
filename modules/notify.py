"""
Module: Run-completion notifications.

A six-iteration run takes ten minutes or more, so people walk away from the
terminal. Setting WEBHOOK_URL posts the outcome to Slack, Discord, or any
endpoint that accepts a JSON POST.

Nothing here may fail a run. A notification is a courtesy; a broken webhook URL
or an unreachable host must not turn a successful agent run into a failed one,
so every entry point returns a bool and swallows its own errors.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

_LOG = logging.getLogger("agent.notify")

WEBHOOK_ENV_VAR = "WEBHOOK_URL"
DEFAULT_TIMEOUT = 10

# Only these carry the run over a network. file:// and friends are rejected --
# the URL comes from the environment, and a scheme mistake should be a clear
# no-op rather than something surprising.
ALLOWED_SCHEMES = ("http", "https")

STATUS_MARKERS = {
    "success": "PASSED",
    "failed": "FAILED",
    "max_retries": "FAILED",
    "error": "ERROR",
    "aborted": "ABORTED",
    "budget_exceeded": "STOPPED",
    "dry_run": "DRY RUN",
}


def detect_format(url: str) -> str:
    """Slack and Discord each want their own key; anything else gets the lot."""
    host = urllib.parse.urlparse(url).netloc.lower()
    if "hooks.slack.com" in host or "slack.com" in host:
        return "slack"
    if "discord.com" in host or "discordapp.com" in host:
        return "discord"
    return "generic"


def format_message(
    outcome: str,
    task: str,
    run_id: str = "",
    branch_name: Optional[str] = None,
    pr_url: Optional[str] = None,
    final_message: str = "",
    iterations_used: int = 0,
) -> str:
    """A single human-readable line-set for chat clients."""
    marker = STATUS_MARKERS.get(outcome, outcome.upper())
    lines = [f"RepoPilot {marker}: {task}"]
    if iterations_used:
        lines.append(f"Iterations: {iterations_used}")
    if branch_name:
        lines.append(f"Branch: {branch_name}")
    if pr_url:
        lines.append(f"PR: {pr_url}")
    if final_message:
        lines.append(final_message)
    if run_id:
        lines.append(f"Run: {run_id}")
    return "\n".join(lines)


def build_payload(url: str, **fields) -> dict:
    """
    Shape the payload for whichever service the URL points at.

    A generic endpoint gets the structured fields rather than a rendered
    string, since something parsing this wants data, not prose.
    """
    style = detect_format(url)
    text = format_message(**fields)

    if style == "slack":
        return {"text": text}
    if style == "discord":
        return {"content": text}

    payload = dict(fields)
    payload["text"] = text
    return payload


def send_webhook(
    url: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    **fields,
) -> bool:
    """
    POST the run outcome. Returns True only if the request was accepted.

    Never raises: a notification failure is reported through the log and the
    return value, never by interrupting the caller.
    """
    url = url or os.environ.get(WEBHOOK_ENV_VAR)
    if not url:
        return False

    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        _LOG.warning(
            "%s has scheme '%s'; only %s are sent. Skipping notification.",
            WEBHOOK_ENV_VAR, scheme or "(none)", "/".join(ALLOWED_SCHEMES),
        )
        return False

    try:
        body = json.dumps(build_payload(url, **fields)).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _LOG.warning("could not serialise notification payload: %s", exc)
        return False

    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "repopilot-agent",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None) or response.getcode()
            if 200 <= status < 300:
                _LOG.debug("notification delivered (%s)", status)
                return True
            _LOG.warning("webhook returned HTTP %s", status)
            return False
    except urllib.error.HTTPError as exc:
        _LOG.warning("webhook rejected the notification: HTTP %s", exc.code)
    except Exception as exc:
        # Deliberately broad: DNS, TLS, timeouts, proxies. The run already
        # finished, and none of it is worth surfacing as a failure.
        _LOG.warning("could not deliver notification: %s", exc)
    return False


def notify_run_complete(result, task: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Send an AgentRunResult, if WEBHOOK_URL is set. Never raises."""
    try:
        return send_webhook(
            outcome=result.outcome,
            task=task,
            run_id=result.run_id,
            branch_name=result.branch_name,
            pr_url=result.pr_url,
            final_message=result.final_message,
            iterations_used=result.iterations_used,
        )
    except Exception as exc:  # pragma: no cover - belt and braces
        _LOG.warning("notification failed: %s", exc)
        return False
