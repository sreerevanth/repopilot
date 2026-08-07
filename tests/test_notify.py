"""
Tests for run-completion notifications (modules/notify.py).

The governing property is that nothing here can fail a run. A broken webhook
URL, an unreachable host or a rejected request must be reported through the
return value and the log, never by raising into the caller — the agent work is
already done by the time this fires, and losing it to a notification bug would
be absurd.

No test contacts the network; urlopen is replaced throughout.
"""

import json
import socket
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import notify  # noqa: E402
from modules.notify import (  # noqa: E402
    ALLOWED_SCHEMES,
    WEBHOOK_ENV_VAR,
    detect_format,
    format_message,
    notify_run_complete,
    send_webhook,
)

SLACK = "https://hooks.slack.com/services/T000/B000/xxxx"
DISCORD = "https://discord.com/api/webhooks/123/abcdef"
GENERIC = "https://ops.internal.example/hooks/agent"

FIELDS = dict(
    outcome="success",
    task="Fix the parser TypeError",
    run_id="agent_20260807_abc",
    branch_name="agent/fix-abc",
    pr_url="https://github.com/o/r/pull/9",
    final_message="Task completed successfully. Tests pass.",
    iterations_used=3,
)


class _Response:
    """Minimal stand-in for the object urlopen returns (a context manager)."""

    def __init__(self, status):
        self.status = status

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Capture:
    """Records the request instead of sending it."""

    def __init__(self, status=200, raises=None):
        self.status = status
        self.raises = raises
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout=None):
        if self.raises is not None:
            raise self.raises
        self.request = request
        self.timeout = timeout
        return _Response(self.status)

    @property
    def payload(self):
        return json.loads(self.request.data.decode("utf-8"))


@pytest.fixture
def capture(monkeypatch):
    cap = Capture()
    monkeypatch.setattr(notify.urllib.request, "urlopen", cap)
    return cap


@pytest.fixture(autouse=True)
def no_ambient_webhook(monkeypatch):
    """A real WEBHOOK_URL in the environment must not leak into tests."""
    monkeypatch.delenv(WEBHOOK_ENV_VAR, raising=False)


# ── routing ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,expected",
    [
        (SLACK, "slack"),
        ("https://slack.com/api/hooks/x", "slack"),
        (DISCORD, "discord"),
        ("https://discordapp.com/api/webhooks/1/x", "discord"),
        (GENERIC, "generic"),
        ("http://localhost:9000/hook", "generic"),
    ],
)
def test_service_is_detected_from_the_url(url, expected):
    assert detect_format(url) == expected


def test_slack_gets_a_text_key(capture):
    send_webhook(url=SLACK, **FIELDS)
    assert set(capture.payload) == {"text"}


def test_discord_gets_a_content_key(capture):
    send_webhook(url=DISCORD, **FIELDS)
    assert set(capture.payload) == {"content"}


def test_generic_endpoints_get_structured_fields(capture):
    """Something parsing this wants data, not prose."""
    send_webhook(url=GENERIC, **FIELDS)
    payload = capture.payload

    assert payload["outcome"] == "success"
    assert payload["run_id"] == FIELDS["run_id"]
    assert payload["pr_url"] == FIELDS["pr_url"]
    assert payload["iterations_used"] == 3
    assert "text" in payload  # rendered form as well


# ── the message ───────────────────────────────────────────────────────────


def test_message_leads_with_status_and_task():
    text = format_message(**FIELDS)
    assert text.splitlines()[0] == "RepoPilot PASSED: Fix the parser TypeError"


@pytest.mark.parametrize(
    "outcome,marker",
    [
        ("success", "PASSED"),
        ("failed", "FAILED"),
        ("max_retries", "FAILED"),
        ("error", "ERROR"),
        ("aborted", "ABORTED"),
        ("budget_exceeded", "STOPPED"),
    ],
)
def test_each_outcome_has_a_readable_marker(outcome, marker):
    assert marker in format_message(outcome=outcome, task="t")


def test_unknown_outcome_still_renders():
    assert "SOMETHING_NEW" in format_message(outcome="something_new", task="t")


def test_optional_fields_are_omitted_when_absent():
    text = format_message(outcome="failed", task="t")
    assert "Branch:" not in text
    assert "PR:" not in text
    assert "Iterations:" not in text


def test_pr_url_is_included_when_present():
    assert FIELDS["pr_url"] in format_message(**FIELDS)


# ── request mechanics ─────────────────────────────────────────────────────


def test_it_posts_json(capture):
    send_webhook(url=GENERIC, **FIELDS)
    assert capture.request.method == "POST"
    assert capture.request.headers["Content-type"] == "application/json"


def test_a_timeout_is_always_applied(capture):
    """An unreachable host must not hang the process after the work is done."""
    send_webhook(url=GENERIC, timeout=7, **FIELDS)
    assert capture.timeout == 7


def test_a_2xx_is_success(capture):
    assert send_webhook(url=GENERIC, **FIELDS) is True


@pytest.mark.parametrize("status", [301, 400, 500])
def test_a_non_2xx_is_not_success(monkeypatch, status):
    monkeypatch.setattr(notify.urllib.request, "urlopen", Capture(status=status))
    assert send_webhook(url=GENERIC, **FIELDS) is False


# ── it must never fail a run ──────────────────────────────────────────────


def test_no_url_is_a_silent_no_op(capture):
    assert send_webhook(**FIELDS) is False
    assert capture.request is None


def test_env_var_is_used_when_no_url_is_passed(capture, monkeypatch):
    monkeypatch.setenv(WEBHOOK_ENV_VAR, GENERIC)
    assert send_webhook(**FIELDS) is True


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://h/x", "hooks.slack.com/x"]
)
def test_non_http_schemes_are_refused(capture, url):
    """The URL comes from the environment; a scheme mistake should be a no-op."""
    assert send_webhook(url=url, **FIELDS) is False
    assert capture.request is None


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("u", 404, "not found", {}, None),
        socket.gaierror("name resolution failed"),
        TimeoutError("timed out"),
        OSError("certificate verify failed"),
        RuntimeError("something unforeseen"),
    ],
)
def test_transport_failures_return_false_rather_than_raising(monkeypatch, error):
    monkeypatch.setattr(notify.urllib.request, "urlopen", Capture(raises=error))
    assert send_webhook(url=GENERIC, **FIELDS) is False


def test_unserialisable_fields_do_not_raise(capture):
    class Weird:
        pass

    assert send_webhook(url=GENERIC, outcome="success", task=Weird()) is False


# ── the caller-facing entry point ─────────────────────────────────────────


def make_result(**overrides):
    base = dict(
        run_id="agent_20260807_abc", outcome="success",
        branch_name="agent/fix-abc", pr_url=None,
        iterations_used=2, final_message="done",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_notify_run_complete_sends_the_result(capture, monkeypatch):
    monkeypatch.setenv(WEBHOOK_ENV_VAR, GENERIC)
    assert notify_run_complete(make_result(), task="Fix it") is True

    payload = capture.payload
    assert payload["outcome"] == "success"
    assert payload["task"] == "Fix it"
    assert payload["iterations_used"] == 2


def test_notify_run_complete_is_a_no_op_without_the_env_var(capture):
    assert notify_run_complete(make_result(), task="Fix it") is False
    assert capture.request is None


def test_notify_run_complete_survives_a_malformed_result(capture, monkeypatch):
    """It is called on every run; a missing attribute must not crash main."""
    monkeypatch.setenv(WEBHOOK_ENV_VAR, GENERIC)
    assert notify_run_complete(SimpleNamespace(), task="t") is False


def test_failed_runs_are_notified_too(capture, monkeypatch):
    """Walking away matters most when the run does not succeed."""
    monkeypatch.setenv(WEBHOOK_ENV_VAR, SLACK)
    notify_run_complete(make_result(outcome="max_retries"), task="Fix it")

    assert "FAILED" in capture.payload["text"]


def test_only_http_schemes_are_allowed():
    assert set(ALLOWED_SCHEMES) == {"http", "https"}
