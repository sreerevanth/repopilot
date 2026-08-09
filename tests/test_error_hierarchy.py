"""
Tests for the exception hierarchy (modules/errors.py).

Seven exception classes existed across six modules, each inheriting directly
from `RuntimeError` or `ValueError` and sharing nothing. `main.py` caught bare
`Exception` and printed a traceback for all of them — so "the Docker daemon is
not running" and "there is a bug in the agent loop" were reported identically.

`AgentError` separates the two. The safety property is that every exception now
has *two* bases: it is an `AgentError` **and** still whatever it was before, so
every existing `except ValueError` and `except RuntimeError` behaves unchanged.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.doc_lookup import LookupRefused  # noqa: E402
from modules.errors import (  # noqa: E402
    AgentError,
    ConfigurationError,
    ExecutionError,
    ProviderError,
    StateError,
)
from modules.llm_client import BudgetExceededError, SystemPromptError  # noqa: E402
from modules.parallel_tasks import WorktreeError  # noqa: E402
from modules.run_state import ResumeError  # noqa: E402
from modules.sandbox import SandboxUnavailableError  # noqa: E402
from modules.task_source import TaskResolutionError  # noqa: E402

# Every exception, with the base it had before this change.
EXISTING = [
    (LookupRefused, ValueError),
    (SystemPromptError, ValueError),
    (BudgetExceededError, RuntimeError),
    (WorktreeError, RuntimeError),
    (ResumeError, RuntimeError),
    (SandboxUnavailableError, RuntimeError),
    (TaskResolutionError, RuntimeError),
]


# ── nothing that caught them before has stopped ───────────────────────────


@pytest.mark.parametrize(
    "exception,original", EXISTING, ids=lambda x: getattr(x, "__name__", "")
)
def test_the_original_base_is_kept(exception, original):
    """
    The property that makes this safe to land at once. `code_modifier.py`
    catches ValueError in four places and `agent_loop.py` catches RuntimeError;
    replacing the base rather than adding one would silently stop those working.
    """
    assert issubclass(exception, original)


def test_catching_by_the_original_base_still_works():
    with pytest.raises(ValueError):
        raise LookupRefused("refused")

    with pytest.raises(RuntimeError):
        raise BudgetExceededError("limit reached")


# ── and the new base catches everything ───────────────────────────────────


@pytest.mark.parametrize(
    "exception,_", EXISTING, ids=lambda x: getattr(x, "__name__", "")
)
def test_every_exception_is_an_agent_error(exception, _):
    assert issubclass(exception, AgentError)


def test_one_handler_catches_them_all():
    caught = 0
    for exception, _ in EXISTING:
        try:
            raise exception("x")
        except AgentError:
            caught += 1

    assert caught == len(EXISTING)


def test_an_unexpected_error_is_not_caught():
    """
    The whole point. A TypeError means this tool has a bug, and swallowing it
    as though it were a user-facing condition is how a bug becomes a mystery.
    """
    with pytest.raises(TypeError):
        try:
            raise TypeError("a real bug")
        except AgentError:
            pytest.fail("AgentError should not catch a TypeError")


# ── the categories ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exception,category",
    [
        (LookupRefused, ConfigurationError),
        (SystemPromptError, ConfigurationError),
        (TaskResolutionError, ConfigurationError),
        (BudgetExceededError, ProviderError),
        (WorktreeError, ExecutionError),
        (SandboxUnavailableError, ExecutionError),
        (ResumeError, StateError),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_each_exception_has_a_category(exception, category):
    assert issubclass(exception, category)


@pytest.mark.parametrize(
    "category", [ConfigurationError, ProviderError, ExecutionError, StateError]
)
def test_every_category_is_an_agent_error(category):
    assert issubclass(category, AgentError)


def test_the_categories_are_distinct():
    """A ConfigurationError is not an ExecutionError; catching one is a choice."""
    assert not issubclass(ConfigurationError, ExecutionError)
    assert not issubclass(ProviderError, StateError)


# ── the message ───────────────────────────────────────────────────────────


def test_a_plain_error_reports_its_message():
    assert SandboxUnavailableError("Docker is unavailable.").user_message() == \
        "Docker is unavailable."


def test_a_remedy_is_appended_when_there_is_one():
    class WithRemedy(ConfigurationError):
        remedy = "Set ANTHROPIC_API_KEY, or pass --api-key."

    assert WithRemedy("No API key found.").user_message() == \
        "No API key found. Set ANTHROPIC_API_KEY, or pass --api-key."


def test_no_remedy_leaves_no_trailing_space():
    assert AgentError("Something went wrong.").user_message() == "Something went wrong."


# ── how main.py uses it ───────────────────────────────────────────────────


def test_an_agent_error_is_handled_before_the_catch_all():
    """Order matters: after `except Exception` it would never be reached."""
    source = (ROOT / "main.py").read_text(encoding="utf-8")

    assert source.index("except AgentError as e:") < source.index("except Exception as e:\n        # Anything else")


def test_an_agent_error_gets_no_traceback():
    """
    A traceback into code the user did not write is noise when the problem is
    a missing runner or an unset key.
    """
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    block = source[source.index("def _report_expected_error"):source.index("def main():")]
    code = "\n".join(
        line for line in block.splitlines() if not line.strip().startswith("#")
    )

    assert "traceback" not in code
    assert "user_message()" in code


def test_an_unexpected_error_still_gets_a_traceback():
    """That is the part worth putting in a bug report."""
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    block = source[source.index("except Exception as e:\n        # Anything else"):]

    assert "traceback.print_exc()" in block


def test_both_handlers_exit_the_same_way():
    """A run that failed must not exit 0 because it failed in a tidier way."""
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    reporter = source[
        source.index("def _report_expected_error"):source.index("def main():")
    ]

    assert "sys.exit(1)" in reporter
    assert source.count("sys.exit(1)") >= 2


def test_every_expected_failure_reports_the_same_way():
    """
    Raised in review of #260: task resolution caught TaskResolutionError
    separately, printed a bare message and wrote no GITHUB_OUTPUT -- so a CI
    run saw no outcome=error at all for a bad issue URL.
    """
    source = (ROOT / "main.py").read_text(encoding="utf-8")

    assert "_report_expected_error(exc)" in source
    assert source.count("_report_expected_error(") >= 3


def test_the_reporter_writes_github_output():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    reporter = source[
        source.index("def _report_expected_error"):source.index("def main():")
    ]

    assert "write_github_output" in reporter
    assert '"outcome": "error"' in reporter
