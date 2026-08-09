"""
Tests for tox and nox as runners (modules/sandbox.py).

Both manage their own interpreters and virtualenvs, so the sandbox invokes them
and stays out of the way. Adding them is two registry entries now that #140
folded the five parallel runner tables into one.

The part worth being careful about is what "simultaneously against 3.9, 3.10 and
3.11" means inside a container. The default image ships one interpreter, so tox
skips every environment it cannot find a Python for and exits successfully
having tested one version. That is a worse outcome than failing, because it
reads as a stronger result than it is.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sandbox import (  # noqa: E402
    ALLOWED_RUNNERS,
    DOCKER_RUNNERS,
    MULTI_VERSION_IMAGE_NOTE,
    MULTI_VERSION_RUNNERS,
    RUNNERS,
    SubprocessSandbox,
    linters_for_runner,
    runners_for_language,
)


# ── they are registered ───────────────────────────────────────────────────


@pytest.mark.parametrize("runner", ["tox", "nox"])
def test_the_runner_exists(runner):
    assert runner in RUNNERS


@pytest.mark.parametrize("runner", ["tox", "nox"])
def test_the_command_is_the_bare_tool(runner):
    """No arguments: both read their own config and decide what to run."""
    assert ALLOWED_RUNNERS[runner] == [runner]


@pytest.mark.parametrize("runner", ["tox", "nox"])
def test_the_container_command_matches_the_host(runner):
    """
    Unlike pytest, neither is invoked through a specific interpreter, so there
    is nothing to differ between host and image.
    """
    assert DOCKER_RUNNERS[runner] == ALLOWED_RUNNERS[runner]


@pytest.mark.parametrize("runner", ["tox", "nox"])
def test_they_are_python_runners(runner):
    assert RUNNERS[runner].language == "python"


def test_they_appear_alongside_the_other_python_runners():
    """The registry from #140 is what makes this discoverable rather than a lookup."""
    assert runners_for_language("python") == ["nox", "pytest", "python", "tox"]


@pytest.mark.parametrize("runner", ["tox", "nox"])
def test_python_linters_are_associated(runner):
    assert "ruff" in linters_for_runner(runner)


# ── they behave like every other runner ───────────────────────────────────


@pytest.mark.parametrize("runner", ["tox", "nox"])
def test_a_missing_tool_fails_cleanly(runner, monkeypatch):
    """
    Neither is a module invoked through an interpreter, so availability is the
    ordinary `which` check rather than the import probe pytest needs.
    """
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = SubprocessSandbox(tempfile.mkdtemp()).run_tests(runner)

    assert result.exit_code == -3
    assert "not found" in result.stderr


@pytest.mark.parametrize("runner", ["tox", "nox"])
def test_neither_needs_an_import_probe(runner):
    """
    MODULE_RUNNERS exists because `which(sys.executable)` always succeeds for
    `python -m pytest`. tox and nox are real executables on PATH.
    """
    assert RUNNERS[runner].module is None


@pytest.mark.parametrize("runner", ["tox", "nox"])
def test_neither_falls_back(runner):
    """
    pytest falls back to running files with `python` because that still
    executes them. There is no weaker way to run a tox matrix.
    """
    assert RUNNERS[runner].fallback is None


# ── the container caveat is stated ────────────────────────────────────────


def test_both_are_marked_as_multi_version():
    assert MULTI_VERSION_RUNNERS == {"tox", "nox"}


def test_the_note_names_the_actual_limitation():
    assert "image" in MULTI_VERSION_IMAGE_NOTE
    assert "python:3.11-slim" in MULTI_VERSION_IMAGE_NOTE


def test_the_note_says_what_to_do_about_it():
    """A caveat without a remedy is just discouragement."""
    assert "multi-python" in MULTI_VERSION_IMAGE_NOTE


def test_docker_warns_rather_than_refusing():
    """
    Testing one version is still worth something, and the image may well be a
    multi-python one. Silence is the problem, not the single-version run.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "sandbox.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("if runner in MULTI_VERSION_RUNNERS:"):][:400]

    assert "_LOG.warning" in block
    assert "raise" not in block


# ── nothing else changed ──────────────────────────────────────────────────


def test_the_existing_runners_are_untouched():
    for runner in ("pytest", "python", "go", "cargo", "npm_test", "make"):
        assert runner in ALLOWED_RUNNERS


def test_the_registry_grew_by_exactly_two():
    assert len(RUNNERS) == 14
