"""
Tests for the make runner (modules/sandbox.py).

`make` was registered, but as bare `make` — which runs the Makefile's default
target. For most projects that is `all` or a build, so `run_tests("make")`
compiled the project and reported success while every test was skipped.

A green result nobody earned is worse than a red one, because nothing prompts
anyone to look.
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sandbox import (  # noqa: E402
    ALLOWED_RUNNERS,
    DOCKER_RUNNERS,
    RUNNERS,
    SubprocessSandbox,
    runners_for_language,
)

needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")

BOTH_TARGETS = 'all:\n\t@echo "BUILT"\n\ntest:\n\t@echo "TESTS RAN"\n'
BUILD_ONLY = 'all:\n\t@echo "BUILT"\n'
FAILING_TEST = 'test:\n\t@echo "1 failed"; exit 1\n'


@pytest.fixture
def project(tmp_path):
    def write(makefile):
        (tmp_path / "Makefile").write_text(makefile)
        return SubprocessSandbox(str(tmp_path), timeout_seconds=60)

    return write


# ── the registry entries ──────────────────────────────────────────────────


def test_make_runs_the_test_target():
    assert ALLOWED_RUNNERS["make"] == ["make", "test"]


def test_the_default_target_is_still_reachable():
    """
    Some projects genuinely test from the default target. Removing that option
    rather than renaming it would break them.
    """
    assert ALLOWED_RUNNERS["make_default"] == ["make"]


@pytest.mark.parametrize("runner", ["make", "make_default"])
def test_the_container_command_matches_the_host(runner):
    """Nothing about make differs inside an image."""
    assert DOCKER_RUNNERS[runner] == ALLOWED_RUNNERS[runner]


@pytest.mark.parametrize("runner", ["make", "make_default"])
def test_both_are_registered_under_make(runner):
    assert RUNNERS[runner].language == "make"


def test_both_appear_for_the_language():
    assert runners_for_language("make") == ["make", "make_default"]


# ── behaviour ─────────────────────────────────────────────────────────────


@needs_make
def test_the_test_target_is_what_runs(project):
    """The bug: this used to print BUILT."""
    result = project(BOTH_TARGETS).run_tests("make")

    assert "TESTS RAN" in result.stdout
    assert "BUILT" not in result.stdout


@needs_make
def test_a_passing_test_target_succeeds(project):
    assert project(BOTH_TARGETS).run_tests("make").success is True


@needs_make
def test_a_failing_test_target_fails(project):
    result = project(FAILING_TEST).run_tests("make")

    assert result.success is False
    assert "1 failed" in result.stdout


@needs_make
def test_a_missing_test_target_fails_loudly(project):
    """
    The case that matters most. Before, a Makefile with no test target built
    instead and reported success — the run went green having tested nothing.
    """
    result = project(BUILD_ONLY).run_tests("make")

    assert result.success is False
    assert "No rule to make target" in (result.stderr + result.stdout)


@needs_make
def test_the_default_target_still_works_when_asked_for(project):
    result = project(BOTH_TARGETS).run_tests("make_default")

    assert "BUILT" in result.stdout


@needs_make
def test_extra_arguments_reach_make(project):
    """`--runner-args` should still be able to name a different target."""
    result = project(BOTH_TARGETS).run_tests("make", ["all"])

    assert "BUILT" in result.stdout


# ── nothing else moved ────────────────────────────────────────────────────


def test_no_other_runner_changed():
    for runner in ("pytest", "python", "go", "cargo", "npm_test", "bash"):
        assert runner in ALLOWED_RUNNERS


def test_the_registry_grew_by_one():
    """
    14 before this change, after tox and nox landed in #153. Tied to
    ALLOWED_RUNNERS so the two figures cannot drift apart.
    """
    assert len(RUNNERS) == len(ALLOWED_RUNNERS) == 15
