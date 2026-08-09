"""
Tests for the runner registry (modules/sandbox.py).

Runner information lived in five parallel dicts keyed by the same strings:
ALLOWED_RUNNERS, DOCKER_RUNNERS, MODULE_RUNNERS, RUNNER_FALLBACKS and the linter
table. Adding a language meant remembering all of them, and DOCKER_RUNNERS was
ten of twelve an exact copy of ALLOWED_RUNNERS — it existed only because two
entries name the Python interpreter differently.

Those dicts are now derived from one registry. They are still exported, so
nothing importing them had to change, and they can no longer disagree.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sandbox import (  # noqa: E402
    ALLOWED_LINTERS,
    ALLOWED_RUNNERS,
    DOCKER_RUNNERS,
    MODULE_RUNNERS,
    RUNNER_FALLBACKS,
    RUNNERS,
    LanguageRunner,
    linters_for_runner,
    runners_for_language,
)


# ── the registry is the source ────────────────────────────────────────────


def test_every_entry_is_a_language_runner():
    for name, entry in RUNNERS.items():
        assert isinstance(entry, LanguageRunner), name


def test_the_key_matches_the_name():
    for name, entry in RUNNERS.items():
        assert entry.name == name


def test_every_runner_has_a_command():
    for name, entry in RUNNERS.items():
        assert entry.command, name


def test_every_runner_declares_a_language():
    for name, entry in RUNNERS.items():
        assert entry.language, name


def test_entries_are_immutable():
    """A shared registry that callers can mutate is a bug waiting to happen."""
    with pytest.raises(Exception):
        RUNNERS["pytest"].command = ["rm", "-rf", "/"]


# ── the derived views match what they replaced ────────────────────────────


def test_allowed_runners_covers_every_entry():
    assert set(ALLOWED_RUNNERS) == set(RUNNERS)


def test_docker_runners_covers_every_entry():
    """
    The old table was maintained by hand and could silently omit a runner that
    ALLOWED_RUNNERS had. Deriving it makes that impossible.
    """
    assert set(DOCKER_RUNNERS) == set(RUNNERS)


def test_docker_falls_back_to_the_host_command():
    """Ten of twelve runners are identical either side; only Python differs."""
    for name, entry in RUNNERS.items():
        if entry.container_command is None:
            assert DOCKER_RUNNERS[name] == ALLOWED_RUNNERS[name], name


@pytest.mark.parametrize("name", ["python", "pytest"])
def test_python_runners_differ_inside_a_container(name):
    """
    The host uses sys.executable; the image uses whatever `python` resolves to.
    This is the entire reason a separate Docker table existed.
    """
    assert DOCKER_RUNNERS[name] != ALLOWED_RUNNERS[name]
    assert DOCKER_RUNNERS[name][0] == "python"


def test_module_runners_are_derived():
    assert MODULE_RUNNERS == {"pytest": "pytest"}


def test_fallbacks_are_derived():
    assert RUNNER_FALLBACKS == {"pytest": "python"}


def test_a_fallback_points_at_a_real_runner():
    """A fallback naming a runner that does not exist would fail at use time."""
    for name, fallback in RUNNER_FALLBACKS.items():
        assert fallback in RUNNERS, f"{name} falls back to unknown {fallback}"


# ── the runners that were there before are still there ────────────────────


@pytest.mark.parametrize(
    "name",
    ["python", "pytest", "node", "npm_test", "vitest", "jest",
     "bash", "make", "go", "cargo", "ruby", "rspec"],
)
def test_no_runner_was_lost(name):
    assert name in ALLOWED_RUNNERS


def test_the_count_matches_the_registry():
    """
    Pinned so a runner cannot be dropped unnoticed. Raised from 12 to 14 when
    tox and nox were added for #153.
    """
    assert len(ALLOWED_RUNNERS) == len(RUNNERS) == 14


# ── what the registry makes possible ──────────────────────────────────────


def test_runners_can_be_listed_by_language():
    """Not answerable from the old tables, which had no notion of a language."""
    assert runners_for_language("python") == ["nox", "pytest", "python", "tox"]
    assert runners_for_language("javascript") == ["jest", "node", "npm_test", "vitest"]


def test_an_unknown_language_lists_nothing():
    assert runners_for_language("cobol") == []


def test_linters_are_associated_with_runners():
    assert "ruff" in linters_for_runner("pytest")
    assert "clippy" in linters_for_runner("cargo")


def test_an_unknown_runner_has_no_linters():
    assert linters_for_runner("nonesuch") == ()


def test_associated_linters_exist_in_the_linter_table():
    """An association naming a linter with no command would break at use time."""
    for name, entry in RUNNERS.items():
        for linter in entry.linters:
            assert linter in ALLOWED_LINTERS, f"{name} -> unknown linter {linter}"
