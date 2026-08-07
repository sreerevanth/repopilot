"""
Tests for the pre-test lint step (modules/sandbox.py, modules/agent_loop.py).

A syntax error or an undefined name reaches the model as a precise location in
under a second, where the same mistake via pytest arrives as a collection error
buried in a traceback.

The hazard being guarded is a lint gate that fails on things the model did not
cause: it would then loop until max_iterations on every run.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.sandbox import ALLOWED_LINTERS, SubprocessSandbox  # noqa: E402

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"

TEST_FILE = "from mod import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def project(tmp_path, source):
    (tmp_path / "mod.py").write_text(source)
    (tmp_path / "test_mod.py").write_text(TEST_FILE)
    return SubprocessSandbox(str(tmp_path), timeout_seconds=120)


def ruff_available():
    sandbox = SubprocessSandbox(tempfile.mkdtemp())
    probe = sandbox.run([sys.executable, "-m", "ruff", "--version"])
    return probe.exit_code == 0


needs_ruff = pytest.mark.skipif(not ruff_available(), reason="ruff not installed")


# ── the table ─────────────────────────────────────────────────────────────


def test_expected_linters_are_registered():
    assert {"ruff", "flake8", "pyflakes", "eslint", "tsc"} <= set(ALLOWED_LINTERS)


@pytest.mark.parametrize("name", sorted(ALLOWED_LINTERS))
def test_every_linter_has_a_command(name):
    assert ALLOWED_LINTERS[name] and isinstance(ALLOWED_LINTERS[name], list)


@pytest.mark.parametrize("name", ["ruff", "flake8"])
def test_python_linters_select_errors_not_style(name):
    """
    Style rules fire on code the model did not write. ruff's default set flags
    unsorted imports in otherwise valid code, which would fail every iteration
    regardless of what the model produced.
    """
    joined = " ".join(ALLOWED_LINTERS[name])
    assert "E9" in joined and "F" in joined


def test_eslint_reports_errors_only():
    assert "--quiet" in ALLOWED_LINTERS["eslint"]


def test_js_linters_do_not_auto_install():
    """npx would otherwise fetch from the registry, breaking sandbox isolation."""
    for name in ("eslint", "tsc"):
        assert "--no-install" in ALLOWED_LINTERS[name]


def test_clippy_does_not_promote_warnings_to_errors():
    assert "-D" not in ALLOWED_LINTERS["clippy"]


# ── refusals ──────────────────────────────────────────────────────────────


def test_unknown_linter_is_refused(tmp_path):
    result = SubprocessSandbox(str(tmp_path)).run_lint("nonesuch")

    assert result.exit_code == -3
    assert result.success is False
    assert "nonesuch" in result.stderr
    assert "ruff" in result.stderr  # lists what is available


def test_missing_executable_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda exe: None)
    result = SubprocessSandbox(str(tmp_path)).run_lint("ruff")

    assert result.exit_code == -3
    assert "not found" in result.stderr


# ── real linting ──────────────────────────────────────────────────────────


@needs_ruff
def test_valid_code_passes_despite_style_issues(tmp_path):
    """The regression guard: unsorted imports must not fail the gate."""
    sandbox = project(tmp_path, "def add(a, b):\n    return a + b\n")
    assert sandbox.run_lint("ruff").success is True


@needs_ruff
@pytest.mark.parametrize(
    "source,marker",
    [
        ("def add(a, b):\n    return undefined_name + b\n", "F821"),
        ("def add(a, b)\n    return a + b\n", "syntax"),
        ("import os\n\ndef add(a, b):\n    return a + b\n", "F401"),
    ],
)
def test_real_errors_are_caught(tmp_path, source, marker):
    result = project(tmp_path, source).run_lint("ruff")

    assert result.success is False
    assert marker.lower() in (result.stdout + result.stderr).lower()


@needs_ruff
def test_the_message_reaches_the_model(tmp_path):
    """The loop feeds stdout/stderr back on the next iteration."""
    result = project(tmp_path, "def add(a, b):\n    return nope + b\n").run_lint("ruff")
    assert "nope" in result.stdout + result.stderr


@needs_ruff
def test_extra_args_are_forwarded(tmp_path):
    sandbox = project(tmp_path, "def add(a, b):\n    return a + b\n")
    assert "--statistics" in sandbox.run_lint("ruff", ["--statistics"]).command


# ── configuration and wiring ──────────────────────────────────────────────


def test_lint_is_off_by_default():
    config = AgentConfig(repo_root=".", task="t")
    assert config.lint_runner is None
    assert config.lint_args == []


def test_lint_args_are_not_shared_between_configs():
    """A mutable default would leak arguments across runs."""
    first = AgentConfig(repo_root=".", task="t")
    first.lint_args.append("--statistics")
    assert AgentConfig(repo_root=".", task="t").lint_args == []


def test_lint_runs_before_the_test_suite():
    source = AGENT_LOOP.read_text(encoding="utf-8")
    start = source.index("if cfg.lint_runner:")
    assert source.index("self._run_execution()", start) > start


def test_a_lint_failure_short_circuits_the_iteration():
    """Running a suite that cannot import is wasted time."""
    source = AGENT_LOOP.read_text(encoding="utf-8")
    start = source.index("if cfg.lint_runner:")
    block = source[start:start + 900]

    assert "if not lint_result.success:" in block
    assert "continue" in block
    assert "last_exec = lint_result" in block  # fed back to the model
