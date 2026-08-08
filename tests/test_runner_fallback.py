"""
Tests for the pytest -> python fallback (modules/sandbox.py).

Two problems, one of which the issue does not mention.

`_resolve_runner` checked `shutil.which(candidates[0])`. For pytest that
executable is `sys.executable`, which always exists, so the "runner not found"
guard never fired. A missing pytest instead surfaced as exit 1 with "No module
named pytest" — indistinguishable from a failing suite once wrapped in an
ExecutionResult, so the agent would read it as a test failure and start
rewriting code that was never broken.

The second is the fallback itself, and it is weaker than a suite run. These
tests pin that the result says so.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import sandbox as sb  # noqa: E402
from modules.sandbox import (  # noqa: E402
    MODULE_RUNNERS,
    RUNNER_FALLBACKS,
    SubprocessSandbox,
    _discover_test_files,
    _module_is_available,
    _resolve_runner,
)


@pytest.fixture
def no_pytest(monkeypatch):
    """Simulate pytest not being installed."""
    monkeypatch.setattr(sb, "_module_is_available", lambda module: False)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "test_ok.py").write_text(
        "def test_a():\n    assert 1 == 1\n\nassert 2 == 2\n"
    )
    (tmp_path / "helper.py").write_text("x = 1\n")
    return SubprocessSandbox(str(tmp_path), timeout_seconds=60)


# ── the detection gap ─────────────────────────────────────────────────────


def test_pytest_is_registered_as_a_module_runner():
    """
    Without this, availability is judged by `which(sys.executable)`, which is
    always true and so never rejects anything.
    """
    assert MODULE_RUNNERS["pytest"] == "pytest"


def test_a_present_module_is_detected():
    assert _module_is_available("pytest") is True


def test_an_absent_module_is_detected():
    assert _module_is_available("definitely_not_installed_anywhere") is False


def test_an_absent_module_makes_the_runner_unresolvable(no_pytest):
    assert _resolve_runner("pytest") is None


def test_a_present_module_resolves_normally():
    assert _resolve_runner("pytest") is not None


def test_non_module_runners_are_unaffected():
    """`go`, `cargo` and friends are still judged by which()."""
    assert "go" not in MODULE_RUNNERS
    assert "cargo" not in MODULE_RUNNERS


# ── discovery ─────────────────────────────────────────────────────────────


def test_conventional_test_files_are_found(tmp_path):
    (tmp_path / "test_a.py").write_text("")
    (tmp_path / "b_test.py").write_text("")
    (tmp_path / "notes.py").write_text("")

    found = _discover_test_files(str(tmp_path))

    assert set(found) == {"test_a.py", "b_test.py"}


def test_non_test_files_are_not_run(tmp_path):
    """
    `python <file>` executes module scope, so running arbitrary files could
    execute anything. Only conventional test names are picked up.
    """
    (tmp_path / "deploy.py").write_text("raise SystemExit('should never run')")

    assert _discover_test_files(str(tmp_path)) == []


def test_vendor_directories_are_skipped(tmp_path):
    for directory in ("node_modules", "__pycache__", ".venv"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "test_vendor.py").write_text("")

    assert _discover_test_files(str(tmp_path)) == []


def test_nested_test_files_are_found(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_deep.py").write_text("")

    assert _discover_test_files(str(tmp_path)) == ["tests/test_deep.py"]


def test_paths_use_forward_slashes(tmp_path):
    """
    DockerSandbox mounts the repo into a Linux container, so a path discovered
    on Windows as "tests\\test_x.py" would not resolve inside it. Matches the
    convention already used by repo_ingestion and secret_scanner.
    """
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "test_nested.py").write_text("")

    found = _discover_test_files(str(tmp_path))

    assert found == ["a/b/test_nested.py"]
    assert all("\\" not in path for path in found)


def test_the_file_count_is_capped(tmp_path):
    """One `python` invocation per file; an unbounded repo would hang the run."""
    for i in range(60):
        (tmp_path / f"test_{i:03d}.py").write_text("")

    assert len(_discover_test_files(str(tmp_path))) <= sb.MAX_FALLBACK_FILES


# ── the fallback ──────────────────────────────────────────────────────────


def test_the_fallback_is_only_defined_where_it_makes_sense():
    """There is no way to run a Go or Rust suite with a Python interpreter."""
    assert RUNNER_FALLBACKS == {"pytest": "python"}


def test_files_are_run_when_pytest_is_missing(no_pytest, project):
    result = project.run_tests("pytest")

    assert "test file(s)" in result.command
    assert result.stdout.count("--- test_") == 1


def test_a_passing_file_yields_a_passing_result(no_pytest, project):
    assert project.run_tests("pytest").success is True


def test_a_failing_file_yields_a_failing_result(no_pytest, tmp_path):
    (tmp_path / "test_bad.py").write_text("assert 1 == 2, 'boom'\n")
    result = SubprocessSandbox(str(tmp_path), timeout_seconds=60).run_tests("pytest")

    assert result.success is False
    assert "boom" in result.stdout + result.stderr


def test_the_result_says_it_fell_back(no_pytest, project):
    assert "[fallback]" in project.run_tests("pytest").stdout


def test_the_result_states_the_limitation(no_pytest, project):
    """
    `python <file>` runs module scope. Assertions inside pytest-collected
    functions never execute, so a pass here is much weaker than a suite pass
    and the output must not let anyone assume otherwise.
    """
    assert "not that a test suite succeeded" in project.run_tests("pytest").stdout


def test_each_file_is_labelled_in_the_output(no_pytest, tmp_path):
    (tmp_path / "test_one.py").write_text("")
    (tmp_path / "test_two.py").write_text("")
    sandbox = SubprocessSandbox(str(tmp_path), timeout_seconds=60)
    stdout = sandbox.run_tests("pytest").stdout

    assert "--- test_one.py" in stdout
    assert "--- test_two.py" in stdout


def test_no_test_files_means_no_fallback(no_pytest, tmp_path):
    """Reporting success on an empty run would be worse than reporting nothing."""
    result = SubprocessSandbox(str(tmp_path), timeout_seconds=30).run_tests("pytest")

    assert result.exit_code == -3
    assert "not possible either" in result.stderr


def test_a_runner_without_a_fallback_still_fails_cleanly(tmp_path):
    result = SubprocessSandbox(str(tmp_path), timeout_seconds=30).run_tests("nonesuch")

    assert result.exit_code == -3
    assert "not found or not allowed" in result.stderr


def test_normal_operation_is_unchanged(project):
    """With pytest installed, nothing about the existing path changes."""
    assert "-m" in project.run_tests("pytest").command
