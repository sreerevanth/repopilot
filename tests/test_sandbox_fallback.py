"""
Tests for DockerSandbox fallback visibility (modules/sandbox.py).

DockerSandbox exists to provide `--network=none`, a memory cap and a CPU cap.
When it cannot, the caller has to be able to tell — at the time, via a warning
or a raise, and afterwards, from the recorded result.

No test here needs a real Docker daemon; availability is monkeypatched.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import sandbox as sandbox_mod  # noqa: E402
from modules.sandbox import (  # noqa: E402
    SANDBOX_DOCKER,
    SANDBOX_SUBPROCESS,
    SANDBOX_SUBPROCESS_FALLBACK,
    DockerSandbox,
    ExecutionResult,
    SandboxUnavailableError,
    SubprocessSandbox,
    _docker_is_usable,
)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    return tmp_path


@pytest.fixture
def no_docker(monkeypatch):
    monkeypatch.setattr(sandbox_mod, "_docker_is_usable", lambda *a, **k: False)


@pytest.fixture
def with_docker(monkeypatch):
    """Docker reports usable; the CLI call itself is stubbed out."""
    monkeypatch.setattr(sandbox_mod, "_docker_is_usable", lambda *a, **k: True)

    def fake_run(self, command, env=None, stdin_data=None):
        return ExecutionResult(
            command=" ".join(command), exit_code=0, stdout="", stderr="",
            timed_out=False, duration_seconds=0.0,
        )

    monkeypatch.setattr(SubprocessSandbox, "run", fake_run)


# ── provenance is recorded ────────────────────────────────────────────────


def test_subprocess_runs_are_labelled(project):
    result = SubprocessSandbox(str(project)).run_tests("pytest")
    assert result.sandbox == SANDBOX_SUBPROCESS
    assert result.isolated is False


def test_container_runs_are_labelled(project, with_docker):
    result = DockerSandbox(str(project)).run_tests("pytest")
    assert result.sandbox == SANDBOX_DOCKER
    assert result.isolated is True


def test_fallback_runs_are_labelled_distinctly(project, no_docker):
    """
    Distinct from plain "subprocess": the caller asked for isolation and did
    not get it, which is not the same as never having asked.
    """
    result = DockerSandbox(str(project)).run_tests("pytest")
    assert result.sandbox == SANDBOX_SUBPROCESS_FALLBACK
    assert result.isolated is False


def test_only_docker_counts_as_isolated():
    for kind in (SANDBOX_SUBPROCESS, SANDBOX_SUBPROCESS_FALLBACK):
        result = ExecutionResult("c", 0, "", "", False, 0.0, sandbox=kind)
        assert result.isolated is False


def test_sandbox_appears_in_summary(project, no_docker):
    result = DockerSandbox(str(project)).run_tests("pytest")
    assert f"sandbox={SANDBOX_SUBPROCESS_FALLBACK}" in result.summary()


def test_sandbox_field_defaults_for_positional_construction():
    """Existing callers that build ExecutionResult positionally still work."""
    result = ExecutionResult("cmd", 0, "out", "err", False, 1.0)
    assert result.sandbox == SANDBOX_SUBPROCESS


# ── the fallback is loud ──────────────────────────────────────────────────


def test_fallback_logs_a_warning(project, no_docker, caplog):
    with caplog.at_level("WARNING", logger="agent.sandbox"):
        DockerSandbox(str(project)).run_tests("pytest")

    assert any(r.levelname == "WARNING" for r in caplog.records)
    text = caplog.text
    assert "Docker is unavailable" in text
    # The warning must say what was lost, not just that something changed.
    assert "isolation" in text
    assert "NOT in effect" in text


def test_no_warning_when_docker_is_used(project, with_docker, caplog):
    with caplog.at_level("WARNING", logger="agent.sandbox"):
        DockerSandbox(str(project)).run_tests("pytest")
    assert caplog.records == []


def test_fallback_still_runs_the_tests(project, no_docker):
    """Degrading loudly must not mean degrading into a broken run."""
    result = DockerSandbox(str(project)).run_tests("pytest")
    assert result.success is True


# ── strict mode ───────────────────────────────────────────────────────────


def test_strict_raises_instead_of_degrading(project, no_docker):
    with pytest.raises(SandboxUnavailableError) as excinfo:
        DockerSandbox(str(project), strict=True).run_tests("pytest")
    assert "NOT in effect" in str(excinfo.value)


def test_strict_is_off_by_default(project, no_docker):
    result = DockerSandbox(str(project)).run_tests("pytest")
    assert result.sandbox == SANDBOX_SUBPROCESS_FALLBACK


def test_strict_does_not_raise_when_docker_works(project, with_docker):
    result = DockerSandbox(str(project), strict=True).run_tests("pytest")
    assert result.sandbox == SANDBOX_DOCKER


def test_error_type_is_catchable_as_runtime_error():
    assert issubclass(SandboxUnavailableError, RuntimeError)


# ── the availability probe ────────────────────────────────────────────────


def test_probe_false_when_cli_is_absent(monkeypatch):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: None)
    assert _docker_is_usable() is False


def test_probe_false_when_daemon_does_not_answer(monkeypatch):
    """
    The important case: Docker Desktop installed but not running leaves the
    binary on PATH. Before this change that counted as available, and the
    daemon error came back as a non-zero exit that looked like a test failure.
    """
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, returncode=1, stdout=b"", stderr=b"")

    monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)
    assert _docker_is_usable() is False


def test_probe_true_when_daemon_answers(monkeypatch):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(
            a, returncode=0, stdout=b"24.0.7", stderr=b""
        )

    monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)
    assert _docker_is_usable() is True


@pytest.mark.parametrize("exc", [OSError, subprocess.TimeoutExpired("docker", 1)])
def test_probe_survives_a_hanging_or_broken_cli(monkeypatch, exc):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")

    def fake_run(*a, **k):
        raise exc if isinstance(exc, Exception) else exc()

    monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)
    assert _docker_is_usable() is False
