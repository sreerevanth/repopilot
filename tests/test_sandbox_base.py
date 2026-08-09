"""
Tests for the Sandbox base class (modules/sandbox.py).

`SubprocessSandbox` and `DockerSandbox` were unrelated classes that happened to
share some method names, and they had already drifted apart: `DockerSandbox` was
missing `run`, `run_file` and `run_lint` entirely, so `--lint` or `--run-file`
against it raised AttributeError rather than doing anything.

Nothing detected that, because nothing required the two to agree. An ABC makes
the divergence impossible rather than merely fixed.
"""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sandbox import (  # noqa: E402
    DockerSandbox,
    ExecutionResult,
    Sandbox,
    SubprocessSandbox,
)

BACKENDS = [SubprocessSandbox, DockerSandbox]
INTERFACE = ["run", "run_tests", "run_file", "run_lint", "run_pre_commit"]


# ── the contract ──────────────────────────────────────────────────────────


def test_the_base_is_abstract():
    assert inspect.isabstract(Sandbox)


def test_the_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Sandbox()


def test_a_backend_without_run_cannot_be_instantiated():
    """
    The guarantee that matters: a new backend cannot half-exist. Firecracker or
    a remote executor either implements run or fails at construction.
    """

    class Incomplete(Sandbox):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_a_backend_with_run_gets_the_whole_surface():
    class Minimal(Sandbox):
        working_dir = "."
        timeout = 5

        def run(self, command, env=None, stdin_data=None):
            return ExecutionResult(
                command=" ".join(command), exit_code=0, stdout="", stderr="",
                timed_out=False, duration_seconds=0.0,
            )

    backend = Minimal()

    for name in INTERFACE:
        assert hasattr(backend, name), name


# ── both backends satisfy it ──────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda c: c.__name__)
def test_every_backend_subclasses_the_base(backend):
    assert issubclass(backend, Sandbox)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", INTERFACE)
def test_every_backend_has_every_method(backend, method):
    """This is the assertion that was false before: Docker had three of four."""
    assert callable(getattr(backend, method, None))


def test_the_two_backends_expose_the_same_surface():
    def public(cls):
        return {
            name for name, _ in inspect.getmembers(cls, inspect.isfunction)
            if not name.startswith("_")
        }

    # cleanup is Docker-only: there is no container to remove in a subprocess.
    assert public(SubprocessSandbox) <= public(DockerSandbox)


# ── shared behaviour lives in one place ───────────────────────────────────


def test_run_is_the_only_thing_a_backend_must_implement():
    abstract = {
        name for name in Sandbox.__abstractmethods__
    }

    assert abstract == {"run"}


def test_docker_implements_run_itself():
    """Containerisation belongs to the backend, not to each caller."""
    assert "run" in DockerSandbox.__dict__


def test_docker_inherits_the_shared_methods():
    """
    They were previously absent. Inheriting rather than copying is what stops
    them going missing again.
    """
    for name in ("run_file", "run_lint", "run_pre_commit"):
        assert name not in DockerSandbox.__dict__
        assert hasattr(DockerSandbox, name)


def test_docker_overrides_run_tests():
    """
    DOCKER_RUNNERS names the interpreter inside the image, not the one running
    the agent, so this one genuinely differs.
    """
    assert "run_tests" in DockerSandbox.__dict__


# ── behaviour is unchanged ────────────────────────────────────────────────


def test_a_subprocess_command_still_runs(tmp_path):
    result = SubprocessSandbox(str(tmp_path), timeout_seconds=30).run(
        [sys.executable, "-c", "print('hello')"]
    )

    assert result.success is True
    assert "hello" in result.stdout


def test_an_unknown_runner_still_fails_cleanly(tmp_path):
    result = SubprocessSandbox(str(tmp_path), timeout_seconds=30).run_tests("nonesuch")

    assert result.exit_code == -3
