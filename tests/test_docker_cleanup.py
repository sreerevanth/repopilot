"""
Tests for DockerSandbox container cleanup (modules/sandbox.py).

`--rm` is honoured by the daemon when a container *exits*. On timeout,
`subprocess.run` kills the docker CLI with SIGKILL — uncatchable, so it is never
proxied to the container, which keeps running and never exits, so `--rm` never
fires. These pin the paths that clean up after that.

No daemon required: docker invocations are recorded through a stub.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import sandbox as sandbox_mod  # noqa: E402
from modules.sandbox import (  # noqa: E402
    CONTAINER_LABEL_KEY,
    CONTAINER_NAME_PREFIX,
    DockerSandbox,
    ExecutionResult,
    _force_remove_container,
    _new_container_name,
)


@pytest.fixture
def docker_calls(monkeypatch):
    """Record every docker subprocess invocation; report success by default."""
    calls: list[list[str]] = []

    def fake_run(argv, *a, **k):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")
    # which() alone is not enough: _docker_is_usable also probes the daemon with
    # `docker version`. Without this the sandbox decides Docker is unavailable
    # and falls back to a host command, so no container flags appear.
    monkeypatch.setattr(sandbox_mod, "_docker_is_usable", lambda *a, **k: True)
    monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_mod, "_docker_is_usable", lambda *a, **k: True)
    sb = DockerSandbox(str(tmp_path), timeout_seconds=5)
    yield sb
    sandbox_mod._ACTIVE_CONTAINERS.clear()


def stub_result(**kwargs):
    base = dict(
        command="docker run", exit_code=0, stdout="", stderr="",
        timed_out=False, duration_seconds=0.1,
    )
    base.update(kwargs)
    return ExecutionResult(**base)


def patch_execution(monkeypatch, result=None, raises=None):
    """Replace SubprocessSandbox.run, capturing the docker argv it was given."""
    seen: dict = {}

    def fake_run(self, command, env=None, stdin_data=None):
        seen["argv"] = list(command)
        if raises is not None:
            raise raises
        return result if result is not None else stub_result()

    monkeypatch.setattr(sandbox_mod.SubprocessSandbox, "run", fake_run)
    return seen


def removals(calls):
    return [c for c in calls if c[:3] == ["docker", "rm", "--force"]]


# ── the container is findable ─────────────────────────────────────────────


def test_container_is_named(sandbox, monkeypatch, docker_calls):
    seen = patch_execution(monkeypatch)
    sandbox.run_tests("pytest")

    argv = seen["argv"]
    assert "--name" in argv
    assert argv[argv.index("--name") + 1].startswith(CONTAINER_NAME_PREFIX)


def test_container_is_labelled(sandbox, monkeypatch, docker_calls):
    """The label is what makes a sweep possible after a SIGKILL."""
    seen = patch_execution(monkeypatch)
    sandbox.run_tests("pytest")
    assert f"{CONTAINER_LABEL_KEY}=1" in seen["argv"]


def test_names_are_unique_per_run():
    """A reused name would collide with a container that outlived its run."""
    assert len({_new_container_name() for _ in range(200)}) == 200


def test_rm_flag_is_kept(sandbox, monkeypatch, docker_calls):
    """Explicit removal complements --rm; it does not replace it."""
    seen = patch_execution(monkeypatch)
    sandbox.run_tests("pytest")
    assert "--rm" in seen["argv"]


# ── timeout: the case --rm cannot cover ───────────────────────────────────


def test_timeout_forces_removal(sandbox, monkeypatch, docker_calls):
    patch_execution(monkeypatch, result=stub_result(timed_out=True, exit_code=-1))
    sandbox.run_tests("pytest")
    assert len(removals(docker_calls)) == 1


def test_timeout_removes_the_container_that_was_started(
    sandbox, monkeypatch, docker_calls
):
    seen = patch_execution(
        monkeypatch, result=stub_result(timed_out=True, exit_code=-1)
    )
    sandbox.run_tests("pytest")

    started = seen["argv"][seen["argv"].index("--name") + 1]
    assert removals(docker_calls)[0][3] == started


def test_timeout_still_returns_the_result(sandbox, monkeypatch, docker_calls):
    """Cleanup must not swallow the outcome the caller is waiting for."""
    patch_execution(monkeypatch, result=stub_result(timed_out=True, exit_code=-1))
    result = sandbox.run_tests("pytest")
    assert result.timed_out is True
    assert result.exit_code == -1


def test_success_does_not_issue_a_redundant_removal(sandbox, monkeypatch, docker_calls):
    """--rm already handled it; a second daemon round-trip is wasted work."""
    patch_execution(monkeypatch)
    sandbox.run_tests("pytest")
    assert removals(docker_calls) == []


# ── interrupts ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit()])
def test_interrupt_removes_the_container(sandbox, monkeypatch, docker_calls, exc):
    """Ctrl+C raises BaseException, which `except Exception` would miss."""
    patch_execution(monkeypatch, raises=exc)

    with pytest.raises(type(exc)):
        sandbox.run_tests("pytest")

    assert len(removals(docker_calls)) == 1


def test_interrupt_is_not_swallowed(sandbox, monkeypatch, docker_calls):
    patch_execution(monkeypatch, raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        sandbox.run_tests("pytest")


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit()])
def test_registry_is_left_clean_after_an_interrupt(
    sandbox, monkeypatch, docker_calls, exc
):
    patch_execution(monkeypatch, raises=exc)
    with pytest.raises(type(exc)):
        sandbox.run_tests("pytest")
    assert sandbox_mod._ACTIVE_CONTAINERS == set()


def test_registry_is_left_clean_after_success(sandbox, monkeypatch, docker_calls):
    patch_execution(monkeypatch)
    sandbox.run_tests("pytest")
    assert sandbox_mod._ACTIVE_CONTAINERS == set()


# ── context manager ───────────────────────────────────────────────────────


def test_context_manager_returns_the_sandbox(sandbox):
    with sandbox as entered:
        assert entered is sandbox


def test_context_manager_does_not_suppress_exceptions(sandbox, docker_calls):
    with pytest.raises(ValueError):
        with sandbox:
            raise ValueError("boom")


def test_cleanup_removes_tracked_containers(sandbox, docker_calls):
    sandbox._containers.add("repopilot-sandbox-abc123")
    sandbox.cleanup()

    assert removals(docker_calls) == [
        ["docker", "rm", "--force", "repopilot-sandbox-abc123"]
    ]
    assert sandbox._containers == set()


def test_cleanup_is_idempotent(sandbox, docker_calls):
    sandbox._containers.add("repopilot-sandbox-abc123")
    sandbox.cleanup()
    sandbox.cleanup()
    assert len(removals(docker_calls)) == 1


# ── removal is tolerant ───────────────────────────────────────────────────


def test_removal_tolerates_a_container_that_is_already_gone(monkeypatch):
    """`--rm` racing ahead of us is the normal case, not an error."""
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_mod, "_docker_is_usable", lambda *a, **k: True)

    def fake_run(argv, *a, **k):
        return subprocess.CompletedProcess(
            argv, returncode=1, stdout="", stderr="Error: No such container: x"
        )

    monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)
    assert _force_remove_container("x") is False  # reported, not raised


@pytest.mark.parametrize(
    "exc", [OSError("nope"), subprocess.TimeoutExpired("docker", 1)]
)
def test_removal_survives_a_broken_or_hanging_cli(monkeypatch, exc):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_mod, "_docker_is_usable", lambda *a, **k: True)

    def fake_run(*a, **k):
        raise exc

    monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)
    assert _force_remove_container("x") is False


def test_removal_is_a_noop_without_docker(monkeypatch):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: None)
    assert _force_remove_container("x") is False


# ── orphan sweep ──────────────────────────────────────────────────────────


def test_sweep_filters_on_the_project_label(monkeypatch):
    """A sweep must not touch containers this project did not start."""
    calls: list[list[str]] = []

    def fake_run(argv, *a, **k):
        calls.append(list(argv))
        if argv[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(argv, 0, stdout="abc\ndef\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_mod, "_docker_is_usable", lambda *a, **k: True)
    monkeypatch.setattr(sandbox_mod.subprocess, "run", fake_run)

    removed = DockerSandbox.sweep_orphaned_containers()

    assert removed == ["abc", "def"]
    assert f"label={CONTAINER_LABEL_KEY}" in calls[0]


def test_sweep_is_a_noop_without_docker(monkeypatch):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: None)
    assert DockerSandbox.sweep_orphaned_containers() == []


def test_sweep_is_not_automatic():
    """
    It cannot distinguish a leaked container from one belonging to another
    agent running right now, so it must stay an explicit call.
    """
    module = Path(__file__).resolve().parents[1] / "modules" / "sandbox.py"
    source = module.read_text()
    assert "atexit.register(_cleanup_active_containers)" in source
    assert "atexit.register(DockerSandbox.sweep_orphaned_containers)" not in source


# ── atexit hook ───────────────────────────────────────────────────────────


def test_atexit_hook_clears_the_registry(docker_calls):
    sandbox_mod._ACTIVE_CONTAINERS.add("repopilot-sandbox-leaked")
    sandbox_mod._cleanup_active_containers()

    assert removals(docker_calls) == [
        ["docker", "rm", "--force", "repopilot-sandbox-leaked"]
    ]
    assert sandbox_mod._ACTIVE_CONTAINERS == set()


def test_atexit_hook_never_raises(monkeypatch):
    """It runs during interpreter shutdown, where an exception is unhelpful."""
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda exe: "/usr/bin/docker")
    monkeypatch.setattr(sandbox_mod, "_docker_is_usable", lambda *a, **k: True)

    def explode(*a, **k):
        raise RuntimeError("interpreter is going away")

    monkeypatch.setattr(sandbox_mod.subprocess, "run", explode)
    sandbox_mod._ACTIVE_CONTAINERS.add("repopilot-sandbox-x")

    sandbox_mod._cleanup_active_containers()  # must not raise
    assert sandbox_mod._ACTIVE_CONTAINERS == set()
