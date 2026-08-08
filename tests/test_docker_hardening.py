"""
Tests for DockerSandbox container hardening (modules/sandbox.py).

The `docker run` argv is assembled by `_build_docker_command`, which is split
out from `run_tests` precisely so these can run without a daemon. A security
control that is only exercised when Docker happens to be installed is a control
nobody checks.

These assert the flag set. They do not prove a real container starts — that
needs Docker, and `scripts/verify_docker_sandbox.py` covers it.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sandbox import DockerSandbox, _docker_user_flags  # noqa: E402


@pytest.fixture
def cmd(tmp_path):
    sandbox = DockerSandbox(str(tmp_path))
    return sandbox._build_docker_command(["python", "-m", "pytest"], "test-container")


def _flag_value(argv, flag):
    """Value following a space-separated flag, or None."""
    return argv[argv.index(flag) + 1] if flag in argv else None


# ── the controls the issue asked for ──────────────────────────────────────


def test_all_capabilities_are_dropped(cmd):
    assert _flag_value(cmd, "--cap-drop") == "ALL"


def test_no_new_privileges_is_set(cmd):
    """Stops a setuid binary inside the image regaining what --cap-drop removed."""
    assert _flag_value(cmd, "--security-opt") == "no-new-privileges"


def test_pids_are_limited(cmd):
    """Without this a fork bomb is bounded only by the memory cap."""
    assert "--pids-limit=256" in cmd


def test_pids_limit_is_configurable(tmp_path):
    argv = DockerSandbox(str(tmp_path), pids_limit=1024)._build_docker_command(["x"], "test-container")
    assert "--pids-limit=1024" in argv
    assert "--pids-limit=256" not in argv


# ── --user and its platform guard ─────────────────────────────────────────


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX only")
def test_runs_as_the_invoking_user_on_posix(cmd):
    assert _flag_value(cmd, "--user") == f"{os.getuid()}:{os.getgid()}"


def test_user_flag_is_omitted_when_getuid_is_unavailable(monkeypatch):
    """
    os.getuid does not exist on Windows. Referencing it unconditionally would
    raise AttributeError before Docker was ever invoked.
    """
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delattr(os, "getgid", raising=False)
    assert _docker_user_flags() == []


def test_command_still_builds_without_getuid(monkeypatch, tmp_path):
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delattr(os, "getgid", raising=False)

    sandbox = DockerSandbox(str(tmp_path))
    argv = sandbox._build_docker_command(["python", "-m", "pytest"], "test-container")
    assert argv[0] == "docker"
    assert "--user" not in argv
    # Everything that is not UID-dependent must still be applied.
    assert "--cap-drop" in argv
    assert "no-new-privileges" in argv
    assert "--pids-limit=256" in argv


def test_user_flag_uses_uid_gid_form(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 1001, raising=False)
    assert _docker_user_flags() == ["--user", "1000:1001"]


# ── compensating controls for --user ──────────────────────────────────────


def test_tmp_is_writable(cmd):
    """
    --user leaves no passwd entry, so HOME is unset and pytest's cache, npm and
    go all fail on a read-only home. A tmpfs restores that without host access.
    """
    tmpfs = _flag_value(cmd, "--tmpfs")
    assert tmpfs is not None
    assert tmpfs.startswith("/tmp:")
    assert "nosuid" in tmpfs
    assert "nodev" in tmpfs


def test_home_points_at_the_tmpfs(cmd):
    assert "HOME=/tmp" in cmd


# ── read-only root filesystem (opt-in) ────────────────────────────────────


def test_read_only_is_off_by_default(cmd):
    """It breaks any runner writing outside /workspace and /tmp."""
    assert "--read-only" not in cmd


def test_read_only_can_be_enabled(tmp_path):
    argv = DockerSandbox(str(tmp_path), read_only=True)._build_docker_command(["x"], "test-container")
    assert "--read-only" in argv


# ── existing guarantees must survive ──────────────────────────────────────


def test_existing_isolation_flags_are_preserved(cmd):
    assert "--network=none" in cmd
    assert "--memory=512m" in cmd
    assert "--cpus=1" in cmd
    assert "--rm" in cmd


def test_workspace_stays_writable(cmd, tmp_path):
    """Made rw in #8 so tests can create fixtures; that is not reverted here."""
    assert f"{os.path.abspath(str(tmp_path))}:/workspace:rw" in cmd
    assert _flag_value(cmd, "-w") == "/workspace"


def test_inner_command_is_not_string_interpolated(tmp_path):
    """shlex.join, so a path with spaces or quotes cannot break out of sh -c."""
    argv = DockerSandbox(str(tmp_path))._build_docker_command(
        ["pytest", "-k", "weird name'; rm -rf /"], "test-container"
    )
    assert argv[-2] == "-c"
    assert "'weird name'\"'\"'; rm -rf /'" in argv[-1]


def test_image_precedes_the_inner_command(cmd):
    """Flags after the image name are passed to the container, not to docker."""
    assert cmd.index("python:3.11-slim") < cmd.index("sh")


def test_custom_image_is_honoured(tmp_path):
    sandbox = DockerSandbox(str(tmp_path), image="node:20-slim")
    argv = sandbox._build_docker_command(["x"], "test-container")
    assert "node:20-slim" in argv
