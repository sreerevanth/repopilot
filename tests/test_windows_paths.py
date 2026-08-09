"""
Tests for cross-platform path handling (modules/code_modifier.py, modules/sandbox.py).

Two places treated paths with the host's flavour, so the same input behaved
differently depending on which machine the agent ran on.

`_safe_abs_path` normalised backslashes and then called `Path.is_absolute()`.
That picks the host flavour: `C:/Windows/evil.txt` is absolute on Windows and
rejected, and merely unusual on Linux, where it silently created a directory
named `C:` inside the repository.

`_build_docker_command` interpolated the host path directly. Docker Desktop on
Windows needs forward slashes, and a colon there is the separator between host
path, container path and mode.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.code_modifier import CodeModificationEngine  # noqa: E402
from modules.sandbox import _docker_mount_path  # noqa: E402


@pytest.fixture
def engine():
    root = tempfile.mkdtemp()
    return CodeModificationEngine(root, f"{root}/.backups")


# ── paths the model must not be able to write to ──────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "sub/../../../out.txt",
        r"C:\Windows\system32\evil.txt",
        "C:/Windows/system32/evil.txt",
        r"\\server\share\evil.txt",
        "//server/share/evil.txt",
        r"\Windows\evil.txt",
    ],
)
def test_escaping_paths_are_refused(engine, path):
    with pytest.raises(ValueError):
        engine._safe_abs_path(path)


def test_a_drive_relative_path_is_refused(engine):
    """
    "C:file.txt" is absolute on neither flavour, but it still names a location
    on another drive rather than in this repository.
    """
    with pytest.raises(ValueError) as excinfo:
        engine._safe_abs_path("C:file.txt")

    assert "Drive-qualified" in str(excinfo.value)


def test_the_rejection_is_the_same_on_every_platform(engine):
    """
    The point of the fix. Previously a drive-absolute path was rejected on
    Windows and quietly accepted on Linux, so the same model output produced
    different behaviour depending on the machine.
    """
    for path in (r"C:\Windows\evil.txt", "C:/Windows/evil.txt"):
        with pytest.raises(ValueError):
            engine._safe_abs_path(path)


# ── ordinary paths still work ─────────────────────────────────────────────


@pytest.mark.parametrize("path", ["file.py", "sub/file.py", "a/b/c/file.py", "./file.py"])
def test_normal_paths_are_allowed(engine, path):
    assert engine._safe_abs_path(path)


def test_backslash_separators_are_accepted(engine):
    """A model on Windows may well emit sub\\file.py; that is not an escape."""
    assert engine._safe_abs_path(r"sub\file.py")


def test_an_empty_path_is_refused(engine):
    with pytest.raises(ValueError):
        engine._safe_abs_path("   ")


# ── docker mounts ─────────────────────────────────────────────────────────


def test_a_posix_path_is_unchanged():
    assert _docker_mount_path("/home/user/repo") == "/home/user/repo"


@pytest.mark.parametrize(
    "windows,expected",
    [
        (r"C:\Users\saket\repo", "C:/Users/saket/repo"),
        (r"D:\repopilot", "D:/repopilot"),
        (r"C:\Program Files\repo", "C:/Program Files/repo"),
    ],
)
def test_windows_paths_become_forward_slashed(windows, expected):
    assert _docker_mount_path(windows) == expected


def test_an_absolute_path_is_not_re_resolved():
    """
    os.path.abspath uses the host flavour, so calling it on "D:/repo" from
    Linux would prepend the working directory and produce nonsense.
    """
    assert _docker_mount_path("D:/repo") == "D:/repo"


def test_a_relative_path_is_resolved():
    assert Path(_docker_mount_path("some/dir")).is_absolute()


@pytest.mark.parametrize("unc", [r"\\server\share\repo", "//server/share/repo"])
def test_a_network_path_is_refused(unc):
    """
    Docker cannot bind-mount a UNC path. Saying so beats emitting an argument
    that fails later with Docker's own less specific message.
    """
    with pytest.raises(ValueError) as excinfo:
        _docker_mount_path(unc)

    assert "network path" in str(excinfo.value)


def test_the_mount_argument_uses_the_converted_path():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "sandbox.py"
    ).read_text(encoding="utf-8")

    assert '_docker_mount_path(self.working_dir)' in source
    assert 'f"{self.working_dir}:/workspace:rw"' not in source
