#!/usr/bin/env python3
"""
Verify DockerSandbox hardening against a real Docker daemon.

tests/test_docker_hardening.py asserts the `docker run` argv without needing a
daemon. That proves the flags are passed; it cannot prove a container still
runs correctly with them applied. This script closes that gap.

    python scripts/verify_docker_sandbox.py

Exits 0 if every check passes, 1 otherwise. Skips cleanly if Docker is absent.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sandbox import DockerSandbox  # noqa: E402

IMAGE = os.environ.get("REPOPILOT_VERIFY_IMAGE", "python:3.11-slim")

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")


def docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=30,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def make_project(tmp: Path) -> Path:
    """A suite that exercises the workspace mount and the tmpfs home."""
    project = tmp / "proj"
    project.mkdir()
    (project / "test_sandbox_env.py").write_text(
        "import os\n"
        "\n"
        "def test_can_write_to_workspace():\n"
        "    with open('artifact.txt', 'w') as fh:\n"
        "        fh.write('written from inside the container')\n"
        "\n"
        "def test_home_is_writable():\n"
        "    home = os.environ.get('HOME')\n"
        "    assert home, 'HOME is unset -- --user has no passwd entry'\n"
        "    path = os.path.join(home, 'probe')\n"
        "    with open(path, 'w') as fh:\n"
        "        fh.write('ok')\n"
        "\n"
        "def test_network_is_off():\n"
        "    import socket\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    except OSError:\n"
        "        return\n"
        "    raise AssertionError('network reachable -- --network=none not applied')\n"
    )
    return project


def main() -> int:
    print(f"Verifying DockerSandbox hardening (image: {IMAGE})\n")

    if not docker_ready():
        print("  SKIP  Docker CLI or daemon unavailable - nothing verified.")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        project = make_project(tmp)

        sandbox = DockerSandbox(str(project), image=IMAGE, timeout_seconds=300)

        argv = sandbox._build_docker_command(["python", "-m", "pytest", "-q"])
        print("  argv:", " ".join(argv), "\n")

        result = sandbox.run_tests("pytest", ["-q"])

        check(
            "suite passes with all hardening flags applied",
            result.success,
            "" if result.success else (result.stdout + result.stderr)[-1500:],
        )
        check(
            "run did not hit the timeout",
            not result.timed_out,
        )

        artifact = project / "artifact.txt"
        check(
            "container could write into the workspace mount",
            artifact.exists(),
            "" if artifact.exists() else "artifact.txt was not created",
        )

        # The point of --user: files must not come back owned by root.
        if artifact.exists() and hasattr(os, "getuid"):
            owner = artifact.stat().st_uid
            check(
                "workspace files are owned by the invoking user, not root",
                owner == os.getuid(),
                f"owner uid={owner}, invoking uid={os.getuid()}",
            )
        elif not hasattr(os, "getuid"):
            print("  SKIP  ownership check (no getuid on this platform)")

        # Opt-in read-only root filesystem.
        ro = DockerSandbox(
            str(project), image=IMAGE, timeout_seconds=300, read_only=True
        )
        ro_result = ro.run_tests("pytest", ["-q"])
        check(
            "read_only=True still runs the suite",
            ro_result.success,
            "" if ro_result.success
            else "read_only is opt-in; a failure here is informational, "
                 "not a regression: " + (ro_result.stdout + ro_result.stderr)[-800:],
        )

    print()
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"All {len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
