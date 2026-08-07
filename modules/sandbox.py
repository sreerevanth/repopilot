"""
Module 5: Execution Sandbox
Runs code in isolation using subprocess with:
- Timeout enforcement
- stdout/stderr capture
- Exit code tracking
- Restricted environment
- Optional Docker support
"""

import atexit
import logging
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import Optional

_LOG = logging.getLogger("agent.sandbox")

# Every container this module starts is named and labelled, so a leaked one can
# be found and removed. `--rm` alone is not enough: it is honoured by the daemon
# when the container *exits*, and a container whose CLI was killed never does.
CONTAINER_NAME_PREFIX = "repopilot-sandbox"
CONTAINER_LABEL_KEY = "com.repopilot.sandbox"

# Names of containers believed to be running right now, for the atexit sweep.
_ACTIVE_CONTAINERS: set[str] = set()
_ATEXIT_REGISTERED = False


@dataclass
class ExecutionResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self) -> str:
        status = "PASS" if self.success else ("TIMEOUT" if self.timed_out else "FAIL")
        lines = [
            f"{status} | exit={self.exit_code} | {self.duration_seconds:.2f}s",
            f"  cmd: {self.command}",
        ]
        if self.stdout.strip():
            lines.append(f"  stdout: {self.stdout[:500]}")
        if self.stderr.strip():
            lines.append(f"  stderr: {self.stderr[:500]}")
        return "\n".join(lines)


ALLOWED_RUNNERS = {
    "python": [sys.executable],
    "pytest": [sys.executable, "-m", "pytest"],
    "node": ["node"],
    "npm_test": ["npm", "test", "--"],
    "bash": ["bash"],
    "make": ["make"],
    "go": ["go", "test", "./..."],
    "cargo": ["cargo", "test"],
    "ruby": ["ruby"],
    "rspec": ["bundle", "exec", "rspec"],
}

DOCKER_RUNNERS = {
    "python": ["python"],
    "pytest": ["python", "-m", "pytest"],
    "node": ["node"],
    "npm_test": ["npm", "test", "--"],
    "bash": ["bash"],
    "make": ["make"],
    "go": ["go", "test", "./..."],
    "cargo": ["cargo", "test"],
    "ruby": ["ruby"],
    "rspec": ["bundle", "exec", "rspec"],
}

BLOCKED_ENV_VARS = {
    "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID",
    "GITHUB_TOKEN", "GH_TOKEN",
    "DATABASE_URL", "REDIS_URL",
}


def _build_safe_env(extra_env: Optional[dict] = None) -> dict:
    """Build a safe environment dict, removing sensitive vars."""
    env = {key: value for key, value in os.environ.items() if key not in BLOCKED_ENV_VARS}
    env.setdefault("PATH", os.environ.get("PATH", ""))
    if extra_env:
        env.update(extra_env)
    return env


def _resolve_runner(runner_name: str) -> Optional[list[str]]:
    """Resolve a runner name to an executable command list."""
    if runner_name not in ALLOWED_RUNNERS:
        return None
    candidates = ALLOWED_RUNNERS[runner_name]
    executable = candidates[0]
    if shutil.which(executable):
        return candidates
    return None


def _coerce_output(value) -> str:
    """Normalize subprocess output from text or byte mode."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _new_container_name() -> str:
    return f"{CONTAINER_NAME_PREFIX}-{uuid.uuid4().hex[:12]}"


def _force_remove_container(name: str, timeout: int = 30) -> bool:
    """
    Remove a container by name, tolerating the case where it is already gone.

    `docker rm -f` exits non-zero with "No such container" when `--rm` already
    cleaned up, and can transiently report "removal in progress". Neither is an
    error worth surfacing, so this reports success/failure and never raises.
    """
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "rm", "--force", name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _LOG.debug("could not remove container %s: %s", name, exc)
        return False

    if proc.returncode == 0:
        _LOG.debug("removed leaked container %s", name)
        return True

    _LOG.debug("container %s not removed (likely already gone): %s",
               name, proc.stderr.strip()[:200])
    return False


def _cleanup_active_containers() -> None:
    """atexit hook. Must never raise -- it runs during interpreter shutdown."""
    for name in list(_ACTIVE_CONTAINERS):
        try:
            _force_remove_container(name)
        except Exception:  # pragma: no cover - defensive during shutdown
            pass
        _ACTIVE_CONTAINERS.discard(name)


def _register_atexit_once() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_cleanup_active_containers)
        _ATEXIT_REGISTERED = True


class SubprocessSandbox:
    """
    Subprocess-based sandbox (no Docker required).
    Suitable for trusted local repos. For untrusted code, use DockerSandbox.
    """

    def __init__(
        self,
        working_dir: str,
        timeout_seconds: int = 60,
        max_output_bytes: int = 1024 * 1024,
    ):
        self.working_dir = os.path.abspath(working_dir)
        self.timeout = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def run(
        self,
        command: list[str],
        env: Optional[dict] = None,
        stdin_data: Optional[str] = None,
    ) -> ExecutionResult:
        """Run an arbitrary command list in the working dir sandbox."""
        import time

        safe_env = _build_safe_env(env)
        cmd_str = shlex.join(command)
        start = time.monotonic()

        try:
            proc = subprocess.run(
                command,
                cwd=self.working_dir,
                env=safe_env,
                input=stdin_data,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - start

            return ExecutionResult(
                command=cmd_str,
                exit_code=proc.returncode,
                stdout=proc.stdout[:self.max_output_bytes],
                stderr=proc.stderr[:self.max_output_bytes],
                timed_out=False,
                duration_seconds=elapsed,
            )

        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                command=cmd_str,
                exit_code=-1,
                stdout=_coerce_output(exc.stdout)[:self.max_output_bytes],
                stderr=_coerce_output(exc.stderr)[:self.max_output_bytes],
                timed_out=True,
                duration_seconds=elapsed,
            )

        except FileNotFoundError:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                command=cmd_str,
                exit_code=127,
                stdout="",
                stderr=f"Command not found: {command[0]}",
                timed_out=False,
                duration_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                command=cmd_str,
                exit_code=-2,
                stdout="",
                stderr=f"Sandbox error: {exc}",
                timed_out=False,
                duration_seconds=elapsed,
            )

    def run_tests(self, runner: str = "pytest", extra_args: Optional[list[str]] = None) -> ExecutionResult:
        """Run the project's test suite using a named runner."""
        cmd = _resolve_runner(runner)
        if cmd is None:
            return ExecutionResult(
                command=runner,
                exit_code=-3,
                stdout="",
                stderr=f"Runner '{runner}' not found or not allowed",
                timed_out=False,
                duration_seconds=0.0,
            )
        return self.run(cmd + (extra_args or []))

    def run_file(self, relative_path: str, runner: str = "python") -> ExecutionResult:
        """Run a specific file using the given runner."""
        cmd = _resolve_runner(runner)
        if cmd is None:
            return ExecutionResult(
                command=f"{runner} {relative_path}",
                exit_code=-3,
                stdout="",
                stderr=f"Runner '{runner}' not found or not allowed",
                timed_out=False,
                duration_seconds=0.0,
            )
        abs_path = os.path.join(self.working_dir, relative_path)
        return self.run(cmd + [abs_path])


class DockerSandbox:
    """
    Docker-based sandbox for untrusted code.
    Falls back to SubprocessSandbox if Docker is unavailable.
    """

    def __init__(
        self,
        working_dir: str,
        image: str = "python:3.11-slim",
        timeout_seconds: int = 120,
    ):
        self.working_dir = os.path.abspath(working_dir)
        self.image = image
        self.timeout = timeout_seconds
        self._docker_available = shutil.which("docker") is not None
        self._containers: set[str] = set()

    def run_tests(self, runner: str = "pytest", extra_args: Optional[list[str]] = None) -> ExecutionResult:
        if not self._docker_available:
            sb = SubprocessSandbox(self.working_dir, self.timeout)
            return sb.run_tests(runner, extra_args)

        runner_cmd = DOCKER_RUNNERS.get(runner) or ["python", "-m", "pytest"]
        inner_cmd = runner_cmd + (extra_args or [])

        name = _new_container_name()
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", name,
            "--label", f"{CONTAINER_LABEL_KEY}=1",
            "--network=none",
            "--memory=512m",
            "--cpus=1",
            "-v", f"{self.working_dir}:/workspace:rw",
            "-w", "/workspace",
            self.image,
            "sh", "-c", shlex.join(inner_cmd),
        ]

        sb = SubprocessSandbox(self.working_dir, self.timeout)

        _register_atexit_once()
        _ACTIVE_CONTAINERS.add(name)
        self._containers.add(name)
        try:
            result = sb.run(docker_cmd)
        except BaseException:
            # KeyboardInterrupt and SystemExit reach here; the CLI is gone but
            # the container is not, so remove it before the exception continues.
            _force_remove_container(name)
            raise
        finally:
            _ACTIVE_CONTAINERS.discard(name)
            self._containers.discard(name)

        if result.timed_out:
            # subprocess.run kills the docker CLI with SIGKILL on timeout. That
            # signal cannot be caught or proxied, so the container keeps running
            # -- still holding its memory and CPU reservation -- and `--rm` will
            # not fire because the container never exits. Remove it explicitly.
            _LOG.warning(
                "DockerSandbox: run timed out; force-removing container %s", name
            )
            _force_remove_container(name)

        return result

    # ── cleanup surface ───────────────────────────────────────────────

    def cleanup(self) -> None:
        """Remove any container this instance started that is still tracked."""
        for name in list(self._containers):
            _force_remove_container(name)
            self._containers.discard(name)
            _ACTIVE_CONTAINERS.discard(name)

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False  # never swallow the exception

    @classmethod
    def sweep_orphaned_containers(cls) -> list[str]:
        """
        Remove every container this project has ever labelled, and return the
        ids removed.

        For the case nothing in-process can cover: the agent killed with
        SIGKILL, or the machine losing power. Deliberately *not* automatic on
        startup -- a sweep cannot tell a leaked container from one belonging to
        another agent running right now, so it is the caller's decision.
        """
        if shutil.which("docker") is None:
            return []
        try:
            listed = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"label={CONTAINER_LABEL_KEY}"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _LOG.debug("could not list sandbox containers: %s", exc)
            return []

        ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        return [cid for cid in ids if _force_remove_container(cid)]
