"""
Module 5: Execution Sandbox
Runs code in isolation using subprocess with:
- Timeout enforcement
- stdout/stderr capture
- Exit code tracking
- Restricted environment
- Optional Docker support
"""

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

_LOG = logging.getLogger("agent.sandbox")

# Which executor actually ran a command. Recorded on every ExecutionResult so a
# run log can be audited after the fact -- without this, a run that silently
# lost its isolation is indistinguishable from one that kept it.
SANDBOX_DOCKER = "docker"
SANDBOX_SUBPROCESS = "subprocess"
SANDBOX_SUBPROCESS_FALLBACK = "subprocess-fallback"


class SandboxUnavailableError(RuntimeError):
    """
    Raised when DockerSandbox(strict=True) cannot provide isolation.

    Degrading to an unisolated executor is a reasonable default for interactive
    use and a bad one for CI, so strict mode turns it into a hard failure.
    """


@dataclass
class ExecutionResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    sandbox: str = SANDBOX_SUBPROCESS

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def isolated(self) -> bool:
        """True only when the command actually ran inside a container."""
        return self.sandbox == SANDBOX_DOCKER

    def summary(self) -> str:
        status = "PASS" if self.success else ("TIMEOUT" if self.timed_out else "FAIL")
        lines = [
            f"{status} | exit={self.exit_code} | {self.duration_seconds:.2f}s "
            f"| sandbox={self.sandbox}",
            f"  cmd: {self.command}",
        ]
        if self.stdout.strip():
            lines.append(f"  stdout: {self.stdout[:500]}")
        if self.stderr.strip():
            lines.append(f"  stderr: {self.stderr[:500]}")
        return "\n".join(lines)


# `npx --no-install` resolves the project's own node_modules/.bin and fails if the
# package is absent, rather than downloading it. That matters here: the sandbox
# is meant to be hermetic, and a runner that silently fetches from the registry
# would defeat DockerSandbox's --network=none and surprise offline users.
#
# vitest is invoked as `vitest run` on purpose. Bare `vitest` starts a watch
# server when it thinks it is interactive; under this sandbox it would hold the
# process open until timeout_seconds elapsed and be reported as a test timeout.
ALLOWED_RUNNERS = {
    "python": [sys.executable],
    "pytest": [sys.executable, "-m", "pytest"],
    "node": ["node"],
    "npm_test": ["npm", "test", "--"],
    "vitest": ["npx", "--no-install", "vitest", "run"],
    "jest": ["npx", "--no-install", "jest"],
    "bash": ["bash"],
    "make": ["make"],
    "go": ["go", "test", "./..."],
    "cargo": ["cargo", "test"],
    "ruby": ["ruby"],
    "rspec": ["bundle", "exec", "rspec"],
}

# Linters run before the suite. A syntax error or an undefined name is caught in
# under a second and gives the model a precise location, where the same mistake
# via pytest arrives as a collection error buried in a traceback.
#
# Each entry is the command; the runner is chosen explicitly rather than
# guessed, because a repo with both ruff and eslint configured has no single
# right answer.
# Errors only, not style. A default rule set would fail on things the model did
# not cause -- ruff's I001 fires on unsorted imports in perfectly valid code --
# and since the model cannot fix what it did not write, the gate would loop
# until max_iterations on every run. E9 is syntax errors, F is pyflakes
# (undefined names, unused imports). Widen it per project with --lint-args.
ALLOWED_LINTERS = {
    "ruff": [sys.executable, "-m", "ruff", "check", "--select", "E9,F", "."],
    "flake8": [sys.executable, "-m", "flake8", "--select=E9,F63,F7,F82", "."],
    "pyflakes": [sys.executable, "-m", "pyflakes", "."],
    "eslint": ["npx", "--no-install", "eslint", "--quiet", "."],
    "tsc": ["npx", "--no-install", "tsc", "--noEmit"],
    "govet": ["go", "vet", "./..."],
    "clippy": ["cargo", "clippy"],
}

DOCKER_RUNNERS = {
    "python": ["python"],
    "pytest": ["python", "-m", "pytest"],
    "node": ["node"],
    "npm_test": ["npm", "test", "--"],
    "vitest": ["npx", "--no-install", "vitest", "run"],
    "jest": ["npx", "--no-install", "jest"],
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


def _docker_is_usable(probe_timeout: int = 15) -> bool:
    """
    True only if the docker CLI exists *and* a daemon answers it.

    `shutil.which("docker")` alone is not enough. Docker Desktop installed but
    not running is the common case on developer laptops, and it leaves the
    binary on PATH. `docker run` then exits non-zero with "Cannot connect to the
    Docker daemon", which is indistinguishable from a failing test suite once it
    is wrapped in an ExecutionResult -- the agent reads it as a test failure and
    starts rewriting working code.
    """
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=probe_timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _coerce_output(value) -> str:
    """Normalize subprocess output from text or byte mode."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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

    def run_lint(
        self,
        linter: str,
        extra_args: Optional[list[str]] = None,
    ) -> ExecutionResult:
        """
        Run a linter over the working directory.

        Returns the same ExecutionResult shape as run_tests, so the loop can
        feed the output back to the model without special-casing it.
        """
        command = ALLOWED_LINTERS.get(linter)
        if command is None:
            return ExecutionResult(
                command=f"(unknown linter: {linter})",
                exit_code=-3,
                stdout="",
                stderr=(
                    f"Unknown linter '{linter}'. "
                    f"Available: {', '.join(sorted(ALLOWED_LINTERS))}"
                ),
                timed_out=False,
                duration_seconds=0.0,
            )

        if shutil.which(command[0]) is None:
            return ExecutionResult(
                command=" ".join(command),
                exit_code=-3,
                stdout="",
                stderr=f"Linter executable not found: {command[0]}",
                timed_out=False,
                duration_seconds=0.0,
            )

        return self.run(command + (extra_args or []))

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
        strict: bool = False,
    ):
        """
        `strict=True` raises SandboxUnavailableError instead of falling back to
        an unisolated executor. Use it in CI, where losing isolation should stop
        the run rather than quietly change what the guarantees are.
        """
        self.working_dir = os.path.abspath(working_dir)
        self.image = image
        self.timeout = timeout_seconds
        self.strict = strict
        self._docker_available = _docker_is_usable()

    def run_tests(self, runner: str = "pytest", extra_args: Optional[list[str]] = None) -> ExecutionResult:
        if not self._docker_available:
            reason = (
                "Docker is unavailable (no CLI on PATH, or no daemon answering). "
                "Network isolation, the 512MB memory cap and the 1-CPU limit are "
                "NOT in effect."
            )
            if self.strict:
                raise SandboxUnavailableError(reason)

            _LOG.warning("DockerSandbox: %s Falling back to SubprocessSandbox.", reason)
            sb = SubprocessSandbox(self.working_dir, self.timeout)
            result = sb.run_tests(runner, extra_args)
            result.sandbox = SANDBOX_SUBPROCESS_FALLBACK
            return result

        runner_cmd = DOCKER_RUNNERS.get(runner) or ["python", "-m", "pytest"]
        inner_cmd = runner_cmd + (extra_args or [])

        docker_cmd = [
            "docker", "run", "--rm",
            "--network=none",
            "--memory=512m",
            "--cpus=1",
            "-v", f"{self.working_dir}:/workspace:rw",
            "-w", "/workspace",
            self.image,
            "sh", "-c", shlex.join(inner_cmd),
        ]

        sb = SubprocessSandbox(self.working_dir, self.timeout)
        result = sb.run(docker_cmd)
        result.sandbox = SANDBOX_DOCKER
        return result
