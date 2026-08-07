"""
Module 5: Execution Sandbox
Runs code in isolation using subprocess with:
- Timeout enforcement
- stdout/stderr capture
- Exit code tracking
- Restricted environment
- Optional Docker support
"""

import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


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

# ---------------------------------------------------------------------------
# Environment sanitization
#
# Commands launched by this sandbox execute LLM-generated code that no human has
# reviewed. The process environment is therefore treated the same way file paths
# are treated elsewhere in this project: untrusted by default.
#
# This is an ALLOWLIST. Any variable not named below is dropped. A denylist only
# protects the names somebody happened to think of, so every service a user
# newly adopts is unprotected until someone remembers to extend the list.
#
# Adding a name here is a deliberate decision. Prefer the runtime passthrough
# hook (see PASSTHROUGH_ENV_VAR) over widening the defaults for a local need.
# ---------------------------------------------------------------------------

# Core shell / OS. The Windows entries are not optional: CPython cannot
# initialize sockets or SSL without SYSTEMROOT, so stripping it breaks any test
# that touches the network stack, and subprocesses fail without COMSPEC.
_CORE_ENV_VARS = {
    "PATH", "HOME", "SHELL", "USER", "LOGNAME", "HOSTNAME",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "TMPDIR",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
    "TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "USERPROFILE", "USERNAME",
    "HOMEDRIVE", "HOMEPATH", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
}

_PYTHON_ENV_VARS = {
    "PYTHONPATH", "PYTHONHOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
    "PYTHONHASHSEED", "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONWARNINGS",
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "PYENV_ROOT", "PYENV_VERSION",
    "PYTEST_ADDOPTS", "PY_COLORS", "PIP_CACHE_DIR",
}

_NODE_ENV_VARS = {
    "NODE_PATH", "NODE_ENV", "NODE_OPTIONS",
    "npm_config_cache", "npm_config_prefix",
}

_GO_ENV_VARS = {
    "GOROOT", "GOPATH", "GOCACHE", "GOMODCACHE", "GOTMPDIR",
    "GOFLAGS", "GOTOOLCHAIN", "GOOS", "GOARCH", "CGO_ENABLED",
}

_RUST_ENV_VARS = {
    "CARGO_HOME", "RUSTUP_HOME", "RUSTC", "RUSTFLAGS", "CARGO_TARGET_DIR",
}

_RUBY_ENV_VARS = {
    "GEM_HOME", "GEM_PATH", "RUBYOPT", "BUNDLE_PATH", "BUNDLE_GEMFILE",
    "RBENV_ROOT", "RBENV_VERSION",
}

_BUILD_ENV_VARS = {
    "JAVA_HOME", "MAKEFLAGS", "CC", "CXX",
    # Generic CI markers. Many suites branch on these; none carry credentials.
    # GITHUB_* is deliberately absent -- see the note in _build_safe_env.
    "CI", "CONTINUOUS_INTEGRATION",
}

# DockerSandbox shells out to the `docker` CLI *through* SubprocessSandbox.run,
# so the allowlist applies to the CLI itself. Without these, Docker Desktop,
# colima, Rancher Desktop and remote-daemon setups all fail to find a daemon.
# These configure the client on the host; `docker run` does not forward host
# environment into the container, so they are not exposed to sandboxed code
# when the Docker path is active.
_DOCKER_CLI_ENV_VARS = {
    "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
    "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY", "DOCKER_BUILDKIT",
}

ALLOWED_ENV_VARS = (
    _CORE_ENV_VARS
    | _PYTHON_ENV_VARS
    | _NODE_ENV_VARS
    | _GO_ENV_VARS
    | _RUST_ENV_VARS
    | _RUBY_ENV_VARS
    | _BUILD_ENV_VARS
    | _DOCKER_CLI_ENV_VARS
)

# Escape hatch. Set to a comma-separated list of variable names to pass through
# in addition to the defaults, e.g. for an authenticated module proxy:
#   export REPOPILOT_SANDBOX_ENV_PASSTHROUGH=GOPROXY,npm_config_registry
# Names added this way are logged, because widening the boundary should be
# visible in the run log rather than silent.
PASSTHROUGH_ENV_VAR = "REPOPILOT_SANDBOX_ENV_PASSTHROUGH"

_LOG = logging.getLogger("agent.sandbox")


def _passthrough_names() -> set[str]:
    """Read additional allowed variable names from PASSTHROUGH_ENV_VAR."""
    raw = os.environ.get(PASSTHROUGH_ENV_VAR, "")
    return {name.strip() for name in raw.split(",") if name.strip()}


def _build_safe_env(extra_env: Optional[dict] = None) -> dict:
    """
    Build the environment for a sandboxed command.

    Only variables named in ALLOWED_ENV_VARS (plus any listed in
    PASSTHROUGH_ENV_VAR) are passed through; everything else is dropped. This
    keeps credentials such as ANTHROPIC_API_KEY, SSH_AUTH_SOCK and KUBECONFIG
    out of reach of code the model wrote.

    It also drops the GitHub Actions runner variables. GITHUB_ENV and
    GITHUB_PATH name writable files that inject environment entries and PATH
    entries into *subsequent workflow steps*, so leaking them lets sandboxed
    code influence a job that holds contents:write and pull-requests:write.

    Matching is case-insensitive because Windows normalizes environment keys to
    upper case, which would otherwise drop lower-case entries like
    ``npm_config_cache``. No allowlisted name collides with a credential name
    under case folding.

    ``extra_env`` is a trusted-caller override applied *after* filtering, so
    callers inside RepoPilot can inject variables a runner needs. Never forward
    model-supplied data into it -- doing so reopens the boundary this closes.
    """
    allowed = {name.lower() for name in ALLOWED_ENV_VARS}

    extra_names = _passthrough_names()
    if extra_names:
        allowed |= {name.lower() for name in extra_names}
        _LOG.info(
            "sandbox: %s widened the environment allowlist with: %s",
            PASSTHROUGH_ENV_VAR, ", ".join(sorted(extra_names)),
        )

    env = {key: value for key, value in os.environ.items() if key.lower() in allowed}
    env.setdefault("PATH", os.environ.get("PATH", ""))

    stripped = sorted(set(os.environ) - set(env))
    if stripped:
        # Names only, never values. A test that fails because it cannot see a
        # variable it needs is much easier to diagnose if the sandbox says so.
        _LOG.debug(
            "sandbox: stripped %d environment variable(s): %s",
            len(stripped), ", ".join(stripped),
        )

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

    def run_tests(self, runner: str = "pytest", extra_args: Optional[list[str]] = None) -> ExecutionResult:
        if not self._docker_available:
            sb = SubprocessSandbox(self.working_dir, self.timeout)
            return sb.run_tests(runner, extra_args)

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
        return sb.run(docker_cmd)
