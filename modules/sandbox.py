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
import fnmatch
import logging
import os
import re
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
import time
import sys
import uuid
from dataclasses import dataclass
from typing import Optional

# Which executor actually ran a command. Recorded on every ExecutionResult so a
# run log can be audited after the fact -- without this, a run that silently
# lost its isolation is indistinguishable from one that kept it.
SANDBOX_DOCKER = "docker"
SANDBOX_SUBPROCESS = "subprocess"
SANDBOX_SUBPROCESS_FALLBACK = "subprocess-fallback"

# Every container this module starts is named and labelled, so a leaked one can
# be found and removed. `--rm` alone is not enough: it is honoured by the daemon
# when the container *exits*, and a container whose CLI was killed never does.
CONTAINER_NAME_PREFIX = "repopilot-sandbox"
CONTAINER_LABEL_KEY = "com.repopilot.sandbox"

# Names of containers believed to be running right now, for the atexit sweep.
_ACTIVE_CONTAINERS: set[str] = set()
_ATEXIT_REGISTERED = False


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
@dataclass(frozen=True)
class LanguageRunner:
    """
    Everything the sandbox needs to know about one runner, in one place.

    This information used to live in five parallel dicts keyed by the same
    strings -- ALLOWED_RUNNERS, DOCKER_RUNNERS, MODULE_RUNNERS,
    RUNNER_FALLBACKS and the linter table. Adding a language meant remembering
    all of them, and DOCKER_RUNNERS was 10/12 an exact copy of ALLOWED_RUNNERS
    that existed only because two entries name the interpreter differently.

    The old dicts are still exported, derived from this registry, so nothing
    that imports them needs to change.
    """

    name: str
    language: str
    command: list[str]
    # Only set where the container genuinely differs -- for Python the host uses
    # sys.executable and the image uses whatever `python` resolves to. Ten of
    # the twelve runners are identical either side and say nothing here.
    container_command: Optional[list[str]] = None
    # Runners invoked as `python -m <module>`: shutil.which sees the
    # interpreter, which always exists, so availability needs an import probe.
    module: Optional[str] = None
    # What to try when this runner is unavailable, where a weaker signal beats
    # none. Only defined where the fallback can actually run the same suite.
    fallback: Optional[str] = None
    linters: tuple = ()


RUNNERS: dict = {
    r.name: r
    for r in [
        LanguageRunner("python", "python", [sys.executable], ["python"]),
        LanguageRunner(
            "pytest", "python", [sys.executable, "-m", "pytest"],
            ["python", "-m", "pytest"],
            module="pytest", fallback="python",
            linters=("ruff", "flake8", "pyflakes"),
        ),
        LanguageRunner("node", "javascript", ["node"]),
        LanguageRunner("npm_test", "javascript", ["npm", "test", "--"],
                       linters=("eslint", "tsc")),
        LanguageRunner("vitest", "javascript",
                       ["npx", "--no-install", "vitest", "run"],
                       linters=("eslint", "tsc")),
        LanguageRunner("jest", "javascript", ["npx", "--no-install", "jest"],
                       linters=("eslint", "tsc")),
        LanguageRunner("bash", "bash", ["bash"]),
        LanguageRunner("make", "make", ["make"]),
        LanguageRunner("go", "go", ["go", "test", "./..."], linters=("govet",)),
        LanguageRunner("cargo", "rust", ["cargo", "test"], linters=("clippy",)),
        LanguageRunner("ruby", "ruby", ["ruby"]),
        LanguageRunner("rspec", "ruby", ["bundle", "exec", "rspec"]),
    ]
}


def runners_for_language(language: str) -> list[str]:
    """Runner names registered for a language, for callers choosing one."""
    return sorted(name for name, r in RUNNERS.items() if r.language == language)


def linters_for_runner(runner: str) -> tuple:
    """Linters that make sense alongside a runner."""
    entry = RUNNERS.get(runner)
    return entry.linters if entry else ()


# ── Derived views ───────────────────────────────────────────────────────
#
# Kept so existing imports keep working. Deriving them is the point: the
# tables can no longer disagree with each other, because there is only one
# table.

ALLOWED_RUNNERS = {name: r.command for name, r in RUNNERS.items()}
DOCKER_RUNNERS = {
    name: (r.container_command or r.command) for name, r in RUNNERS.items()
}
MODULE_RUNNERS = {name: r.module for name, r in RUNNERS.items() if r.module}
RUNNER_FALLBACKS = {name: r.fallback for name, r in RUNNERS.items() if r.fallback}


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
PRE_COMMIT_CONFIG = ".pre-commit-config.yaml"

ALLOWED_LINTERS = {
    "ruff": [sys.executable, "-m", "ruff", "check", "--select", "E9,F", "."],
    "flake8": [sys.executable, "-m", "flake8", "--select=E9,F63,F7,F82", "."],
    "pyflakes": [sys.executable, "-m", "pyflakes", "."],
    "eslint": ["npx", "--no-install", "eslint", "--quiet", "."],
    "tsc": ["npx", "--no-install", "tsc", "--noEmit"],
    "govet": ["go", "vet", "./..."],
    "clippy": ["cargo", "clippy"],
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

    NOTE: This filters only the environment handed to sandbox subprocesses.
    The parent process keeps its own environment, so git_integration can
    still read GITHUB_TOKEN to open pull requests.

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


# Runners invoked as `python -m <module>`. `shutil.which` checks the executable,
# which for these is the interpreter itself and therefore always present -- so
# the "runner not found" guard never fires for them. Their availability has to
# be probed by importing the module instead.

# What each runner falls back to when it is unavailable. Only defined where the
# fallback can actually run the same suite: `python -m pytest` and plain
# `python <file>` both execute test files, so a repo whose tests are runnable as
# scripts still gets a real signal. There is no equivalent for cargo or go.


def _module_is_available(module: str) -> bool:
    """True if `python -m <module>` would find the module."""
    probe = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        timeout=30,
    )
    return probe.returncode == 0


# Conventional test-file names. Deliberately narrow: running arbitrary files
# with `python <file>` executes whatever is at module scope, so this only picks
# up files a developer would recognise as tests.
TEST_FILE_GLOBS = ("test_*.py", "*_test.py")
MAX_FALLBACK_FILES = 25


def _discover_test_files(root: str) -> list[str]:
    """Test files under root, as paths relative to it."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in ("node_modules", "__pycache__", "venv")
        ]
        for name in sorted(filenames):
            if any(fnmatch.fnmatch(name, g) for g in TEST_FILE_GLOBS):
                # Forward slashes, matching repo_ingestion and secret_scanner.
                # DockerSandbox mounts the repo into a Linux container, so a
                # Windows-discovered "tests\\test_x.py" would not resolve there.
                relative = os.path.relpath(os.path.join(dirpath, name), root)
                found.append(relative.replace(os.sep, "/"))
    return sorted(found)[:MAX_FALLBACK_FILES]


def _resolve_runner(runner_name: str) -> Optional[list[str]]:
    """Resolve a runner name to an executable command list."""
    if runner_name not in ALLOWED_RUNNERS:
        return None
    candidates = ALLOWED_RUNNERS[runner_name]
    executable = candidates[0]
    if not shutil.which(executable):
        return None

    module = MODULE_RUNNERS.get(runner_name)
    if module and not _module_is_available(module):
        # `python -m pytest` with pytest absent exits 1 with "No module named
        # pytest", which is indistinguishable from a failing suite once wrapped
        # in an ExecutionResult -- the agent reads it as a test failure and
        # starts rewriting code that was never broken.
        return None

    return candidates


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


# pytest-cov writes a total to stdout as "TOTAL ... 87%". Parsing that is
# cheaper and more portable than requiring a coverage.json, and it works with
# whatever coverage config the project already has.
_COVERAGE_TOTAL = re.compile(r"^TOTAL\s+.*?(\d+(?:\.\d+)?)%", re.MULTILINE)


def parse_coverage_percent(output: str) -> Optional[float]:
    """Extract the total coverage percentage from a pytest-cov report."""
    matches = _COVERAGE_TOTAL.findall(output or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def coverage_args(source_dir: str = ".") -> list[str]:
    """Arguments that make pytest emit a terminal coverage total."""
    return [f"--cov={source_dir}", "--cov-report=term"]


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


class Sandbox(ABC):
    """
    What every execution backend must provide.

    `SubprocessSandbox` and `DockerSandbox` were unrelated classes that happened
    to share some method names, and they had already drifted: `DockerSandbox`
    was missing `run`, `run_file`, `run_lint` and `run_pre_commit` entirely, so
    `--lint`, `--run-file` or a repository's pre-commit hooks raised
    AttributeError against it rather than doing anything. Nothing detected that,
    because nothing required the two to agree.

    Subclasses implement `run`. Everything else is expressed in terms of it, so
    a new backend -- Firecracker, a remote executor -- gets the whole surface by
    implementing one method, and one that forgets it cannot be instantiated.
    """

    working_dir: str
    timeout: int

    @abstractmethod
    def run(
        self,
        command: list[str],
        env: Optional[dict] = None,
        stdin_data: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a command list in the sandbox and capture its result."""

    def run_tests(self, runner: str = "pytest", extra_args: Optional[list[str]] = None) -> ExecutionResult:
        """Run the project's test suite using a named runner."""
        cmd = _resolve_runner(runner)

        if cmd is None:
            fallback = RUNNER_FALLBACKS.get(runner)
            fallback_cmd = _resolve_runner(fallback) if fallback else None

            if fallback_cmd is not None:
                files = _discover_test_files(self.working_dir)
                if files:
                    _LOG.warning(
                        "%s is unavailable; running %d test file(s) with %s "
                        "instead. This is a weaker signal than a suite run.",
                        runner, len(files), fallback,
                    )
                    return self._run_files_individually(fallback_cmd, files, extra_args)

            detail = f"Runner '{runner}' not found or not allowed"
            if fallback:
                detail += (
                    f". Falling back to '{fallback}' was not possible either "
                    f"(no test files found, or {fallback} is unavailable)."
                )
            return ExecutionResult(
                command=runner,
                exit_code=-3,
                stdout="",
                stderr=detail,
                timed_out=False,
                duration_seconds=0.0,
            )

        return self.run(cmd + (extra_args or []))

    def _run_files_individually(
        self,
        base_cmd: list[str],
        files: list[str],
        extra_args: Optional[list[str]] = None,
    ) -> ExecutionResult:
        """
        Run each test file as a script and combine the results.

        This is weaker than a real suite run and the returned result says so:
        `python <file>` executes module scope, so a file whose assertions live
        inside pytest-collected functions will exit 0 without running anything.
        A pass here means "nothing raised", not "the tests passed".
        """
        started = time.time()
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        worst_exit = 0
        timed_out = False

        for relative in files:
            result = self.run(base_cmd + [relative] + (extra_args or []))
            header = f"--- {relative} (exit {result.exit_code}) ---"
            stdout_parts.append(f"{header}\n{result.stdout}".rstrip())
            if result.stderr.strip():
                stderr_parts.append(f"{header}\n{result.stderr}".rstrip())
            if result.timed_out:
                timed_out = True
            if result.exit_code != 0:
                worst_exit = result.exit_code or 1

        note = (
            f"[fallback] pytest was unavailable; ran {len(files)} file(s) with "
            f"{os.path.basename(base_cmd[0])} instead. A pass here means nothing "
            f"raised at module scope, not that a test suite succeeded."
        )
        return ExecutionResult(
            command=f"{' '.join(base_cmd)} <{len(files)} test file(s)>",
            exit_code=worst_exit,
            stdout=note + "\n\n" + "\n\n".join(stdout_parts),
            stderr="\n\n".join(stderr_parts),
            timed_out=timed_out,
            duration_seconds=time.time() - started,
        )

    def run_pre_commit(
        self,
        files: list[str],
        config: str = PRE_COMMIT_CONFIG,
    ) -> ExecutionResult:
        """
        Run the repository's pre-commit hooks against the changed files.

        Hooks fall into two kinds and pre-commit reports both as "Failed":

        - **auto-fixing** (black, isort, trailing-whitespace) rewrite the file
          and exit non-zero to say they did something. Nothing is wrong; the
          code is now formatted.
        - **checking** (check-yaml, flake8) exit non-zero because something is
          actually wrong.

        Treating the first as a failure would send the agent back to fix code
        that a hook just fixed for it. The two are told apart by running again:
        an auto-fix passes the second time, a genuine failure does not.
        """
        if not shutil.which("pre-commit"):
            return ExecutionResult(
                command="pre-commit",
                exit_code=-3,
                stdout="",
                stderr="pre-commit is not installed. Run: pip install pre-commit",
                timed_out=False,
                duration_seconds=0.0,
            )
        if not files:
            return ExecutionResult(
                command="pre-commit (no files)",
                exit_code=0,
                stdout="No changed files to check.",
                stderr="",
                timed_out=False,
                duration_seconds=0.0,
            )

        command = ["pre-commit", "run", "--config", config, "--files", *files]
        first = self.run(command)
        if first.success:
            return first

        # Non-zero: re-run to find out which kind it was.
        second = self.run(command)
        if second.success:
            _LOG.info(
                "pre-commit modified %d file(s) and now passes; continuing.",
                len(files),
            )
            return ExecutionResult(
                command=" ".join(command),
                exit_code=0,
                stdout=(
                    "[pre-commit] Hooks reformatted the changed files and now "
                    "pass. The rewritten files are what will be tested.\n\n"
                    + first.stdout
                ),
                stderr="",
                timed_out=False,
                duration_seconds=first.duration_seconds + second.duration_seconds,
            )

        return second

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


class SubprocessSandbox(Sandbox):
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



def _docker_user_flags() -> list[str]:
    """
    Run the container as the invoking user, on platforms where that means
    something.

    Without --user the container runs as root over a read-write bind mount of
    the user's repository. Bind mounts preserve UIDs, so anything the executed
    code creates lands on the host owned by root -- files inside someone's own
    project that they cannot edit or delete without sudo.

    os.getuid/os.getgid do not exist on Windows, so the flag is omitted there.
    That is not a gap being papered over: Docker Desktop on Windows and macOS
    maps bind-mount ownership itself, so there is no host UID for the container
    to match and passing one can break the mount instead of securing it.
    """
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return []
    return ["--user", f"{getuid()}:{getgid()}"]


class DockerSandbox(Sandbox):
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
        pids_limit: int = 256,
        read_only: bool = False,
    ):
        """
        `strict=True` raises SandboxUnavailableError instead of falling back to
        an unisolated executor. Use it in CI, where losing isolation should stop
        the run rather than quietly change what the guarantees are.

        `pids_limit` caps process creation; a fork bomb is otherwise bounded
        only by the memory limit. Raise it for parallel runners that spawn a
        worker per core (pytest-xdist, jest) on a large machine.

        `read_only` makes the container filesystem read-only apart from the
        workspace mount and /tmp. Off by default because it breaks any runner
        that writes outside those paths; see the note in the README.
        """
        self.working_dir = os.path.abspath(working_dir)
        self.image = image
        self.timeout = timeout_seconds
        self.strict = strict
        self.pids_limit = pids_limit
        self.read_only = read_only
        self._docker_available = _docker_is_usable()
        self._containers: set[str] = set()

    def _build_docker_command(self, inner_cmd: list[str], name: str) -> list[str]:
        """
        Assemble the `docker run` argv.

        Split out from run_tests so the flag set can be asserted on directly --
        a security control that is only exercised when a daemon happens to be
        present is a control nobody checks.
        """
        cmd = [
            "docker", "run", "--rm",
            "--name", name,
            "--label", f"{CONTAINER_LABEL_KEY}=1",
            "--network=none",
            "--memory=512m",
            "--cpus=1",
            # Drop everything the container does not need. A test runner needs
            # no Linux capabilities, and no-new-privileges stops a setuid binary
            # inside the image from regaining any.
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            f"--pids-limit={self.pids_limit}",
        ]
        cmd += _docker_user_flags()

        if self.read_only:
            cmd.append("--read-only")

        # --user leaves the process without a passwd entry, so HOME is unset and
        # anything that wants a writable home -- pytest's cache, npm, go's build
        # cache -- fails in confusing ways. A tmpfs at /tmp plus HOME pointing at
        # it restores that without granting any access to the host.
        cmd += [
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "-e", "HOME=/tmp",
            "-v", f"{self.working_dir}:/workspace:rw",
            "-w", "/workspace",
            self.image,
            "sh", "-c", shlex.join(inner_cmd),
        ]
        return cmd

    def run(
        self,
        command: list[str],
        env: Optional[dict] = None,
        stdin_data: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Run an arbitrary command inside a container.

        The container lifecycle previously lived inside run_tests, so Docker
        could only ever run a test suite. Lifting it here is what gives the
        inherited run_lint, run_file and run_pre_commit containerisation --
        before, none of those methods existed on this class at all.
        """
        if not self._docker_available:
            reason = (
                "Docker is unavailable (no CLI on PATH, or no daemon answering). "
                "Network isolation, the 512MB memory cap and the 1-CPU limit are "
                "NOT in effect."
            )
            if self.strict:
                raise SandboxUnavailableError(reason)

            _LOG.warning("DockerSandbox: %s Falling back to SubprocessSandbox.", reason)
            result = SubprocessSandbox(self.working_dir, self.timeout).run(
                command, env, stdin_data
            )
            result.sandbox = SANDBOX_SUBPROCESS_FALLBACK
            return result

        name = _new_container_name()
        docker_cmd = self._build_docker_command(command, name)
        host = SubprocessSandbox(self.working_dir, self.timeout)

        _register_atexit_once()
        _ACTIVE_CONTAINERS.add(name)
        self._containers.add(name)
        try:
            result = host.run(docker_cmd, env, stdin_data)
            result.sandbox = SANDBOX_DOCKER
        except BaseException:
            # KeyboardInterrupt and SystemExit reach here; the CLI is gone but
            # the container is not, so remove it before the exception continues.
            _force_remove_container(name)
            raise
        finally:
            _ACTIVE_CONTAINERS.discard(name)
            self._containers.discard(name)

        if result.timed_out:
            # subprocess.run kills the docker CLI with SIGKILL on timeout, which
            # leaves the container running. --rm does not cover this.
            _force_remove_container(name)

        return result

    def run_tests(
        self, runner: str = "pytest", extra_args: Optional[list[str]] = None
    ) -> ExecutionResult:
        """
        Overridden because DOCKER_RUNNERS names the interpreter inside the image
        (`python`), not the one running the agent (`sys.executable`).
        """
        if not self._docker_available and not self.strict:
            return super().run_tests(runner, extra_args)

        runner_cmd = DOCKER_RUNNERS.get(runner) or ["python", "-m", "pytest"]
        return self.run(runner_cmd + (extra_args or []))

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
