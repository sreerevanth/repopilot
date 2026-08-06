"""
Tests for sandbox environment sanitization (modules/sandbox.py).

The sandbox executes LLM-generated code, so the environment handed to it is a
security boundary. These tests pin that boundary in both directions: secrets
must not survive, and the toolchain variables the runners need must.
"""

import os
import sys
from pathlib import Path

import pytest

# tests/ has no __init__.py and the project ships no pytest config, so the repo
# root is only on sys.path when invoked as `python -m pytest`. Add it explicitly
# so a bare `pytest tests/test_sandbox_env.py` works too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sandbox import (  # noqa: E402
    ALLOWED_ENV_VARS,
    PASSTHROUGH_ENV_VAR,
    _build_safe_env,
)

# Credentials and capability handles that must never reach sandboxed code.
SECRETS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITLAB_TOKEN",
    "DATABASE_URL",
    "REDIS_URL",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "SSH_AUTH_SOCK",
    "KUBECONFIG",
    "NPM_TOKEN",
    "PYPI_TOKEN",
    "SLACK_TOKEN",
    "STRIPE_SECRET_KEY",
    "DOCKER_PASSWORD",
    "AZURE_CLIENT_SECRET",
    "HF_TOKEN",
    "CARGO_REGISTRY_TOKEN",
]

# GitHub Actions runner variables. GITHUB_ENV and GITHUB_PATH name writable
# files that inject env entries and PATH entries into subsequent workflow
# steps, so leaking them escalates "runs code" into "controls the workflow".
ACTIONS_VARS = [
    "GITHUB_ENV",
    "GITHUB_PATH",
    "GITHUB_OUTPUT",
    "ACTIONS_RUNTIME_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_URL",
]

# Variables the supported runners genuinely need.
TOOLCHAIN = {
    "PATH": "/usr/bin",
    "HOME": "/home/dev",
    "TMPDIR": "/tmp",
    "LANG": "en_US.UTF-8",
    "PYTHONPATH": "/src",
    "VIRTUAL_ENV": "/src/.venv",
    "NODE_PATH": "/usr/lib/node_modules",
    "npm_config_cache": "/home/dev/.npm",
    "GOPATH": "/home/dev/go",
    "GOCACHE": "/home/dev/.cache/go-build",
    "GOMODCACHE": "/home/dev/go/pkg/mod",
    "GOROOT": "/usr/local/go",
    "CARGO_HOME": "/home/dev/.cargo",
    "RUSTUP_HOME": "/home/dev/.rustup",
    "GEM_HOME": "/home/dev/.gem",
    "JAVA_HOME": "/usr/lib/jvm/default",
    "SYSTEMROOT": r"C:\Windows",
    "COMSPEC": r"C:\Windows\system32\cmd.exe",
    "CI": "true",
}

# DockerSandbox runs the `docker` CLI *through* SubprocessSandbox.run, so the
# allowlist applies to the CLI. Dropping these breaks Docker Desktop, colima,
# Rancher Desktop and any remote daemon.
DOCKER_CLI = {
    "DOCKER_HOST": "unix:///var/run/docker.sock",
    "DOCKER_CONTEXT": "colima",
    "DOCKER_CONFIG": "/home/dev/.docker",
    "DOCKER_CERT_PATH": "/home/dev/.docker/certs",
    "DOCKER_TLS_VERIFY": "1",
}


@pytest.fixture
def fake_environ(monkeypatch):
    """Replace os.environ with a controlled dict and return it."""

    def _apply(**extra):
        environ = {"PATH": "/usr/bin", "HOME": "/home/dev"}
        environ.update(extra)
        monkeypatch.setattr(os, "environ", environ)
        return environ

    return _apply


@pytest.mark.parametrize("name", SECRETS)
def test_secrets_are_stripped(fake_environ, name):
    fake_environ(**{name: "sensitive-value"})
    assert name not in _build_safe_env()


@pytest.mark.parametrize("name", ACTIONS_VARS)
def test_actions_runner_vars_are_stripped(fake_environ, name):
    """Leaking these lets sandboxed code influence later workflow steps."""
    fake_environ(**{name: "/runner/file"})
    assert name not in _build_safe_env()


@pytest.mark.parametrize("name,value", sorted(TOOLCHAIN.items()))
def test_toolchain_vars_survive(fake_environ, name, value):
    fake_environ(**{name: value})
    assert _build_safe_env().get(name) == value


@pytest.mark.parametrize("name,value", sorted(DOCKER_CLI.items()))
def test_docker_cli_vars_survive(fake_environ, name, value):
    """Regression guard for DockerSandbox, which shells out via SubprocessSandbox."""
    fake_environ(**{name: value})
    assert _build_safe_env().get(name) == value


def test_unknown_vars_are_dropped_by_default(fake_environ):
    """The default is deny: a brand-new service is protected without a code change."""
    fake_environ(SOME_FUTURE_SERVICE_TOKEN="secret", ANOTHER_UNKNOWN_VAR="x")
    env = _build_safe_env()
    assert "SOME_FUTURE_SERVICE_TOKEN" not in env
    assert "ANOTHER_UNKNOWN_VAR" not in env


def test_path_is_always_present(fake_environ):
    fake_environ()
    assert "PATH" in _build_safe_env()


def test_matching_is_case_insensitive(fake_environ):
    """Windows upper-cases environment keys; npm_config_cache must still match."""
    fake_environ(NPM_CONFIG_CACHE="/cache", Path="/usr/bin")
    env = _build_safe_env()
    assert env.get("NPM_CONFIG_CACHE") == "/cache"
    assert env.get("Path") == "/usr/bin"


def test_passthrough_hook_widens_the_allowlist(fake_environ):
    fake_environ(
        **{
            PASSTHROUGH_ENV_VAR: "GOPROXY, MY_INTERNAL_VAR",
            "GOPROXY": "https://proxy.internal",
            "MY_INTERNAL_VAR": "value",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
        }
    )
    env = _build_safe_env()
    assert env.get("GOPROXY") == "https://proxy.internal"
    assert env.get("MY_INTERNAL_VAR") == "value"
    # Widening the hatch must not widen anything else.
    assert "ANTHROPIC_API_KEY" not in env


def test_passthrough_hook_ignores_blank_entries(fake_environ):
    fake_environ(**{PASSTHROUGH_ENV_VAR: " , ,, ", "UNRELATED": "x"})
    env = _build_safe_env()
    assert "UNRELATED" not in env
    assert "" not in env


def test_extra_env_is_a_trusted_caller_override(fake_environ):
    """extra_env is applied after filtering; it is documented as trusted-caller-only."""
    fake_environ()
    env = _build_safe_env({"PYTEST_ADDOPTS": "-q", "RUNNER_SPECIFIC": "1"})
    assert env["PYTEST_ADDOPTS"] == "-q"
    assert env["RUNNER_SPECIFIC"] == "1"


def test_extra_env_does_not_mutate_os_environ(fake_environ):
    environ = fake_environ()
    _build_safe_env({"RUNNER_SPECIFIC": "1"})
    assert "RUNNER_SPECIFIC" not in environ


@pytest.mark.parametrize(
    "marker", ["TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "APIKEY", "AUTH"]
)
def test_allowlist_contains_no_credential_shaped_names(marker):
    """Guards future widening: nothing credential-shaped belongs in the defaults."""
    offenders = [n for n in ALLOWED_ENV_VARS if marker in n.upper().replace("_", "")]
    assert offenders == [], f"credential-shaped name(s) in ALLOWED_ENV_VARS: {offenders}"
