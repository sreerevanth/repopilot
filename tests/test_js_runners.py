"""
Tests for the vitest and jest runners (modules/sandbox.py).

The table entries carry two decisions that are easy to undo by accident, so
both are pinned here: `--no-install`, which stops npx fetching a missing
package from the registry, and vitest's `run` subcommand, which stops it
starting a watch server that would sit until the sandbox timeout.

The integration tests skip cleanly when node or the packages are absent, so
this file is safe in a CI job that only installs Python.
"""

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.sandbox import (  # noqa: E402
    ALLOWED_RUNNERS,
    DOCKER_RUNNERS,
    SubprocessSandbox,
    _resolve_runner,
)

JS_RUNNERS = ("vitest", "jest")


# ── table shape ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("runner", JS_RUNNERS)
def test_runner_is_registered_in_both_tables(runner):
    assert runner in ALLOWED_RUNNERS
    assert runner in DOCKER_RUNNERS


@pytest.mark.parametrize("runner", JS_RUNNERS)
def test_subprocess_and_docker_commands_match(runner):
    """A runner that behaves differently inside Docker is a debugging trap."""
    assert ALLOWED_RUNNERS[runner] == DOCKER_RUNNERS[runner]


def test_runner_tables_cover_the_same_names():
    assert set(ALLOWED_RUNNERS) == set(DOCKER_RUNNERS)


@pytest.mark.parametrize("runner", JS_RUNNERS)
def test_runner_goes_through_npx(runner):
    assert ALLOWED_RUNNERS[runner][0] == "npx"


# ── the two decisions worth pinning ───────────────────────────────────────


@pytest.mark.parametrize("runner", JS_RUNNERS)
def test_npx_may_not_auto_install(runner):
    """
    Without --no-install, npx downloads a missing package from the registry.
    That would breach DockerSandbox's --network=none and surprise offline users.
    """
    assert "--no-install" in ALLOWED_RUNNERS[runner]
    assert "--no-install" in DOCKER_RUNNERS[runner]


def test_vitest_uses_the_run_subcommand():
    """Bare `vitest` can start a watch server and hang until timeout_seconds."""
    cmd = ALLOWED_RUNNERS["vitest"]
    assert "run" in cmd
    assert cmd.index("run") > cmd.index("vitest")


def test_existing_runners_are_untouched():
    assert ALLOWED_RUNNERS["pytest"] == [sys.executable, "-m", "pytest"]
    assert ALLOWED_RUNNERS["npm_test"] == ["npm", "test", "--"]
    assert ALLOWED_RUNNERS["go"] == ["go", "test", "./..."]


# ── resolution ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("runner", JS_RUNNERS)
def test_resolves_when_npx_is_on_path(runner, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda exe: "/usr/bin/npx")
    assert _resolve_runner(runner) == ALLOWED_RUNNERS[runner]


@pytest.mark.parametrize("runner", JS_RUNNERS)
def test_returns_none_when_npx_is_missing(runner, monkeypatch):
    """run_tests turns this into exit_code -3 rather than crashing."""
    monkeypatch.setattr(shutil, "which", lambda exe: None)
    assert _resolve_runner(runner) is None


@pytest.mark.parametrize("runner", JS_RUNNERS)
def test_missing_npx_yields_a_clean_failure(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda exe: None)
    result = SubprocessSandbox(str(tmp_path)).run_tests(runner)
    assert result.exit_code == -3
    assert result.success is False
    assert runner in result.stderr


# ── integration (skipped when the toolchain is absent) ────────────────────
#
# These install the real packages once per session into a temp project. They
# need node and network, so they skip cleanly on a Python-only CI job rather
# than failing. Without a real install the runners can never be exercised --
# an availability probe run inside an empty directory would always say "no".


@pytest.fixture(scope="session")
def js_project(tmp_path_factory):
    """A temp project with vitest and jest installed, or a skip."""
    if shutil.which("npm") is None:
        pytest.skip("npm not available")

    project = tmp_path_factory.mktemp("js_runners")
    (project / "package.json").write_text('{"name":"t","private":true}')

    try:
        proc = subprocess.run(
            ["npm", "install", "--no-audit", "--no-fund", "vitest", "jest"],
            cwd=str(project), capture_output=True, text=True, timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"npm install failed: {exc}")

    if proc.returncode != 0:
        pytest.skip(f"npm install failed: {proc.stderr[-300:]}")

    return project


def _case_dir(js_project):
    """
    A fresh subdirectory of the installed project, so one test's failing file
    cannot leak into another. A subdirectory rather than a symlink on purpose:
    node resolves node_modules by walking up the tree, and symlinks need
    Developer Mode or admin rights on Windows.
    """
    project = js_project / f"case_{uuid.uuid4().hex[:8]}"
    project.mkdir()
    return project


@pytest.fixture
def vitest_project(js_project):
    project = _case_dir(js_project)
    (project / "package.json").write_text(
        '{"name":"c","private":true,"type":"module"}'
    )
    (project / "ok.test.js").write_text(
        "import { test, expect } from 'vitest';\n"
        "test('adds', () => { expect(1 + 1).toBe(2); });\n"
    )
    return project


@pytest.fixture
def jest_project(js_project):
    project = _case_dir(js_project)
    (project / "package.json").write_text('{"name":"c","private":true}')
    (project / "ok.test.js").write_text(
        "test('adds', () => { expect(1 + 1).toBe(2); });\n"
    )
    return project


def test_vitest_passing_suite(vitest_project):
    sandbox = SubprocessSandbox(str(vitest_project), timeout_seconds=300)
    result = sandbox.run_tests("vitest")
    assert result.success is True
    assert result.timed_out is False


def test_vitest_failing_suite_is_reported_as_failure(vitest_project):
    (vitest_project / "bad.test.js").write_text(
        "import { test, expect } from 'vitest';\n"
        "test('fails', () => { expect(1).toBe(2); });\n"
    )
    sandbox = SubprocessSandbox(str(vitest_project), timeout_seconds=300)
    result = sandbox.run_tests("vitest")
    assert result.exit_code != 0
    assert result.success is False
    # The agent feeds this back to the LLM on the next iteration.
    assert "fails" in (result.stdout + result.stderr)


def test_vitest_does_not_hang_in_watch_mode(vitest_project):
    """`run` must make vitest exit; a watch server would trip timed_out."""
    sandbox = SubprocessSandbox(str(vitest_project), timeout_seconds=300)
    result = sandbox.run_tests("vitest")
    assert result.timed_out is False


def test_jest_passing_suite(jest_project):
    result = SubprocessSandbox(str(jest_project), timeout_seconds=300).run_tests("jest")
    assert result.success is True
    assert result.timed_out is False


def test_jest_failing_suite_is_reported_as_failure(jest_project):
    (jest_project / "bad.test.js").write_text(
        "test('fails', () => { expect(1).toBe(2); });\n"
    )
    result = SubprocessSandbox(str(jest_project), timeout_seconds=300).run_tests("jest")
    assert result.exit_code != 0
    assert result.success is False
    assert "fails" in (result.stdout + result.stderr)


def test_extra_args_are_forwarded(vitest_project):
    result = SubprocessSandbox(str(vitest_project), timeout_seconds=300).run_tests(
        "vitest", ["--reporter=basic"]
    )
    assert "--reporter=basic" in result.command
