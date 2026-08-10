"""
Tests for GitHub Actions token scope (.github/workflows/).

`agent-fix.yml` declared its own permissions; `import-guard.yml` and
`prettier.yml` did not, so they inherited the repository default — which for
repositories created before 2023 is read/write on everything.

Neither needs write. The guard imports modules and runs `--help`; the format job
runs `npm ci` and a check. `npm ci` executes lifecycle scripts from the
lockfile, and on the `push` trigger that runs with a real token for this
repository rather than the read-only one a fork PR receives.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKFLOWS = ROOT / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8"))


def only_job(name):
    return list(load(name)["jobs"].values())[0]


@pytest.mark.parametrize("name", ["import-guard", "prettier", "agent-fix"])
def test_every_workflow_declares_permissions(name):
    """Inheriting means the scope depends on a repository setting nobody can
    see from the code."""
    assert "permissions" in only_job(name)


@pytest.mark.parametrize("name", ["import-guard", "prettier"])
def test_the_read_only_jobs_have_no_write(name):
    permissions = only_job(name)["permissions"]

    assert permissions == {"contents": "read"}


def test_agent_fix_keeps_the_write_it_needs():
    """It opens pull requests and comments on issues — narrowing it would
    break the workflow."""
    permissions = only_job("agent-fix")["permissions"]

    assert permissions["contents"] == "write"
    assert permissions["pull-requests"] == "write"


@pytest.mark.parametrize("name", ["import-guard", "prettier", "agent-fix"])
def test_the_workflow_still_parses_and_has_steps(name):
    """A misplaced permissions block would silently detach the steps."""
    job = only_job(name)

    assert isinstance(job.get("steps"), list)
    assert len(job["steps"]) >= 1
