"""
Tests for GITHUB_OUTPUT writing and the workflow that consumes it
(main.py, .github/workflows/agent-fix.yml).

Two problems on the same path.

`write_github_output` used a fixed `EOF` delimiter for multiline values. A value
containing a line that is exactly `EOF` closed the heredoc early, and everything
after it was parsed by the runner as further `key=value` outputs — so a crafted
`final_message` could set `pr_url` and `outcome` to anything.

The workflow then interpolated those outputs straight into a `github-script`
body, where GitHub substitutes them into the script text before Node parses it.
A double quote closed the string literal and the rest executed, with
`contents: write`, `pull-requests: write` and both secrets in scope.

`final_message` is reachable: `agent_loop` sets it from `str(exception)` on the
ingestion-failure path.
"""

import os
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import write_github_output  # noqa: E402

WORKFLOW = ROOT / ".github" / "workflows" / "agent-fix.yml"


@pytest.fixture
def output_file(tmp_path, monkeypatch):
    path = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(path))
    return path


def parse_like_the_runner(text):
    """Keys the Actions runner would take from this file."""
    keys, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if "<<" in lines[i]:
            key, delimiter = lines[i].split("<<", 1)
            keys.append(key)
            i += 1
            while i < len(lines) and lines[i] != delimiter:
                i += 1
        elif "=" in lines[i]:
            keys.append(lines[i].split("=", 1)[0])
        i += 1
    return keys


# ── the delimiter ─────────────────────────────────────────────────────────


def test_a_multiline_value_uses_a_random_delimiter(output_file):
    write_github_output({"final_message": "line one\nline two"})

    assert re.search(r"ghadelimiter_[0-9a-f-]{36}", output_file.read_text())


def test_a_value_containing_EOF_cannot_close_the_heredoc(output_file):
    """The exact attack: EOF on its own line used to end the value."""
    write_github_output({"final_message": "failed\nEOF\npr_url=https://evil.example"})

    assert "pr_url" not in parse_like_the_runner(output_file.read_text())


def test_a_crafted_message_cannot_forge_success(output_file):
    """
    Forging outcome=success is what made this more than cosmetic: the workflow
    posts "I have successfully fixed the issue" with the injected pr_url.
    """
    attack = "failed\nEOF\noutcome=success\npr_url=https://evil.example"
    write_github_output({"final_message": attack, "outcome": "error"})

    keys = parse_like_the_runner(output_file.read_text())

    assert keys == ["final_message", "outcome"]


def test_each_write_gets_a_fresh_delimiter(output_file):
    write_github_output({"a": "x\ny", "b": "p\nq"})
    found = re.findall(r"ghadelimiter_[0-9a-f-]{36}", output_file.read_text())

    assert len(set(found)) == 2


def test_a_single_line_value_is_written_plainly(output_file):
    """No heredoc where none is needed; the old behaviour for the common case."""
    write_github_output({"outcome": "success"})

    assert output_file.read_text() == "outcome=success\n"


def test_no_github_output_is_not_an_error(monkeypatch):
    """Running outside Actions is the normal case."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    write_github_output({"outcome": "success"})


# ── the workflow ──────────────────────────────────────────────────────────


def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def comment_step():
    steps = workflow()["jobs"]["agent-fix"]["steps"]
    return next(s for s in steps if "github-script" in str(s.get("uses", "")))


def test_the_script_interpolates_nothing():
    """
    GitHub substitutes ${{ }} into the script text before Node parses it, so
    any expression there is source rather than data.
    """
    assert "${{" not in comment_step()["with"]["script"]


def test_the_outputs_arrive_as_environment_variables():
    env = comment_step()["env"]

    assert set(env) == {"OUTCOME", "PR_URL", "RUN_ID", "FINAL_MESSAGE"}


def test_the_script_reads_them_from_process_env():
    script = comment_step()["with"]["script"]

    for name in ("OUTCOME", "PR_URL", "RUN_ID", "FINAL_MESSAGE"):
        assert f"process.env.{name}" in script


def test_the_task_step_still_uses_env_for_the_issue_body():
    """
    That step already got this right and must keep it — the issue body is the
    most obviously attacker-controlled value in the workflow.
    """
    steps = workflow()["jobs"]["agent-fix"]["steps"]
    task_step = next(s for s in steps if "AGENT_TASK" in str(s.get("env", "")))

    assert "${{" not in task_step["run"].replace(
        "${{ github.event.repository.default_branch }}", ""
    )


def test_the_workflow_is_still_valid_yaml():
    assert workflow()["jobs"]["agent-fix"]["permissions"]["contents"] == "write"
