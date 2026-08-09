"""
Tests for --no-commit (modules/agent_loop.py).

Leaves successful changes staged instead of committing them, so the author can
inspect the diff and write their own message.

Staging and committing were already separate steps, so this is an early return
between them rather than a new code path — which is why the index ends up in
exactly the state a normal run would have produced just before committing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
MAIN = Path(__file__).resolve().parents[1] / "main.py"


def loop_source():
    return AGENT_LOOP.read_text(encoding="utf-8")


# ── the flag ──────────────────────────────────────────────────────────────


def test_it_is_a_config_field():
    assert "no_commit" in {f.name for f in AgentConfig.__dataclass_fields__.values()}


def test_it_is_off_by_default():
    """Existing behaviour is unchanged for anyone not passing it."""
    assert AgentConfig(repo_root=".", task="t").no_commit is False


def test_main_registers_the_flag():
    assert '"--no-commit"' in MAIN.read_text(encoding="utf-8")


def test_main_passes_it_to_the_config():
    assert "no_commit=args.no_commit" in MAIN.read_text(encoding="utf-8")


# ── where it takes effect ─────────────────────────────────────────────────


def test_staging_still_happens():
    """
    The whole point: the changes end up in the index. Returning before staging
    would leave them merely unsaved, which is not what was asked for.
    """
    source = loop_source()

    assert source.index("stage = self.git.stage_files(changed_paths)") < \
        source.index("if self.config.no_commit:")


def test_it_returns_before_committing():
    source = loop_source()

    assert source.index("if self.config.no_commit:") < \
        source.index("commit = self.git.commit(msg,")


def test_it_reports_success():
    """
    Staging without committing is the requested outcome, not a failure — the
    run should not be marked failed for doing what it was told.
    """
    source = loop_source()
    block = source[source.index("if self.config.no_commit:"):][:600]

    assert "return True" in block


def test_the_user_is_told_what_to_do_next():
    """
    Silence here looks like the commit was forgotten rather than deliberately
    skipped.
    """
    source = loop_source()
    block = source[source.index("if self.config.no_commit:"):][:600]

    assert "git diff --cached" in block


def test_the_no_changes_case_still_short_circuits():
    """An iteration that changed nothing has nothing to stage or report."""
    source = loop_source()

    assert source.index("No changes to commit") < \
        source.index("if self.config.no_commit:")
