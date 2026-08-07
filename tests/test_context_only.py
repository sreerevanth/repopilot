"""
Tests for --context-only (modules/agent_loop.py).

`--dry-run` shows the changes the model proposed, which means the API call has
already been paid for by the time anything is printed. `--context-only` stops
before that call, so you can check which files were selected for nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"


def source() -> str:
    # Explicit encoding: the default is locale-dependent and mangles this
    # module's box-drawing section comments on a default Windows install.
    return AGENT_LOOP.read_text(encoding="utf-8")


def test_context_only_defaults_to_off():
    assert AgentConfig(repo_root=".", task="t").context_only is False


def test_dry_run_is_unaffected():
    assert AgentConfig(repo_root=".", task="t").dry_run is False


def test_it_returns_before_the_llm_is_called():
    """The whole point: no API call, so no cost."""
    text = source()
    guard = text.index("if cfg.context_only:")
    llm_call = text.index("self.llm.initial_request")

    assert guard < llm_call


def test_it_runs_after_the_context_is_built():
    """There would be nothing to print otherwise."""
    text = source()
    assert text.index("context_str = context.render()") < text.index("if cfg.context_only:")


def test_it_prints_the_compiled_context():
    text = source()
    block = text[text.index("if cfg.context_only:"):][:900]

    assert "print(context_str)" in block


def test_it_reports_file_count_and_size():
    block = source()[source().index("if cfg.context_only:"):][:900]

    assert "len(context.files)" in block
    assert "total_chars" in block


def test_it_reports_outlined_files_when_present():
    """#69 can degrade a file to a signature outline; that is worth surfacing."""
    block = source()[source().index("if cfg.context_only:"):][:900]

    assert "outlined" in block


def test_it_uses_a_distinct_outcome():
    """`dry_run` already means "the model ran and here is what it wanted"."""
    block = source()[source().index("if cfg.context_only:"):][:1200]

    assert 'outcome="context_only"' in block


def test_the_outcome_is_documented():
    assert "context_only" in source().split("outcome: str")[1][:200]


def test_it_says_no_call_was_made():
    """So the user is not left wondering whether they were charged."""
    block = source()[source().index("if cfg.context_only:"):][:1200]

    assert "No API call was made" in block


def test_no_files_are_touched():
    """It returns before apply_changes; nothing in the block writes."""
    block = source()[source().index("if cfg.context_only:"):][:1200]

    assert "apply_changes" not in block
    assert "_commit_changes" not in block
