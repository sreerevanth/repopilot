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


# ── it must not need a provider ───────────────────────────────────────────


def test_the_llm_client_is_built_lazily():
    """
    Constructing the client in __init__ made --context-only require the provider
    SDK and a valid API key — for a flag whose entire point is that no request
    is made. The client is now built on first access.
    """
    text = source()

    assert "self._llm: Optional[LLMClient] = None" in text
    assert "def llm(self) -> LLMClient:" in text


def test_context_only_runs_without_an_api_key(tmp_path, monkeypatch):
    from modules.agent_loop import AgentConfig, AutonomousAgent

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "mod.py").write_text("def parse(x):\n    return x\n")

    result = AutonomousAgent(
        AgentConfig(
            repo_root=str(tmp_path), task="fix parse",
            context_only=True, git_enabled=False,
        )
    ).run()

    assert result.outcome == "context_only"


def test_context_only_runs_without_the_provider_sdk(tmp_path, monkeypatch):
    """A reader inspecting context selection should not need to install a SDK."""
    import builtins

    from modules.agent_loop import AgentConfig, AutonomousAgent

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("anthropic not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "mod.py").write_text("def parse(x):\n    return x\n")

    result = AutonomousAgent(
        AgentConfig(
            repo_root=str(tmp_path), task="fix parse",
            context_only=True, git_enabled=False,
        )
    ).run()

    assert result.outcome == "context_only"
