"""
Tests for file-based prompts (modules/llm_client.py, prompts/).

The prompts moved out of Python so they can be edited and diffed. Two things
must hold: the text the model receives is unchanged, and a missing or broken
prompts directory falls back to the built-in text rather than leaving the agent
with no prompt at all.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import llm_client  # noqa: E402
from modules.llm_client import (  # noqa: E402
    PROMPT_DIR_ENV_VAR,
    _BUILTIN_RETRY_PROMPT,
    _BUILTIN_SYSTEM_PROMPT,
    _BUILTIN_TASK_PROMPT,
    load_prompt,
    prompt_dir,
)

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

PAIRS = [
    ("system", _BUILTIN_SYSTEM_PROMPT),
    ("initial", _BUILTIN_TASK_PROMPT),
    ("retry", _BUILTIN_RETRY_PROMPT),
]


# ── the files match the built-ins ─────────────────────────────────────────


@pytest.mark.parametrize("name,builtin", PAIRS)
def test_file_is_byte_identical_to_the_builtin(name, builtin):
    """
    The whole point is that behaviour does not change. A drift here would alter
    what the model sees while looking like a pure refactor.
    """
    assert (PROMPTS / f"{name}.txt").read_text(encoding="utf-8") == builtin


@pytest.mark.parametrize("name,_", PAIRS)
def test_every_prompt_file_exists(name, _):
    assert (PROMPTS / f"{name}.txt").is_file()


def test_output_schema_survived_the_move():
    assert "OUTPUT FORMAT (STRICT" in llm_client.SYSTEM_PROMPT
    assert '"action"' in llm_client.SYSTEM_PROMPT


@pytest.mark.parametrize(
    "template,fields",
    [
        ("TASK_PROMPT_TEMPLATE", ["{task}", "{context}"]),
        (
            "RETRY_PROMPT_TEMPLATE",
            ["{task}", "{context}", "{stdout}", "{stderr}", "{exit_code}",
             "{previous_changes_summary}"],
        ),
    ],
)
def test_placeholders_are_intact(template, fields):
    """A dropped placeholder would raise KeyError only at call time."""
    text = getattr(llm_client, template)
    for field in fields:
        assert field in text


def test_templates_still_format():
    llm_client.TASK_PROMPT_TEMPLATE.format(task="t", context="c")
    llm_client.RETRY_PROMPT_TEMPLATE.format(
        task="t", context="c", stdout="o", stderr="e",
        exit_code=1, previous_changes_summary="s",
    )


# ── falling back ──────────────────────────────────────────────────────────


def test_missing_file_falls_back(tmp_path):
    assert load_prompt("system", _BUILTIN_SYSTEM_PROMPT) is not None
    assert load_prompt("does-not-exist", "fallback text") == "fallback text"


def test_missing_directory_falls_back(monkeypatch):
    monkeypatch.setenv(PROMPT_DIR_ENV_VAR, "/definitely/not/a/directory")
    assert load_prompt("system", _BUILTIN_SYSTEM_PROMPT) == _BUILTIN_SYSTEM_PROMPT


def test_empty_file_falls_back(tmp_path, monkeypatch):
    """An empty prompt would be accepted by the API and produce nonsense."""
    (tmp_path / "system.txt").write_text("   \n\n")
    monkeypatch.setenv(PROMPT_DIR_ENV_VAR, str(tmp_path))

    assert load_prompt("system", _BUILTIN_SYSTEM_PROMPT) == _BUILTIN_SYSTEM_PROMPT


def test_a_directory_where_a_file_should_be_falls_back(tmp_path, monkeypatch):
    (tmp_path / "system.txt").mkdir()
    monkeypatch.setenv(PROMPT_DIR_ENV_VAR, str(tmp_path))

    assert load_prompt("system", _BUILTIN_SYSTEM_PROMPT) == _BUILTIN_SYSTEM_PROMPT


def test_module_import_survives_a_missing_prompt_dir(monkeypatch):
    """Reimporting with no prompts must not raise."""
    monkeypatch.setenv(PROMPT_DIR_ENV_VAR, "/definitely/not/a/directory")
    reloaded = importlib.reload(llm_client)

    assert reloaded.SYSTEM_PROMPT == _BUILTIN_SYSTEM_PROMPT
    monkeypatch.delenv(PROMPT_DIR_ENV_VAR)
    importlib.reload(llm_client)


# ── overriding ────────────────────────────────────────────────────────────


def test_env_var_redirects_the_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(PROMPT_DIR_ENV_VAR, str(tmp_path))
    assert prompt_dir() == tmp_path


def test_default_directory_is_the_repo_prompts_folder(monkeypatch):
    monkeypatch.delenv(PROMPT_DIR_ENV_VAR, raising=False)
    assert prompt_dir() == PROMPTS


def test_a_custom_prompt_is_used(tmp_path, monkeypatch):
    (tmp_path / "system.txt").write_text("you are a haiku poet\n")
    monkeypatch.setenv(PROMPT_DIR_ENV_VAR, str(tmp_path))

    assert load_prompt("system", _BUILTIN_SYSTEM_PROMPT) == "you are a haiku poet\n"


def test_files_are_read_as_utf8(tmp_path, monkeypatch):
    """Locale-dependent reads mangle non-ASCII on a default Windows install."""
    (tmp_path / "system.txt").write_text(
        "respond precisely — no waffle\n", encoding="utf-8"
    )
    monkeypatch.setenv(PROMPT_DIR_ENV_VAR, str(tmp_path))

    assert "—" in load_prompt("system", _BUILTIN_SYSTEM_PROMPT)
