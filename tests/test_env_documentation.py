"""
Tests for environment variable documentation (.env.example, README, .gitignore).

The code reads six user-facing environment variables. Three were absent from the
README — `OPENAI_API_KEY`, `GEMINI_API_KEY` and `GH_TOKEN` — even though
`--provider openai` and `--provider gemini` are both offered in `--help` and
neither works without one.

The usual fix, a committed `.env.example`, could not be added: `.gitignore` had
`.env.*`, which matched the template as well as the real file.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

USER_FACING = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "NO_COLOR",
]


# ── the template is committable ───────────────────────────────────────────


def test_env_example_exists():
    assert (ROOT / ".env.example").exists()


def test_env_example_is_not_ignored():
    """`.env.*` matched it, so it could not be committed without the negation."""
    result = subprocess.run(
        ["git", "check-ignore", ".env.example"],
        capture_output=True, text=True, cwd=ROOT,
    )

    assert result.returncode != 0, ".env.example is still gitignored"


def test_the_real_env_file_is_still_ignored():
    """The negation must not open a hole for the file it was protecting."""
    result = subprocess.run(
        ["git", "check-ignore", ".env"],
        capture_output=True, text=True, cwd=ROOT,
    )

    assert result.returncode == 0, ".env is no longer ignored"


@pytest.mark.parametrize("variant", [".env.local", ".env.production"])
def test_other_env_variants_are_still_ignored(variant):
    result = subprocess.run(
        ["git", "check-ignore", variant],
        capture_output=True, text=True, cwd=ROOT,
    )

    assert result.returncode == 0


# ── the template has no real values ───────────────────────────────────────


def test_the_template_carries_no_credentials():
    """A template with a real key in it is worse than no template."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert not re.search(r"sk-ant-api03-[A-Za-z0-9_-]{20,}", text)
    assert not re.search(r"ghp_[A-Za-z0-9]{20,}", text)


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"])
def test_the_provider_keys_are_listed_empty(name):
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert f"{name}=" in text
    assert not re.search(rf"^{name}=\S", text, re.M), f"{name} has a value"


# ── the README covers them ────────────────────────────────────────────────


@pytest.mark.parametrize("name", USER_FACING)
def test_the_readme_documents_the_variable(name):
    """
    OPENAI_API_KEY, GEMINI_API_KEY and GH_TOKEN were the three missing ones,
    and two of them gate provider flags that --help already advertises.
    """
    assert name in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", USER_FACING)
def test_the_template_documents_the_variable(name):
    assert name in (ROOT / ".env.example").read_text(encoding="utf-8")


def test_the_readme_code_fences_are_balanced():
    """A stray fence swallows the rest of the document when rendered."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert text.count("```") % 2 == 0
