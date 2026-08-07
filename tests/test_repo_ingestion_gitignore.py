"""
Tests for .gitignore-aware repository ingestion (modules/repo_ingestion.py).

Ingested file content is sent verbatim to the LLM, so what the walker picks up
is a disclosure boundary, not just a context-budget concern. These tests pin
both halves: gitignored paths stay out, and credential-bearing filenames stay
out whether or not .gitignore happens to mention them.
"""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import repo_ingestion  # noqa: E402
from modules.repo_ingestion import (  # noqa: E402
    _is_secret_file,
    ingest_repository,
    load_ignore_spec,
)

pathspec = pytest.importorskip("pathspec", reason="pathspec drives the ignore spec")


@pytest.fixture
def repo(tmp_path):
    """A small project with a .gitignore covering a file, a glob and a dir."""
    (tmp_path / ".gitignore").write_text(
        textwrap.dedent(
            """\
            .env
            *.log
            build_out/
            secrets/
            """
        )
    )
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-secret\n")
    (tmp_path / "app.log").write_text("noise\n")
    (tmp_path / "main.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# project\n")

    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "creds.txt").write_text("AWS_SECRET=leaked\n")
    (tmp_path / "build_out").mkdir()
    (tmp_path / "build_out" / "bundle.txt").write_text("artifact\n")

    return tmp_path


def _paths(result):
    return {f.path for f in result.files}


def test_gitignored_file_is_excluded(repo):
    assert "app.log" not in _paths(ingest_repository(str(repo)))


def test_gitignored_directory_is_excluded(repo):
    paths = _paths(ingest_repository(str(repo)))
    assert "secrets/creds.txt" not in paths
    assert "build_out/bundle.txt" not in paths


def test_tracked_files_are_still_ingested(repo):
    paths = _paths(ingest_repository(str(repo)))
    assert "main.py" in paths
    assert "README.md" in paths


def test_gitignore_itself_remains_visible(repo):
    """It is useful context and holds no secrets."""
    assert ".gitignore" in _paths(ingest_repository(str(repo)))


def test_env_file_never_reaches_the_model(repo):
    result = ingest_repository(str(repo))
    assert ".env" not in _paths(result)
    assert all("sk-ant-secret" not in f.content for f in result.files)


def test_env_excluded_even_without_a_gitignore(tmp_path):
    """The secret filter must not depend on the project having a .gitignore."""
    (tmp_path / ".env").write_text("DB_PASSWORD=hunter2\n")
    (tmp_path / "main.py").write_text("print('hi')\n")

    result = ingest_repository(str(tmp_path))
    assert ".env" not in _paths(result)
    assert "main.py" in _paths(result)


@pytest.mark.parametrize(
    "name",
    [
        ".env", ".env.local", ".env.production", "prod.env",
        "server.pem", "private.key", "id_rsa", "id_ed25519",
        ".npmrc", ".pypirc", ".netrc", "credentials", "credentials.json",
    ],
)
def test_secret_filenames_are_recognised(name):
    assert _is_secret_file(name) is True


@pytest.mark.parametrize(
    "name", ["main.py", "environment.md", "keyboard.js", ".gitignore"]
)
def test_ordinary_filenames_are_not_treated_as_secrets(name):
    assert _is_secret_file(name) is False


def test_skipped_entries_record_the_reason(repo):
    skipped = ingest_repository(str(repo)).skipped
    assert any("app.log" in s and "gitignored" in s for s in skipped)


def test_missing_gitignore_yields_no_spec(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')\n")
    assert load_ignore_spec(str(tmp_path)) is None


def test_malformed_patterns_do_not_break_ingestion(tmp_path):
    (tmp_path / ".gitignore").write_text("[unclosed\n***/bad\n")
    (tmp_path / "main.py").write_text("print('hi')\n")
    # Must not raise; worst case the spec is dropped and defaults apply.
    assert "main.py" in _paths(ingest_repository(str(tmp_path)))


def test_falls_back_cleanly_when_pathspec_is_absent(repo, monkeypatch):
    """An un-reinstalled checkout must still ingest, just without .gitignore."""
    monkeypatch.setattr(repo_ingestion, "pathspec", None)

    result = ingest_repository(str(repo))
    paths = _paths(result)

    assert load_ignore_spec(str(repo)) is None
    assert "main.py" in paths          # ingestion still works
    assert ".env" not in paths         # secret filter is independent of pathspec
    assert "app.log" in paths          # .gitignore is genuinely not applied
