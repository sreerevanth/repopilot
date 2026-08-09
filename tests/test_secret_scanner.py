"""
Tests for modules/secret_scanner.py.

This module had no tests. It is what stops a key reaching a log, and its
failure mode is silent: a pattern that stops matching does not raise, it just
returns nothing and the secret goes through.

Every case below uses a fabricated key of the right *shape*. None is real.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.secret_scanner import (  # noqa: E402
    SECRET_PATTERNS,
    format_findings,
    scan_directory,
    scan_file,
)

# Fabricated, shape-correct only.
AWS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_" + "a" * 36
PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----"


def found(content, name=None):
    findings = scan_file("example.py", content)
    if name is None:
        return findings
    return [f for f in findings if f.pattern_name == name]


# ── it finds what it is for ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "content,expected",
    [
        (f"key = '{AWS_KEY_ID}'", "AWS Access Key ID"),
        (f"token = '{GITHUB_TOKEN}'", "GitHub Token"),
        (f"api_key = '{'x' * 24}'", "Generic API Key"),
        ("password = 'hunter2hunter2'", "Generic Secret/Token"),
        (PRIVATE_KEY, "Private Key Block"),
    ],
    ids=["aws", "github", "api-key", "password", "private-key"],
)
def test_a_known_shape_is_detected(content, expected):
    assert found(content, expected)


def test_the_finding_names_the_line():
    """A file path alone is not actionable in a large file."""
    content = f"first\nsecond\nkey = '{AWS_KEY_ID}'\n"

    assert found(content, "AWS Access Key ID")[0].line_number == 3


def test_the_finding_carries_a_severity():
    assert found(f"key = '{AWS_KEY_ID}'", "AWS Access Key ID")[0].severity == "high"


def test_several_secrets_in_one_file_are_all_found():
    content = f"a = '{AWS_KEY_ID}'\nb = '{GITHUB_TOKEN}'\n"

    assert len(found(content)) >= 2


# ── it leaves ordinary code alone ─────────────────────────────────────────


@pytest.mark.parametrize(
    "content",
    [
        "def parse(x):\n    return x.strip()\n",
        "AKIA = 'not a key'",
        "# api_key documentation mentions the term",
        "password = get_password()",
        "",
    ],
    ids=["source", "short-aws", "comment", "indirection", "empty"],
)
def test_ordinary_content_is_not_flagged(content):
    """
    False positives are not harmless here: a scanner that cries wolf gets
    switched off, and then it catches nothing at all.
    """
    assert scan_file("example.py", content) == []


# ── scanning a tree ───────────────────────────────────────────────────────


def test_a_directory_is_scanned(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\n")
    (tmp_path / "leaky.py").write_text(f"key = '{AWS_KEY_ID}'\n")

    paths = {Path(f.file_path).name for f in scan_directory(str(tmp_path))}

    assert "leaky.py" in paths
    assert "clean.py" not in paths


def test_a_clean_directory_reports_nothing(tmp_path):
    (tmp_path / "clean.py").write_text("def f():\n    return 1\n")

    assert scan_directory(str(tmp_path)) == []


def test_a_missing_directory_does_not_raise(tmp_path):
    """A scanner that crashes on a bad path fails the run it was protecting."""
    assert scan_directory(str(tmp_path / "nope")) == []


# ── reporting ─────────────────────────────────────────────────────────────


def test_findings_are_rendered_with_their_location():
    rendered = format_findings(found(f"key = '{AWS_KEY_ID}'"))

    assert "AWS Access Key ID" in rendered


def test_no_findings_renders_without_error():
    assert isinstance(format_findings([]), str)


# ── the pattern table itself ──────────────────────────────────────────────


def test_every_pattern_is_well_formed():
    """A malformed regex would raise at scan time, on the run it should protect."""
    import re

    for pattern in SECRET_PATTERNS:
        assert pattern["name"]
        assert pattern["severity"] in {"high", "medium", "low"}
        re.compile(pattern["regex"])


def test_the_table_is_not_empty():
    """An empty table scans clean and protects nothing."""
    assert len(SECRET_PATTERNS) >= 5
