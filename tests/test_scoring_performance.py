"""
Tests for context scoring performance (modules/context_builder.py).

`_score_file` runs against every file in the repository on every iteration, and
the import-extraction regexes dominated it: 4.5ms of 6.3ms on a 320KB minified
bundle. re.MULTILINE on a single enormous line scans the whole blob looking for
line anchors that cannot exist.

The optimisation must not change which files get selected, so most of what
follows is about equivalence rather than speed.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.context_builder import (  # noqa: E402
    IMPORT_PATTERNS,
    IMPORT_SCAN_CHARS,
    NON_CODE_LANGUAGES,
    _extract_imports,
    _score_file,
    build_context,
)
from modules.repo_ingestion import FileRecord, ingest_repository  # noqa: E402


def record(content, language="python", path="mod.py"):
    return FileRecord(
        path=path, abs_path=f"/{path}", content=content, size=len(content),
        extension=Path(path).suffix, language=language, checksum="x",
    )


# ── imports are still found ───────────────────────────────────────────────


def test_python_imports_are_found():
    found = _extract_imports("import os\nfrom pathlib import Path\nimport requests\n", "python")

    assert found == {"os", "pathlib", "requests"}


def test_javascript_requires_are_found():
    found = _extract_imports("const fs = require('fs');\nvar _ = require(\"lodash\");\n", "javascript")

    assert found == {"fs", "lodash"}


def test_dotted_imports_keep_only_the_root():
    assert _extract_imports("import os.path\n", "python") == {"os"}


def test_imports_after_a_comment_are_found():
    assert _extract_imports("# header\n\nimport numpy as np\n", "python") == {"numpy"}


def test_an_indented_import_is_not_matched():
    """The patterns are line-anchored; a function-local import is not a signal."""
    assert _extract_imports("def f():\n    import os\n", "python") == set()


# ── the scan is bounded ───────────────────────────────────────────────────


def test_the_scan_is_limited_to_the_head():
    """
    Imports are at the top by convention in every language scored here. The
    bound is what makes a 300KB file cheap.
    """
    assert IMPORT_SCAN_CHARS <= 32_000


def test_an_import_beyond_the_bound_is_not_scanned():
    content = ("# filler\n" * (IMPORT_SCAN_CHARS // 9 + 100)) + "import buried\n"

    assert "buried" not in _extract_imports(content, "python")


def test_an_import_within_the_bound_is_found():
    content = "import early\n" + ("# filler\n" * 500)

    assert "early" in _extract_imports(content, "python")


@pytest.mark.parametrize("language", ["json", "markdown", "css", "text", "unknown"])
def test_non_code_languages_are_skipped_entirely(language):
    assert _extract_imports("import os\n", language) == set()


def test_code_languages_are_not_skipped():
    assert "python" not in NON_CODE_LANGUAGES
    assert "javascript" not in NON_CODE_LANGUAGES


# ── patterns are compiled once ────────────────────────────────────────────


def test_patterns_are_precompiled():
    """Recompiling per file costs a cache lookup on every file, every iteration."""
    for pattern in IMPORT_PATTERNS:
        assert hasattr(pattern, "finditer")


def test_all_three_patterns_are_kept():
    assert len(IMPORT_PATTERNS) == 3


# ── behaviour is unchanged ────────────────────────────────────────────────


@pytest.fixture
def corpus(tmp_path):
    for name in ("parser", "sandbox", "client", "logger"):
        (tmp_path / f"{name}.py").write_text(
            f"import os\nimport re\n\ndef {name}_run(x):\n    return x\n" * 20
        )
    (tmp_path / "bundle.min.js").write_text("var a=1;" * 20000)
    (tmp_path / "README.md").write_text("# project\n" * 40)
    return ingest_repository(str(tmp_path))


def test_scoring_still_ranks_the_relevant_file_first(corpus):
    context = build_context(corpus, "fix the parser bug")

    assert context.files[0].path == "parser.py"


def test_a_minified_bundle_is_still_scored_not_skipped(corpus):
    """Bounding the scan must not remove the file from consideration."""
    scored = _score_file(record("var a=1;" * 20000, "javascript", "bundle.min.js"), ["a"], "t")

    assert scored.score is not None


def test_import_matches_still_raise_the_score():
    keywords = ["requests"]
    with_import = _score_file(record("import requests\nx = 1\n"), keywords, "use requests")
    without = _score_file(record("x = 1\n"), keywords, "use requests")

    assert with_import.score > without.score


# ── it is actually faster ─────────────────────────────────────────────────


def test_a_minified_file_scores_quickly():
    """
    The reported symptom. Generous threshold -- CI machines vary, and the point
    is catching a regression back to whole-file scanning, not policing ms.
    """
    minified = record("var a=1;" * 40000, "javascript", "bundle.min.js")
    keywords = ["parse", "parser", "function"]

    start = time.perf_counter()
    for _ in range(10):
        _score_file(minified, keywords, "fix the parse function")
    elapsed = (time.perf_counter() - start) / 10

    assert elapsed < 0.05


def test_scoring_cost_does_not_scale_with_file_size():
    """
    A file 10x larger should not cost 10x, now that the import scan is bounded.
    """
    keywords = ["parse"]
    small = record("import os\n" + "def parse(x):\n    return x\n" * 500)
    large = record("import os\n" + "def parse(x):\n    return x\n" * 5000)

    def timed(rec):
        start = time.perf_counter()
        for _ in range(5):
            _score_file(rec, keywords, "fix parse")
        return (time.perf_counter() - start) / 5

    assert timed(large) < timed(small) * 8
