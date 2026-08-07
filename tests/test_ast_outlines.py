"""
Tests for AST outlines (modules/context_builder.py).

A file that does not fit the character budget was dropped entirely, leaving the
model unaware the module existed. It now falls back to a signature-only outline,
which costs roughly a seventh of the space.
"""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.context_builder import (  # noqa: E402
    OUTLINE_HEADER,
    build_context,
    extract_outline,
)
from modules.repo_ingestion import FileRecord, Repository  # noqa: E402

SAMPLE = textwrap.dedent('''
    """Module docstring.

    More detail that should not appear.
    """

    import os
    from typing import Optional

    MAX_SIZE = 1024
    _private = "hidden"


    @dataclass
    class Config:
        path: str
        retries: int = 3

        @property
        def is_valid(self) -> bool:
            return bool(self.path)

        async def load(self, source: str) -> Optional[dict]:
            body = {"a": 1}
            return body


    def helper(x: int, *args, flag: bool = False, **kwargs) -> str:
        secret = "this body must not appear"
        return secret
''')


@pytest.fixture
def outline():
    return extract_outline(SAMPLE, "mod.py")


# ── what the outline contains ─────────────────────────────────────────────


def test_header_marks_it_as_partial(outline):
    """The model must not read an outline as the file's real contents."""
    assert outline.startswith(OUTLINE_HEADER)


def test_signatures_are_preserved(outline):
    expected = "def helper(x: int, *args, flag: bool=False, **kwargs) -> str: ..."
    assert expected in outline


def test_class_and_bases_are_preserved(outline):
    assert "class Config:" in outline


def test_dataclass_fields_are_kept(outline):
    """For a dataclass the fields are the interesting part, not the methods."""
    assert "path: str" in outline
    assert "retries: int" in outline


def test_decorators_are_kept(outline):
    assert "@dataclass" in outline
    assert "@property" in outline


def test_async_defs_are_marked(outline):
    assert "async def load" in outline


def test_imports_are_kept(outline):
    assert "import os" in outline
    assert "from typing import Optional" in outline


def test_upper_case_constants_are_kept(outline):
    assert "MAX_SIZE = ..." in outline


def test_lower_case_module_state_is_dropped(outline):
    assert "_private" not in outline


def test_first_docstring_line_only(outline):
    assert "Module docstring." in outline
    assert "More detail" not in outline


def test_function_bodies_are_gone(outline):
    assert "this body must not appear" not in outline
    assert 'body = {"a": 1}' not in outline


def test_outline_is_smaller_than_the_source(outline):
    assert len(outline) < len(SAMPLE)


def test_saving_is_large_on_a_realistic_file():
    """
    The fixture above is almost entirely signatures, so the ratio there is
    unremarkable. The saving comes from files with real bodies -- measured at
    13-18% on this repo's own modules.
    """
    padded = SAMPLE.replace(
        '    secret = "this body must not appear"',
        "    secret = 'x'\n" + "    # padding\n" * 300,
    )
    assert len(extract_outline(padded, "mod.py")) < len(padded) / 4


def test_outline_is_valid_python(outline):
    """It is read as source by the model; it should not look malformed."""
    import ast
    ast.parse(outline)


# ── when no outline is produced ───────────────────────────────────────────


def test_non_python_files_are_skipped():
    assert extract_outline("# heading\n\ntext", "README.md") is None
    assert extract_outline("body { color: red; }", "app.css") is None


def test_unparseable_source_returns_none():
    """Expected mid-run: the agent may have just broken the file."""
    assert extract_outline("def broken(:\n    pass", "mod.py") is None


def test_file_with_no_definitions_returns_none():
    """Imports and constants alone are not worth a slot in the budget."""
    assert extract_outline("import os\nX = 1\n", "mod.py") is None


def test_empty_file_returns_none():
    assert extract_outline("", "mod.py") is None


def test_class_with_no_members_still_renders():
    result = extract_outline("class Marker:\n    pass\n", "mod.py")
    assert "class Marker:" in result
    assert "..." in result


# ── integration with the budget ───────────────────────────────────────────


def make_repo(*files):
    records = [
        FileRecord(path=p, abs_path=f"/r/{p}", content=c, size=len(c),
                   extension=".py", language="python", checksum="x")
        for p, c in files
    ]
    return Repository(root="/r", files=records)


BIG = (
    '"""Big module."""\n\n\ndef relevant_sandbox_helper(a: int) -> int:\n'
    + "    # padding\n" * 400
    + "    return a\n"
)
SMALL = '"""Small."""\n\n\ndef sandbox_small() -> None:\n    return None\n'


def test_oversized_file_is_outlined_rather_than_dropped():
    repo = make_repo(("big.py", BIG), ("small.py", SMALL))
    ctx = build_context(repo, "sandbox", char_budget=len(SMALL) + 600)

    paths = {f.path for f in ctx.files}
    assert "small.py" in paths
    assert "big.py" in paths           # would have been absent entirely before
    assert ctx.outlined == ["big.py"]


def test_outlined_file_carries_the_outline_not_the_body():
    repo = make_repo(("big.py", BIG), ("small.py", SMALL))
    ctx = build_context(repo, "sandbox", char_budget=len(SMALL) + 600)

    big = next(f for f in ctx.files if f.path == "big.py")
    assert big.content.startswith(OUTLINE_HEADER)
    assert "relevant_sandbox_helper" in big.content
    assert "# padding" not in big.content


def test_the_budget_is_still_respected():
    repo = make_repo(("big.py", BIG), ("small.py", SMALL))
    budget = len(SMALL) + 600
    assert build_context(repo, "sandbox", char_budget=budget).total_chars <= budget


def test_no_outline_when_even_that_will_not_fit():
    repo = make_repo(("big.py", BIG))
    ctx = build_context(repo, "sandbox", char_budget=50)
    assert ctx.files == []
    assert ctx.outlined == []


def test_files_that_fit_are_not_outlined():
    repo = make_repo(("small.py", SMALL))
    ctx = build_context(repo, "sandbox", char_budget=100_000)

    assert ctx.outlined == []
    assert ctx.files[0].content == SMALL


def test_higher_scoring_files_are_outlined_first():
    """
    Scoring orders which outlines fit, rather than excluding any: every Python
    file already scores 10.0 from the language bonus alone, so the score guard
    only ever rejects non-code. With room for one outline, the better match
    wins.
    """
    other = BIG.replace("relevant_sandbox_helper", "unrelated_helper")
    repo = make_repo(("big.py", BIG), ("other.py", other), ("small.py", SMALL))

    ctx = build_context(repo, "sandbox", char_budget=len(SMALL) + 620)

    assert ctx.outlined[0] == "big.py"


def test_non_code_is_not_outlined():
    """extract_outline declines anything that is not Python."""
    assert extract_outline("# notes\n\nsome prose", "NOTES.md") is None


def test_summary_reports_outlined_files():
    repo = make_repo(("big.py", BIG), ("small.py", SMALL))
    ctx = build_context(repo, "sandbox", char_budget=len(SMALL) + 600)
    assert "outline-only" in ctx.summary()
    assert "big.py" in ctx.summary()


def test_outlined_defaults_to_empty():
    """The new field must not break callers constructing BuiltContext."""
    from modules.context_builder import BuiltContext

    assert BuiltContext(files=[], total_chars=0, scoring_details=[]).outlined == []
