"""
Tests for bounded ingestion memory (modules/repo_ingestion.py).

Ingestion submitted every candidate file to the thread pool at once, so the
whole repository was read into memory before the size budget was consulted. A
141 MB checkout peaked at 166 MB RSS to keep the 8 MB the budget allows — the
cost scaled with the repository rather than with the budget.

Reading in batches, and stopping the walk once the budget is full, caps that at
roughly one batch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.repo_ingestion import (  # noqa: E402
    MAX_TOTAL_BYTES,
    READ_BATCH_SIZE,
    ingest_repository,
)


def corpus(tmp_path, count, kb_each):
    body = "x = 1\n" * (kb_each * 1024 // 6)
    for i in range(count):
        (tmp_path / f"mod_{i:04d}.py").write_text(f"# {i}\n{body}")
    return tmp_path


# ── the batch bound ───────────────────────────────────────────────────────


def test_the_batch_size_is_bounded():
    """
    Unbounded is what caused the spike. The value only needs to be small
    relative to a large repository, not tuned.
    """
    assert 0 < READ_BATCH_SIZE <= 256


def test_reads_happen_in_batches():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "repo_ingestion.py"
    ).read_text(encoding="utf-8")

    assert "range(0, len(candidates), READ_BATCH_SIZE)" in source


def test_the_walk_stops_once_the_budget_is_full():
    """
    Without this the remaining files are read and then discarded, which is the
    memory cost the batching is there to avoid.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "repo_ingestion.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("for start in range(0, len(candidates)"):][:600]

    assert "if repo.total_bytes >= MAX_TOTAL_BYTES:" in block
    assert "break" in block


# ── behaviour is unchanged ────────────────────────────────────────────────


def test_a_small_repo_is_fully_ingested(tmp_path):
    corpus(tmp_path, count=5, kb_each=1)

    assert len(ingest_repository(str(tmp_path)).files) == 5


def test_a_repo_spanning_several_batches_is_fully_ingested(tmp_path):
    """The batching must not drop files that fit within the budget."""
    corpus(tmp_path, count=READ_BATCH_SIZE * 2 + 7, kb_each=1)

    repo = ingest_repository(str(tmp_path))

    assert len(repo.files) == READ_BATCH_SIZE * 2 + 7
    assert repo.skipped == []


def test_file_contents_are_intact(tmp_path):
    (tmp_path / "one.py").write_text("def hello():\n    return 'world'\n")

    record = ingest_repository(str(tmp_path)).files[0]

    assert record.content == "def hello():\n    return 'world'\n"


def test_metadata_is_still_populated(tmp_path):
    (tmp_path / "one.py").write_text("x = 1\n")

    record = ingest_repository(str(tmp_path)).files[0]

    assert record.language == "python"
    assert record.checksum
    assert record.size > 0


# ── the budget still holds ────────────────────────────────────────────────


def test_the_budget_is_respected(tmp_path):
    corpus(tmp_path, count=60, kb_each=256)   # ~15 MB, budget is 8 MB

    repo = ingest_repository(str(tmp_path))

    assert repo.total_bytes <= MAX_TOTAL_BYTES


def test_files_beyond_the_budget_are_recorded_as_skipped(tmp_path):
    corpus(tmp_path, count=60, kb_each=256)

    repo = ingest_repository(str(tmp_path))

    assert repo.skipped
    assert any("budget exhausted" in entry for entry in repo.skipped)


def test_every_candidate_is_accounted_for(tmp_path):
    """Kept plus skipped should equal what was found — nothing vanishes."""
    corpus(tmp_path, count=60, kb_each=256)

    repo = ingest_repository(str(tmp_path))

    assert len(repo.files) + len(repo.skipped) == 60


def test_an_empty_repo_yields_nothing(tmp_path):
    repo = ingest_repository(str(tmp_path))

    assert repo.files == []
    assert repo.skipped == []
