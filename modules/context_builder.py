"""
Module 2: Context Builder
Selects the most relevant files for a task without sending the entire repo.
Uses keyword matching, file type priority, import graph hints, and scoring.
"""

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

from modules.repo_ingestion import FileRecord, Repository

CONTEXT_CHAR_BUDGET = 60_000

LANGUAGE_PRIORITY = {
    "python": 10, "javascript": 10, "typescript": 10, "java": 9,
    "go": 9, "rust": 9, "cpp": 8, "c": 8, "csharp": 8,
    "ruby": 8, "php": 8, "swift": 8, "kotlin": 8,
    "bash": 6, "sql": 6, "html": 5, "css": 4, "scss": 4,
    "json": 3, "yaml": 3, "toml": 3, "markdown": 2,
    "text": 1, "unknown": 0,
}


@dataclass
class ScoredFile:
    record: FileRecord
    score: float
    reasons: list[str]


@dataclass
class BuiltContext:
    files: list[FileRecord]
    total_chars: int
    scoring_details: list[ScoredFile]

    def render(self) -> str:
        """Render context as a single string suitable for an LLM prompt."""
        parts = []
        for file_record in self.files:
            parts.append(
                f"<file path=\"{file_record.path}\" language=\"{file_record.language}\">\n"
                f"{file_record.content}\n"
                f"</file>"
            )
        return "\n\n".join(parts)

    def summary(self) -> str:
        lines = [f"Context: {len(self.files)} files, {self.total_chars} chars"]
        for scored_file in self.scoring_details[:5]:
            lines.append(
                f"  [{scored_file.score:.1f}] {scored_file.record.path} - "
                f"{', '.join(scored_file.reasons)}"
            )
        return "\n".join(lines)


def _normalize_repo_path(path: str) -> str:
    """Normalize repository-relative paths to POSIX separators."""
    return PurePosixPath(path.replace("\\", "/")).as_posix()


def _extract_keywords(task: str) -> list[str]:
    """
    Pull meaningful tokens from the task description.
    Strips stopwords and returns lower-case identifiers.
    """
    stopwords = {
        "the", "a", "an", "in", "to", "of", "and", "or", "is", "it",
        "for", "on", "with", "this", "that", "are", "be", "was", "as",
        "at", "by", "from", "but", "not", "have", "has", "do", "does",
        "fix", "add", "update", "change", "make", "please", "should",
        "need", "want", "can", "could", "would", "will", "may", "might",
    }
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", task.lower())
    return [token for token in tokens if token not in stopwords and len(token) > 2]


def _score_file(record: FileRecord, keywords: list[str], task: str) -> ScoredFile:
    score = 0.0
    reasons = []
    normalized_path = _normalize_repo_path(record.path)

    lang_score = LANGUAGE_PRIORITY.get(record.language, 0)
    score += lang_score
    if lang_score >= 8:
        reasons.append(f"lang:{record.language}")

    path_lower = normalized_path.lower().replace("/", " ").replace("_", " ")
    path_matches = sum(1 for keyword in keywords if keyword in path_lower)
    if path_matches:
        score += path_matches * 5
        reasons.append(f"path_match:{path_matches}")

    content_lower = record.content.lower()
    content_matches = sum(content_lower.count(keyword) for keyword in keywords)
    if content_matches:
        score += math.log(content_matches + 1) * 3
        reasons.append(f"content_match:{content_matches}")

    entry_names = {
        "main.py", "app.py", "index.py", "server.py", "cli.py",
        "main.js", "index.js", "app.js", "main.ts", "index.ts",
        "__main__.py", "manage.py", "wsgi.py", "asgi.py",
    }
    basename = PurePosixPath(normalized_path).name.lower()
    if basename in entry_names:
        score += 8
        reasons.append("entry_point")

    config_names = {
        "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
        "package.json", "cargo.toml", "go.mod", "pom.xml",
        "dockerfile", "docker-compose.yml", "makefile",
    }
    if basename in config_names:
        score += 4
        reasons.append("config_file")

    task_lower = task.lower()
    is_test = "test" in normalized_path.lower() or "spec" in normalized_path.lower()
    if is_test and ("test" in task_lower or "bug" in task_lower or "fix" in task_lower):
        score += 6
        reasons.append("test_file")
    elif is_test:
        score -= 3

    import_patterns = [
        r"^import\s+(\w+)",
        r"^from\s+(\w+)",
        r"require\(['\"]([^'\"]+)['\"]",
    ]
    imported_modules: set[str] = set()
    for pattern in import_patterns:
        for match in re.finditer(pattern, record.content, re.MULTILINE):
            imported_modules.add(match.group(1).lower().split(".")[0])

    import_hits = sum(1 for keyword in keywords if keyword in imported_modules)
    if import_hits:
        score += import_hits * 4
        reasons.append(f"import_match:{import_hits}")

    if record.size > 50_000:
        score -= 2

    if not reasons:
        reasons.append("baseline")

    return ScoredFile(record=record, score=score, reasons=reasons)


def build_context(
    repo: Repository,
    task: str,
    extra_paths: Optional[list[str]] = None,
    char_budget: int = CONTEXT_CHAR_BUDGET,
) -> BuiltContext:
    """
    Score all repo files against the task and return the top-scoring files
    that fit within the character budget.
    """
    keywords = _extract_keywords(task)
    scored = [_score_file(record, keywords, task) for record in repo.files]
    scored.sort(key=lambda scored_file: scored_file.score, reverse=True)

    selected: list[FileRecord] = []
    total_chars = 0
    forced_paths = {_normalize_repo_path(path) for path in (extra_paths or [])}

    for scored_file in scored:
        record_path = _normalize_repo_path(scored_file.record.path)
        if record_path in forced_paths:
            selected.append(scored_file.record)
            total_chars += len(scored_file.record.content)
            forced_paths.discard(record_path)

    for scored_file in scored:
        if scored_file.record in selected:
            continue
        file_chars = len(scored_file.record.content)
        if total_chars + file_chars > char_budget:
            continue
        if scored_file.score <= 0:
            break
        selected.append(scored_file.record)
        total_chars += file_chars

    return BuiltContext(files=selected, total_chars=total_chars, scoring_details=scored)
