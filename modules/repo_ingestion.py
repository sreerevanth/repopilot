"""
Module 1: Repository Ingestion
Recursively scans a project, ignores noise, stores file content + metadata.
"""

import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

IGNORE_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".pytest_cache",
    "venv", ".venv", "env", ".env", "dist", "build", ".next", ".nuxt",
    "coverage", ".coverage", ".mypy_cache", ".tox", "eggs", ".eggs",
    "htmlcov", ".DS_Store", "target", "vendor", ".idea", ".vscode",
    "logs", "backups",
}

IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".class",
    ".o", ".a", ".lib", ".exe", ".bin", ".out",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".lock",  # package-lock.json etc. — too noisy
}

MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB per file
MAX_TOTAL_BYTES = 8 * 1024 * 1024  # 8 MB total repo budget


@dataclass
class FileRecord:
    path: str           # relative path from repo root
    abs_path: str       # absolute path on disk
    content: str        # decoded text content
    size: int           # bytes
    extension: str
    language: str       # inferred from extension
    checksum: str       # sha256 of content


@dataclass
class Repository:
    root: str
    files: list[FileRecord] = field(default_factory=list)
    total_bytes: int = 0
    skipped: list[str] = field(default_factory=list)

    def get_file(self, relative_path: str) -> Optional[FileRecord]:
        for f in self.files:
            if f.path == relative_path:
                return f
        return None

    def summary(self) -> str:
        lines = [
            f"Repository: {self.root}",
            f"Files loaded: {len(self.files)}",
            f"Total size: {self.total_bytes / 1024:.1f} KB",
            f"Skipped: {len(self.skipped)} files",
        ]
        ext_counts: dict[str, int] = {}
        for f in self.files:
            ext_counts[f.extension] = ext_counts.get(f.extension, 0) + 1
        for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {ext or '(no ext)'}: {count} files")
        return "\n".join(lines)


EXTENSION_TO_LANGUAGE = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".cs": "csharp", ".swift": "swift", ".kt": "kotlin",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".sql": "sql", ".html": "html", ".css": "css", ".scss": "scss",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml", ".md": "markdown",
    ".txt": "text", ".env": "env", ".ini": "ini", ".cfg": "config",
    ".dockerfile": "dockerfile", "dockerfile": "dockerfile",
    ".makefile": "makefile", "makefile": "makefile",
}


def infer_language(path: str) -> str:
    name = os.path.basename(path).lower()
    ext = Path(path).suffix.lower()
    return (
        EXTENSION_TO_LANGUAGE.get(name)
        or EXTENSION_TO_LANGUAGE.get(ext)
        or "unknown"
    )


def _should_ignore_dir(dirname: str) -> bool:
    return dirname.lower() in IGNORE_DIRS or dirname.startswith(".")


def _should_ignore_file(filepath: str) -> bool:
    ext = Path(filepath).suffix.lower()
    name = os.path.basename(filepath).lower()
    if ext in IGNORE_EXTENSIONS:
        return True
    if name.startswith(".") and name not in (".env", ".gitignore", ".dockerignore"):
        return True
    return False


def ingest_repository(repo_root: str) -> Repository:
    """
    Walk repo_root, read all relevant text files, return a Repository object.
    Respects size budgets and skips binary/irrelevant files.
    """
    root = os.path.abspath(repo_root)
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    repo = Repository(root=root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place (modifies walk)
        dirnames[:] = [d for d in dirnames if not _should_ignore_dir(d)]

        for filename in sorted(filenames):
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")

            if _should_ignore_file(abs_path):
                repo.skipped.append(rel_path)
                continue

            try:
                size = os.path.getsize(abs_path)
            except OSError:
                repo.skipped.append(rel_path)
                continue

            if size > MAX_FILE_SIZE_BYTES:
                repo.skipped.append(f"{rel_path} (too large: {size} bytes)")
                continue

            if repo.total_bytes + size > MAX_TOTAL_BYTES:
                repo.skipped.append(f"{rel_path} (repo budget exhausted)")
                continue

            # Try to read as UTF-8 text
            try:
                with open(abs_path, "r", encoding="utf-8", errors="strict") as fh:
                    content = fh.read()
            except (UnicodeDecodeError, PermissionError):
                repo.skipped.append(f"{rel_path} (binary or unreadable)")
                continue

            ext = Path(abs_path).suffix.lower()
            checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

            record = FileRecord(
                path=rel_path,
                abs_path=abs_path,
                content=content,
                size=size,
                extension=ext,
                language=infer_language(abs_path),
                checksum=checksum,
            )
            repo.files.append(record)
            repo.total_bytes += size

    return repo
