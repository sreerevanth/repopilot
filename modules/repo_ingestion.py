"""
Module 1: Repository Ingestion
Recursively scans a project, ignores noise, stores file content + metadata.
"""

import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fnmatch
import re
from concurrent.futures import ThreadPoolExecutor

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

SECRET_PATTERNS = [
    re.compile(r"sk-ant-[a-zA-Z0-9_-]{40,}"), # Anthropic API Key
    re.compile(r"sk-[a-zA-Z0-9]{48}"),         # OpenAI API Key
    re.compile(r"AIzaSy[a-zA-Z0-9_-]{33}"),     # Gemini API Key
    re.compile(r"AWS_SECRET_ACCESS_KEY\s*=\s*['\"][a-zA-Z0-9+/=]{40}['\"]", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z]+ PRIVATE KEY-----"),
]


def contains_secret(content: str) -> bool:
    """Scan content for potential API keys or secrets."""
    for pattern in SECRET_PATTERNS:
        if pattern.search(content):
            return True
    return False


def parse_gitignore(repo_root: str) -> list[str]:
    """Parse .gitignore patterns from root directory."""
    gitignore_path = os.path.join(repo_root, ".gitignore")
    if not os.path.exists(gitignore_path):
        return []
    patterns = []
    try:
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
    except Exception:
        pass
    return patterns


def is_ignored_by_gitignore(rel_path: str, patterns: list[str]) -> bool:
    """Check if relative path matches any .gitignore patterns."""
    for pattern in patterns:
        # Normalize patterns to match POSIX slashes
        clean_pat = pattern.replace("\\", "/").rstrip("/")
        if clean_pat.startswith("/"):
            clean_pat = clean_pat[1:]
        
        # Match directory or exact files
        if fnmatch.fnmatch(rel_path, clean_pat) or fnmatch.fnmatch(rel_path, clean_pat + "/*") or f"/{clean_pat}/" in f"/{rel_path}/":
            return True
        if fnmatch.fnmatch(os.path.basename(rel_path), clean_pat):
            return True
    return False


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


def _process_file(abs_path: str, rel_path: str) -> Optional[tuple[str, str, int]]:
    """Helper to read and validate a single file."""
    try:
        size = os.path.getsize(abs_path)
        if size > MAX_FILE_SIZE_BYTES:
            return None
        with open(abs_path, "r", encoding="utf-8", errors="strict") as fh:
            content = fh.read()
        if contains_secret(content):
            return None # Skip secret-containing file
        return (content, abs_path, size)
    except (UnicodeDecodeError, PermissionError, OSError):
        return None


def ingest_repository(repo_root: str) -> Repository:
    """
    Walk repo_root, read all relevant text files, return a Repository object.
    Respects size budgets, ignores binary/secret files, and supports .gitignore.
    Uses concurrent ThreadPoolExecutor for parallel file reading.
    """
    root = os.path.abspath(repo_root)
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    repo = Repository(root=root)
    gitignore_patterns = parse_gitignore(root)

    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place (modifies walk)
        dirnames[:] = [d for d in dirnames if not _should_ignore_dir(d)]

        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")

            if _should_ignore_file(abs_path):
                repo.skipped.append(rel_path)
                continue

            if is_ignored_by_gitignore(rel_path, gitignore_patterns):
                repo.skipped.append(f"{rel_path} (gitignore)")
                continue

            candidates.append((abs_path, rel_path))

    # Parallel file reading
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_process_file, abs_p, rel_p): rel_p for abs_p, rel_p in candidates}
        for future in futures:
            rel_p = futures[future]
            res = future.result()
            if res is None:
                repo.skipped.append(rel_p)
                continue
            content, abs_path, size = res

            if repo.total_bytes + size > MAX_TOTAL_BYTES:
                repo.skipped.append(f"{rel_p} (repo budget exhausted)")
                continue

            ext = Path(abs_path).suffix.lower()
            checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

            record = FileRecord(
                path=rel_p,
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
