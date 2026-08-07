"""
Module 1: Repository Ingestion
Recursively scans a project, ignores noise, stores file content + metadata.
"""

import fnmatch
import os
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # optional at runtime so an un-reinstalled checkout still works
    import pathspec
except ImportError:  # pragma: no cover - exercised via the fallback tests
    pathspec = None

_LOG = logging.getLogger("agent.repo_ingestion")

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

# Files that must never enter the LLM prompt, whether or not .gitignore lists
# them. Ingested content is sent verbatim to the model, so a missing or
# incomplete .gitignore should not be the only thing standing between a
# developer's secrets and an outbound API call.
SECRET_FILE_PATTERNS = (
    ".env", ".env.*", "*.env",
    "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    ".npmrc", ".pypirc", ".netrc", "_netrc",
    "credentials", "credentials.json",
    ".htpasswd", "*.keystore", "*.jks",
)

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


def _is_secret_file(filepath: str) -> bool:
    """True if the filename matches a pattern that commonly holds credentials."""
    name = os.path.basename(filepath).lower()
    return any(fnmatch.fnmatch(name, pat) for pat in SECRET_FILE_PATTERNS)


def _should_ignore_file(filepath: str) -> bool:
    ext = Path(filepath).suffix.lower()
    name = os.path.basename(filepath).lower()
    if _is_secret_file(filepath):
        return True
    if ext in IGNORE_EXTENSIONS:
        return True
    if name.startswith(".") and name not in (".gitignore", ".dockerignore"):
        return True
    return False


def load_ignore_spec(root: str):
    """
    Build a PathSpec from the repo's .gitignore and .git/info/exclude.

    Returns None when pathspec is unavailable or there is nothing to parse, in
    which case ingestion falls back to the hardcoded IGNORE_DIRS/EXTENSIONS
    behaviour. Only root-level ignore files are read; nested .gitignore files
    are not resolved (see the note in the README).
    """
    if pathspec is None:
        _LOG.debug("pathspec not installed; falling back to hardcoded ignore rules")
        return None

    lines: list[str] = []
    for candidate in (
        os.path.join(root, ".gitignore"),
        os.path.join(root, ".git", "info", "exclude"),
    ):
        try:
            with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
                lines.extend(fh.read().splitlines())
        except OSError:
            continue

    if not lines:
        return None

    # "gitignore" is the current factory name; "gitwildmatch" is its deprecated
    # predecessor and is still the only one available on pathspec < 0.12.
    for style in ("gitignore", "gitwildmatch"):
        try:
            return pathspec.PathSpec.from_lines(style, lines)
        except (KeyError, LookupError):
            continue
        except Exception as exc:  # malformed patterns must not break ingestion
            _LOG.warning("could not parse ignore patterns (%s); using defaults", exc)
            return None
    return None


def ingest_repository(repo_root: str) -> Repository:
    """
    Walk repo_root, read all relevant text files, return a Repository object.
    Respects size budgets and skips binary/irrelevant files.
    """
    root = os.path.abspath(repo_root)
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    repo = Repository(root=root)
    spec = load_ignore_spec(root)

    def _rel(path: str) -> str:
        return os.path.relpath(path, root).replace(os.sep, "/")

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place (modifies walk). Directories are
        # matched with a trailing slash so gitwildmatch honours "build/" style
        # patterns, and pruning here means an ignored tree is never descended.
        kept = []
        for d in dirnames:
            if _should_ignore_dir(d):
                continue
            rel_dir = _rel(os.path.join(dirpath, d))
            if spec is not None and spec.match_file(rel_dir + "/"):
                repo.skipped.append(f"{rel_dir}/ (gitignored)")
                continue
            kept.append(d)
        dirnames[:] = kept

        for filename in sorted(filenames):
            abs_path = os.path.join(dirpath, filename)
            rel_path = _rel(abs_path)

            if _should_ignore_file(abs_path):
                repo.skipped.append(rel_path)
                continue

            if spec is not None and spec.match_file(rel_path):
                repo.skipped.append(f"{rel_path} (gitignored)")
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
