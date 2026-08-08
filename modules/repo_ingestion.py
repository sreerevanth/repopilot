"""
Module 1: Repository Ingestion
Recursively scans a project, ignores noise, stores file content + metadata.
"""

import fnmatch
import os
import re
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Files read concurrently per batch. Caps peak memory at roughly one batch of
# file contents instead of the whole repository.
READ_BATCH_SIZE = 64

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
    Respects size budgets, ignores binary/secret files, and supports .gitignore.
    Uses concurrent ThreadPoolExecutor for parallel file reading.
    """
    root = os.path.abspath(repo_root)
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    repo = Repository(root=root)
    spec = load_ignore_spec(root)

    def _rel(path: str) -> str:
        return os.path.relpath(path, root).replace(os.sep, "/")

    candidates = []
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

        for filename in filenames:
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

            candidates.append((abs_path, rel_path))

    def _process_file(abs_path: str, rel_path: str):
        """
        Read one file, or None if it cannot be decoded or opened.

        Defined here rather than at module level because the caller below was
        already written against this name -- the definition was lost in a merge
        while the call survived, so ingest_repository raised NameError on every
        run. Mirrors the reader in ingest_repository_parallel.
        """
        try:
            with open(abs_path, "r", encoding="utf-8", errors="strict") as fh:
                content = fh.read()
        except (UnicodeDecodeError, PermissionError, OSError):
            return None
        return content, abs_path, len(content.encode("utf-8"))

    # Read in bounded batches rather than submitting every candidate at once.
    #
    # Submitting all of them holds the entire repository in memory before the
    # budget is ever consulted -- a 141 MB checkout peaked at 166 MB RSS to keep
    # the 8 MB the budget allows, and that scales with the repository rather
    # than with the budget. Batching caps the excess at one batch, and the
    # budget check now stops the walk instead of reading the remainder and
    # discarding it.
    with ThreadPoolExecutor() as executor:
        for start in range(0, len(candidates), READ_BATCH_SIZE):
            if repo.total_bytes >= MAX_TOTAL_BYTES:
                repo.skipped.extend(
                    f"{rel} (repo budget exhausted)"
                    for _, rel in candidates[start:]
                )
                break

            batch = candidates[start:start + READ_BATCH_SIZE]
            futures = {
                executor.submit(_process_file, abs_p, rel_p): rel_p
                for abs_p, rel_p in batch
            }
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


def ingest_repository_parallel(repo_root: str, max_workers: int = 10) -> Repository:
    """
    Parallel version of ingest_repository using ThreadPoolExecutor.
    """
    root = os.path.abspath(repo_root)
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    repo = Repository(root=root)
    
    # First pass: collect all files to process
    paths_to_process = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_ignore_dir(d)]
        for filename in filenames:
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
                
            paths_to_process.append((abs_path, rel_path, size))

    # Process files in parallel
    def process_file(file_info):
        abs_path, rel_path, size = file_info
        try:
            with open(abs_path, "r", encoding="utf-8", errors="strict") as fh:
                content = fh.read()
            ext = Path(abs_path).suffix.lower()
            checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
            return FileRecord(
                path=rel_path,
                abs_path=abs_path,
                content=content,
                size=size,
                extension=ext,
                language=infer_language(abs_path),
                checksum=checksum,
            )
        except (UnicodeDecodeError, PermissionError):
            return rel_path # return string to indicate failure

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, info): info for info in paths_to_process}
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, str):
                repo.skipped.append(f"{result} (binary or unreadable)")
            else:
                if repo.total_bytes + result.size > MAX_TOTAL_BYTES:
                    repo.skipped.append(f"{result.path} (repo budget exhausted)")
                else:
                    repo.files.append(result)
                    repo.total_bytes += result.size

    # Sort files to ensure deterministic order (important for LLM context consistency)
    repo.files.sort(key=lambda x: x.path)
    
    return repo
