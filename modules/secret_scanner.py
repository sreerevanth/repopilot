"""
Module: Secret Scanner
Scans files for common secret patterns before git operations.
Warns about accidental secret commits to prevent credential leaks.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SecretFinding:
    """Represents a detected secret in a file."""
    file_path: str
    line_number: int
    pattern_name: str
    matched_text: str  # redacted snippet
    severity: str  # "high" | "medium" | "low"


# ─────────────────────────────────────────────
# Secret Patterns
# ─────────────────────────────────────────────

SECRET_PATTERNS = [
    # AWS
    {
        "name": "AWS Access Key ID",
        "regex": r"(?:^|[^A-Z0-9])(?:AKIA[0-9A-Z]{16})(?:$|[^A-Z0-9])",
        "severity": "high",
    },
    {
        "name": "AWS Secret Access Key",
        "regex": r"(?:aws_secret_access_key|secret_key)\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
        "severity": "high",
    },
    # GitHub
    {
        "name": "GitHub Token",
        "regex": r"(?:ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})",
        "severity": "high",
    },
    # Generic API Keys
    {
        "name": "Generic API Key",
        "regex": r"(?:api[_-]?key|apikey)\s*[=:]\s*['\"]([A-Za-z0-9_\-]{20,})['\"]",
        "severity": "medium",
    },
    # Generic Secret/Token
    {
        "name": "Generic Secret/Token",
        "regex": r"(?:secret|token|password|passwd|pwd)\s*[=:]\s*['\"]([^\s'\"]{8,})['\"]",
        "severity": "medium",
    },
    # Private Keys
    {
        "name": "Private Key Block",
        "regex": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "severity": "high",
    },
    # Anthropic
    {
        "name": "Anthropic API Key",
        "regex": r"sk-ant-[A-Za-z0-9_\-]{40,}",
        "severity": "high",
    },
    # OpenAI
    {
        "name": "OpenAI API Key",
        "regex": r"sk-[A-Za-z0-9]{40,}",
        "severity": "high",
    },
    # Google
    {
        "name": "Google API Key",
        "regex": r"AIza[0-9A-Za-z_\-]{35}",
        "severity": "high",
    },
    # Database URLs
    {
        "name": "Database Connection String",
        "regex": r"(?:postgres|mysql|mongodb|redis)://[^\s'\"]+:[^\s'\"]+@",
        "severity": "high",
    },
]

# Files to always skip when scanning
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
    ".pdf", ".doc", ".docx",
    ".lock",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache",
    "venv", ".venv", "env", ".env",
}

# Files that commonly contain test/example secrets (lower severity)
ALLOW_FILES = {
    "test_", "mock_", "fake_", "example_", "sample_",
}


def _redact(text: str, max_show: int = 8) -> str:
    """Redact a secret, showing only first few chars."""
    if len(text) <= max_show:
        return text[:3] + "***"
    return text[:max_show] + "***"


def _is_test_file(path: str) -> bool:
    """Check if a file is likely a test/example file."""
    basename = os.path.basename(path).lower()
    return any(basename.startswith(prefix) for prefix in ALLOW_FILES)


def scan_file(file_path: str, content: Optional[str] = None) -> list[SecretFinding]:
    """
    Scan a single file for secret patterns.

    Args:
        file_path: Path to the file.
        content: Optional pre-read content. If None, reads from disk.

    Returns:
        List of SecretFinding objects.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SKIP_EXTENSIONS:
        return []

    if content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, PermissionError):
            return []

    findings = []
    is_test = _is_test_file(file_path)

    for line_num, line in enumerate(content.splitlines(), start=1):
        # Skip comments that mention secrets as documentation
        stripped = line.strip()
        if stripped.startswith("#") and "example" in stripped.lower():
            continue

        for pattern in SECRET_PATTERNS:
            match = re.search(pattern["regex"], line, re.IGNORECASE)
            if match:
                matched_text = match.group(0)
                severity = "low" if is_test else pattern["severity"]

                findings.append(SecretFinding(
                    file_path=file_path,
                    line_number=line_num,
                    pattern_name=pattern["name"],
                    matched_text=_redact(matched_text),
                    severity=severity,
                ))

    return findings


def scan_directory(
    root_dir: str,
    paths: Optional[list[str]] = None,
) -> list[SecretFinding]:
    """
    Scan a directory (or specific file paths) for secrets.

    Args:
        root_dir: Repository root directory.
        paths: Optional list of specific relative paths to scan.
               If None, scans the entire directory tree.

    Returns:
        List of all SecretFinding objects found.
    """
    all_findings = []

    if paths:
        for rel_path in paths:
            abs_path = os.path.join(root_dir, rel_path)
            if os.path.isfile(abs_path):
                findings = scan_file(abs_path)
                # Store relative path for cleaner output
                for f in findings:
                    f.file_path = rel_path
                all_findings.extend(findings)
    else:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for filename in filenames:
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, root_dir).replace(os.sep, "/")
                findings = scan_file(abs_path)
                for f in findings:
                    f.file_path = rel_path
                all_findings.extend(findings)

    return all_findings


def format_findings(findings: list[SecretFinding]) -> str:
    """Format findings as a human-readable report string."""
    if not findings:
        return "[Secret Scanner] No secrets detected."

    high = [f for f in findings if f.severity == "high"]
    medium = [f for f in findings if f.severity == "medium"]
    low = [f for f in findings if f.severity == "low"]

    lines = [
        f"[Secret Scanner] Found {len(findings)} potential secret(s):",
        "",
    ]

    for label, group in [("HIGH", high), ("MEDIUM", medium), ("LOW", low)]:
        if group:
            lines.append(f"  {label} severity ({len(group)}):")
            for f in group:
                lines.append(
                    f"    - {f.file_path}:{f.line_number} "
                    f"[{f.pattern_name}] {f.matched_text}"
                )
            lines.append("")

    lines.append(
        "  Review these findings before committing. "
        "Use .gitignore or remove secrets from tracked files."
    )
    return "\n".join(lines)
