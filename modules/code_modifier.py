"""
Module 4: Code Modification Engine
Safely applies file changes from LLM output.
Creates backups, validates paths, preserves structure.
"""

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from modules.llm_client import FileChange


@dataclass
class ApplyResult:
    path: str
    action: str
    success: bool
    backup_path: Optional[str]
    error: Optional[str]


class CodeModificationEngine:
    def __init__(self, repo_root: str, backup_dir: str):
        self.repo_root = os.path.abspath(repo_root)
        self.backup_dir = os.path.abspath(backup_dir)
        os.makedirs(self.backup_dir, exist_ok=True)
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _safe_abs_path(self, relative_path: str) -> str:
        """
        Resolve relative_path within repo_root.
        Raises ValueError if the resolved path would escape the repo root
        (path traversal prevention).
        """
        if not relative_path or not relative_path.strip():
            raise ValueError("Path is empty")

        candidate = Path(relative_path.replace("\\", "/"))
        if candidate.is_absolute():
            raise ValueError(f"Absolute paths are not allowed: '{relative_path}'")

        abs_path = (Path(self.repo_root) / candidate).resolve()
        repo_root = Path(self.repo_root).resolve()

        try:
            abs_path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                f"Path traversal detected: '{relative_path}' resolves to '{abs_path}'"
            ) from exc
        return str(abs_path)

    def _backup(self, abs_path: str, relative_path: str) -> Optional[str]:
        """Copy file to backup dir before modification. Returns backup path."""
        if not os.path.exists(abs_path):
            return None
        safe_name = relative_path.replace("/", "__").replace("\\", "__")
        backup_path = os.path.join(
            self.backup_dir,
            f"{self._session_id}__{safe_name}"
        )
        shutil.copy2(abs_path, backup_path)
        return backup_path

    def _apply_single(self, change: FileChange) -> ApplyResult:
        try:
            abs_path = self._safe_abs_path(change.path)
        except ValueError as e:
            return ApplyResult(
                path=change.path, action=change.action,
                success=False, backup_path=None, error=str(e)
            )

        backup_path = None

        try:
            if change.action == "delete":
                backup_path = self._backup(abs_path, change.path)
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                return ApplyResult(
                    path=change.path, action="delete",
                    success=True, backup_path=backup_path, error=None
                )

            elif change.action in ("modify", "create"):
                if not change.content and change.action == "modify":
                    return ApplyResult(
                        path=change.path, action=change.action,
                        success=False, backup_path=None,
                        error="LLM returned empty content for modify action"
                    )

                # Backup existing file
                backup_path = self._backup(abs_path, change.path)

                # Ensure parent dirs exist
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)

                # Write new content
                with open(abs_path, "w", encoding="utf-8") as fh:
                    fh.write(change.content)

                return ApplyResult(
                    path=change.path, action=change.action,
                    success=True, backup_path=backup_path, error=None
                )

            else:
                return ApplyResult(
                    path=change.path, action=change.action,
                    success=False, backup_path=None,
                    error=f"Unknown action: {change.action}"
                )

        except Exception as e:
            return ApplyResult(
                path=change.path, action=change.action,
                success=False, backup_path=backup_path,
                error=f"Unexpected error: {e}"
            )

    def apply_changes(self, changes: list[FileChange]) -> list[ApplyResult]:
        """Apply a list of file changes. Returns per-file results."""
        results = []
        for change in changes:
            result = self._apply_single(change)
            results.append(result)
        return results

    def rollback(self, results: list[ApplyResult]) -> list[str]:
        """
        Restore all backed-up files. Used if the iteration should be fully reverted.
        Returns list of restored paths.
        """
        restored = []
        for result in results:
            if result.backup_path and os.path.exists(result.backup_path):
                try:
                    abs_path = self._safe_abs_path(result.path)
                    shutil.copy2(result.backup_path, abs_path)
                    restored.append(result.path)
                except Exception:
                    pass
        return restored

    def verify_changes(self, changes: list[FileChange]) -> list[str]:
        """
        Pre-flight validation before applying.
        Returns list of error strings (empty = all good).
        """
        errors = []
        for change in changes:
            if not change.path:
                errors.append("Change has empty path")
                continue
            if change.action not in ("modify", "create", "delete"):
                errors.append(f"Invalid action '{change.action}' for {change.path}")
            if change.action in ("modify", "create") and not change.content:
                errors.append(f"Empty content for {change.action} on {change.path}")
            try:
                self._safe_abs_path(change.path)
            except ValueError as e:
                errors.append(str(e))
        return errors
