"""
Module 7: Git Integration
Branch creation, staging, committing, pushing.
Optionally creates GitHub PRs via REST API.
"""

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class GitResult:
    command: str
    success: bool
    output: str
    error: str


PUSH_REMEDIES = {
    "non-fast-forward": (
        "The remote branch has commits this one does not. A rebase was either "
        "not attempted or did not resolve it -- fetch and rebase manually, or "
        "push to a new branch name."
    ),
    "no-remote": (
        "No such remote. Add one with `git remote add origin <url>`, or run "
        "without --push."
    ),
    "auth": (
        "Authentication failed. Check your SSH key or that GITHUB_TOKEN is set "
        "and has push access to this repository."
    ),
    "network": (
        "Could not reach the remote. Check connectivity and try again; the "
        "commit is safe locally."
    ),
    "protected": (
        "The remote refused the update -- the branch is likely protected, or a "
        "hook rejected it. Push to a different branch and open a PR."
    ),
    "unknown": (
        "Push failed. The commit is safe locally; push manually to inspect."
    ),
}


def classify_push_failure(stderr: str) -> str:
    """Map git's push error text onto a remedy key."""
    text = (stderr or "").lower()
    # Checked first: a protected-branch refusal may or may not also say
    # "rejected", and rebasing would not help either way.
    if any(k in text for k in ("protected branch", "pre-receive hook",
                               "gh006", "protected")):
        return "protected"
    if "non-fast-forward" in text or "fetch first" in text or "rejected" in text:
        return "non-fast-forward"
    if "does not appear to be a git repository" in text or "no such remote" in text:
        return "no-remote"
    if any(k in text for k in ("permission denied", "authentication failed",
                              "could not read username", "403")):
        return "auth"
    if any(k in text for k in ("could not resolve host", "connection timed out",
                               "network is unreachable", "operation timed out")):
        return "network"
    return "unknown"


class GitIntegration:
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self._verify_git()

    def _verify_git(self):
        result = self._run(["git", "rev-parse", "--is-inside-work-tree"])
        if not result.success:
            raise RuntimeError(f"Not a git repository: {self.repo_root}")

    def _run(self, cmd: list[str], check: bool = False) -> GitResult:
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            success = proc.returncode == 0
            if check and not success:
                raise RuntimeError(
                    f"Git command failed: {' '.join(cmd)}\n{proc.stderr}"
                )
            return GitResult(
                command=" ".join(cmd),
                success=success,
                output=proc.stdout.strip(),
                error=proc.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return GitResult(
                command=" ".join(cmd),
                success=False,
                output="",
                error="Git command timed out",
            )
        except FileNotFoundError:
            return GitResult(
                command=" ".join(cmd),
                success=False,
                output="",
                error="git binary not found",
            )

    def current_branch(self) -> str:
        result = self._run(["git", "branch", "--show-current"])
        return result.output or "unknown"

    def branch_exists(self, branch: str) -> bool:
        return self._run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"]
        ).success

    def _contains(self, ancestor: str, descendant: str) -> bool:
        """True if `descendant` already contains every commit in `ancestor`."""
        return self._run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant]
        ).success

    def create_branch(self, branch_name: str, from_branch: str = "main") -> GitResult:
        """
        Create and checkout a new branch off `from_branch`.

        If the branch already exists, it is only reused when it already contains
        the base branch's tip. Falling back to a plain checkout unconditionally
        looks like success while silently putting the agent on a stale branch --
        it then works against an out-of-date base and opens a PR from it.
        """
        base = self._run(["git", "rev-parse", "--verify", from_branch])
        if not base.success:
            base = self._run(["git", "rev-parse", "--verify", "master"])
            from_branch = "master" if base.success else "HEAD"

        result = self._run(["git", "checkout", "-b", branch_name, from_branch])
        if result.success:
            return result

        if not self.branch_exists(branch_name):
            # Failed for some other reason -- uncommitted changes in the way,
            # an invalid name. Report it rather than papering over it.
            return result

        if not self._contains(from_branch, branch_name):
            return GitResult(
                command=f"git checkout -b {branch_name} {from_branch}",
                success=False,
                output="",
                error=(
                    f"Branch '{branch_name}' already exists and does not contain "
                    f"'{from_branch}'. Checking it out would run the agent against "
                    f"a stale base. Delete it, or choose another branch name."
                ),
            )

        return self._run(["git", "checkout", branch_name])

    def stage_all(self) -> GitResult:
        """Stage all modified and new files."""
        return self._run(["git", "add", "-A"])

    def stage_files(self, paths: list[str]) -> GitResult:
        """Stage specific files."""
        return self._run(["git", "add", "--"] + paths)

    def commit(self, message: str, author: Optional[str] = None) -> GitResult:
        """Create a commit with the given message."""
        cmd = ["git", "commit", "-m", message]
        env = os.environ.copy()
        if author:
            author_name = author.split("<")[0].strip()
            env["GIT_AUTHOR_NAME"] = author_name
            env.setdefault("GIT_COMMITTER_NAME", author_name)
            email_match = re.search(r"<(.+)>", author)
            if email_match:
                author_email = email_match.group(1)
                env["GIT_AUTHOR_EMAIL"] = author_email
                env.setdefault("GIT_COMMITTER_EMAIL", author_email)

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=env,
            )
            return GitResult(
                command=" ".join(cmd),
                success=proc.returncode == 0,
                output=proc.stdout.strip(),
                error=proc.stderr.strip(),
            )
        except Exception as exc:
            return GitResult(command=" ".join(cmd), success=False, output="", error=str(exc))

    def rebase_onto_remote(self, branch: str, remote: str = "origin") -> GitResult:
        """
        Fetch the remote branch and rebase onto it.

        A rebase that hits conflicts is aborted rather than left half-applied --
        an unattended agent has no way to resolve them, and stopping mid-rebase
        would leave the user's working tree in a state they did not ask for.
        """
        fetch = self._run(["git", "fetch", remote, branch])
        if not fetch.success:
            return fetch

        rebase = self._run(["git", "rebase", f"{remote}/{branch}"])
        if rebase.success:
            return rebase

        self._run(["git", "rebase", "--abort"])
        return GitResult(
            command=f"git rebase {remote}/{branch}",
            success=False,
            output=rebase.output,
            error=(
                f"Rebase onto {remote}/{branch} hit conflicts and was aborted; "
                f"the working tree is unchanged. Resolve manually, or push to a "
                f"new branch name.\n{rebase.error}"
            ),
        )

    def push(
        self,
        branch: str,
        remote: str = "origin",
        force: bool = False,
        retry_with_rebase: bool = True,
        set_upstream: bool = True,
    ) -> GitResult:
        """
        Push `branch` to `remote`, recovering from a diverged remote branch.

        A rejected push is the one git failure worth retrying automatically: it
        means the remote branch moved, and rebasing on top is what a person
        would do. Everything else (missing remote, auth, network) gets a
        diagnosis appended instead, because retrying will not help.
        """
        cmd = ["git", "push"]
        if set_upstream:
            cmd.append("--set-upstream")
        cmd += [remote, branch]
        if force:
            cmd.append("--force-with-lease")

        result = self._run(cmd)
        if result.success:
            return result

        reason = classify_push_failure(result.error)

        if reason == "non-fast-forward" and retry_with_rebase and not force:
            rebase = self.rebase_onto_remote(branch, remote)
            if rebase.success:
                retried = self._run(cmd)
                if retried.success:
                    return retried
                result = retried
                reason = classify_push_failure(retried.error)
            else:
                return GitResult(
                    command=" ".join(cmd), success=False,
                    output=result.output, error=rebase.error,
                )

        return GitResult(
            command=" ".join(cmd),
            success=False,
            output=result.output,
            error=f"{result.error}\n\n{PUSH_REMEDIES[reason]}",
        )

    def diff_staged(self) -> str:
        """Return the staged diff as a string for PR descriptions."""
        result = self._run(["git", "diff", "--cached", "--stat"])
        return result.output

    def get_remote_url(self, remote: str = "origin") -> Optional[str]:
        result = self._run(["git", "remote", "get-url", remote])
        return result.output if result.success else None

    def has_uncommitted_changes(self) -> bool:
        result = self._run(["git", "status", "--porcelain"])
        return bool(result.output.strip())

    def create_github_pr(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        github_token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a GitHub PR via REST API.
        Returns the PR URL on success, None on failure.
        Requires GITHUB_TOKEN env var or explicit token.
        """
        token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            return None

        remote_url = self.get_remote_url()
        if not remote_url:
            return None

        match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", remote_url)
        if not match:
            return None
        owner, repo = match.group(1), match.group(2)

        payload = json.dumps({
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": False,
        }).encode("utf-8")

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                return data.get("html_url")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 422 and "already exists" in error_body:
                return f"PR already exists for {head_branch} -> {base_branch}"
            return None
        except Exception:
            return None
