"""
Module 6 (CORE): Autonomous Execution Loop
The orchestrator that ties all modules together.

Flow per iteration:
  1. Ingest / refresh repo state
  2. Build context for the task
  3. Call LLM (first pass or retry with error)
  4. Validate + apply code changes
  5. Run tests / execute in sandbox
  6. If success → commit + push + optional PR → done
  7. If failure → feed error back to LLM → next iteration
  8. If max_retries exceeded → rollback + report
"""

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from modules.repo_ingestion import ingest_repository, Repository
from modules.context_builder import build_context
from modules.llm_client import LLMClient, LLMResponse, FileChange
from modules.code_modifier import CodeModificationEngine, ApplyResult
from modules.sandbox import SubprocessSandbox, ExecutionResult
from modules.git_integration import GitIntegration
from modules.logger import AgentLogger, IterationRecord


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class AgentConfig:
    repo_root: str
    task: str

    # Execution
    test_runner: str = "pytest"            # pytest | npm_test | go | cargo | ...
    test_args: Optional[list] = None       # extra args to pass to runner
    run_file: Optional[str] = None         # run a specific file instead of tests
    run_file_runner: str = "python"
    timeout_seconds: int = 120

    # Loop control
    max_iterations: int = 5
    min_confidence_to_apply: float = 0.25  # skip changes below this
    success_on_llm_done: bool = False       # trust LLM's "done=true" without running?

    # Git
    git_enabled: bool = True
    git_branch_prefix: str = "agent"
    git_base_branch: str = "main"
    git_commit_author: str = "Agent Bot <agent@autonomous.dev>"
    git_push: bool = False                  # push to remote?
    git_create_pr: bool = False             # create GitHub PR?

    # Directories
    backup_dir: str = "backups"
    log_dir: str = "logs"

    # LLM
    anthropic_api_key: Optional[str] = None

    # Context
    force_include_paths: Optional[list] = None  # always include these files


@dataclass
class AgentRunResult:
    run_id: str
    outcome: str        # success | failed | max_retries | error
    branch_name: Optional[str]
    pr_url: Optional[str]
    iterations_used: int
    final_message: str


# ─────────────────────────────────────────────
# The Agent
# ─────────────────────────────────────────────

class AutonomousAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.run_id = f"{config.git_branch_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.branch_name: Optional[str] = None

        # Resolve paths
        cfg = self.config
        self.backup_dir = os.path.abspath(os.path.join(cfg.repo_root, cfg.backup_dir))
        self.log_dir = os.path.abspath(os.path.join(cfg.repo_root, cfg.log_dir))

        # Instantiate modules
        self.logger = AgentLogger(self.log_dir, self.run_id, verbose=True)
        self.llm = LLMClient(api_key=cfg.anthropic_api_key)
        self.modifier = CodeModificationEngine(cfg.repo_root, self.backup_dir)
        self.sandbox = SubprocessSandbox(cfg.repo_root, timeout_seconds=cfg.timeout_seconds)

        self.git: Optional[GitIntegration] = None
        if cfg.git_enabled:
            try:
                self.git = GitIntegration(cfg.repo_root)
            except RuntimeError as e:
                self.logger.warning(f"Git unavailable: {e}. Continuing without git.")

    def _sanitize_branch_name(self, task: str) -> str:
        """Convert task text into a valid git branch name."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", task.lower())[:40].strip("-")
        return f"{self.config.git_branch_prefix}/{slug}-{self.run_id[-6:]}"

    def _run_execution(self) -> ExecutionResult:
        """Run tests or the specified file in the sandbox."""
        cfg = self.config
        if cfg.run_file:
            return self.sandbox.run_file(cfg.run_file, cfg.run_file_runner)
        else:
            return self.sandbox.run_tests(cfg.test_runner, cfg.test_args)

    def _commit_changes(self, iteration: int, changed_paths: list[str]) -> bool:
        """Stage and commit the modified files."""
        if not self.git:
            return True  # No git; just pretend it worked.

        stage = self.git.stage_files(changed_paths)
        self.logger.log_git("stage", stage)
        if not stage.success:
            # Fallback to staging everything
            stage = self.git.stage_all()
            self.logger.log_git("stage (all)", stage)

        if not self.git.has_uncommitted_changes():
            self.logger.info("  No changes to commit (files may be unchanged)")
            return True

        msg = (
            f"agent: iteration {iteration} - {self.config.task[:60]}\n\n"
            f"run_id: {self.run_id}\n"
            f"branch: {self.branch_name}"
        )
        commit = self.git.commit(msg, author=self.config.git_commit_author)
        self.logger.log_git("commit", commit)
        return commit.success

    def run(self) -> AgentRunResult:
        cfg = self.config
        self.logger.start_run(cfg.task, cfg.repo_root)

        # ── Git: create working branch ──
        if self.git:
            self.branch_name = self._sanitize_branch_name(cfg.task)
            branch_result = self.git.create_branch(self.branch_name, cfg.git_base_branch)
            self.logger.log_git(f"create branch '{self.branch_name}'", branch_result)
            if not branch_result.success:
                self.logger.warning("Could not create git branch. Continuing on current branch.")
                self.branch_name = self.git.current_branch()

        last_exec: Optional[ExecutionResult] = None
        last_changes: list[FileChange] = []
        last_apply_results: list[ApplyResult] = []
        outcome = "failed"
        pr_url: Optional[str] = None
        iterations_used = 0

        for iteration in range(1, cfg.max_iterations + 1):
            iterations_used = iteration
            self.logger.start_iteration(iteration)

            # ── Step 1: Ingest repo (re-read to pick up changes from previous iter) ──
            try:
                repo: Repository = ingest_repository(cfg.repo_root)
            except Exception as e:
                self.logger.error(f"Repo ingestion failed: {e}")
                outcome = "error"
                self.logger.finish_run(outcome, self.branch_name, None, str(e))
                return AgentRunResult(
                    run_id=self.run_id, outcome=outcome, branch_name=self.branch_name,
                    pr_url=None, iterations_used=iteration - 1, final_message=str(e)
                )

            # ── Step 2: Build context ──
            context = build_context(
                repo, cfg.task,
                extra_paths=cfg.force_include_paths,
            )
            self.logger.log_context([f.path for f in context.files], context.total_chars)
            context_str = context.render()

            # ── Step 3: Call LLM ──
            try:
                if iteration == 1 or not last_exec:
                    llm_resp: LLMResponse = self.llm.initial_request(cfg.task, context_str)
                else:
                    llm_resp = self.llm.retry_request(
                        task=cfg.task,
                        context_str=context_str,
                        previous_changes=last_changes,
                        stdout=last_exec.stdout,
                        stderr=last_exec.stderr,
                        exit_code=last_exec.exit_code,
                    )
            except Exception as e:
                self.logger.error(f"LLM call failed: {e}")
                outcome = "error"
                break

            self.logger.log_llm_response(
                llm_resp.analysis, llm_resp.confidence,
                llm_resp.changes, llm_resp.done, llm_resp.parse_error
            )

            # Record iteration data
            iter_record = IterationRecord(
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                context_files=[f.path for f in context.files],
                context_chars=context.total_chars,
                llm_analysis=llm_resp.analysis,
                llm_confidence=llm_resp.confidence,
                llm_done=llm_resp.done,
                changes_attempted=[{"path": c.path, "action": c.action, "explanation": c.explanation} for c in llm_resp.changes],
                apply_results=[],
                execution_command=None,
                execution_exit_code=None,
                execution_stdout=None,
                execution_stderr=None,
                execution_timed_out=False,
                execution_success=False,
                parse_error=llm_resp.parse_error,
            )

            # ── Handle parse error ──
            if llm_resp.parse_error:
                self.logger.warning(f"LLM parse error (iter {iteration}): {llm_resp.parse_error}")
                if not llm_resp.changes:
                    self.logger.info("  No usable changes; continuing to next iteration with error context")
                    last_exec = ExecutionResult(
                        command="(no execution - LLM parse error)",
                        exit_code=1,
                        stdout="",
                        stderr=f"LLM returned malformed JSON: {llm_resp.parse_error}",
                        timed_out=False,
                        duration_seconds=0.0,
                    )
                    self.logger.record_iteration(iter_record)
                    continue

            # ── Handle low confidence ──
            if llm_resp.confidence < cfg.min_confidence_to_apply:
                self.logger.warning(
                    f"  LLM confidence {llm_resp.confidence:.2f} < {cfg.min_confidence_to_apply}. "
                    f"Skipping changes this iteration."
                )
                if iteration == cfg.max_iterations:
                    outcome = "failed"
                    break
                self.logger.record_iteration(iter_record)
                continue

            # ── Step 4: Validate + Apply changes ──
            if llm_resp.changes:
                validation_errors = self.modifier.verify_changes(llm_resp.changes)
                if validation_errors:
                    self.logger.warning(f"  Validation errors: {validation_errors}")
                    # Filter to valid changes only
                    valid_changes = [
                        c for c in llm_resp.changes
                        if not any(c.path in e for e in validation_errors)
                    ]
                else:
                    valid_changes = llm_resp.changes

                apply_results = self.modifier.apply_changes(valid_changes)
                self.logger.log_apply_results(apply_results)
                last_changes = valid_changes
                last_apply_results = apply_results

                iter_record.apply_results = [
                    {"path": r.path, "action": r.action, "success": r.success, "error": r.error}
                    for r in apply_results
                ]

                # If ALL apply operations failed, something is very wrong
                if apply_results and all(not r.success for r in apply_results):
                    self.logger.error("  All file modifications failed. Check paths and permissions.")
                    outcome = "error"
                    self.logger.record_iteration(iter_record)
                    break
            else:
                self.logger.info("  No file changes from LLM this iteration.")
                last_changes = []
                last_apply_results = []

            # ── Early exit if LLM says done (optional) ──
            if cfg.success_on_llm_done and llm_resp.done and llm_resp.confidence >= 0.8:
                self.logger.info("  LLM reports task complete. Skipping execution (success_on_llm_done=True).")
                self._commit_changes(iteration, [c.path for c in last_changes])
                outcome = "success"
                iter_record.execution_success = True
                self.logger.record_iteration(iter_record)
                break

            # ── Step 5: Execute ──
            exec_result = self._run_execution()
            self.logger.log_execution(exec_result)
            last_exec = exec_result

            iter_record.execution_command = exec_result.command
            iter_record.execution_exit_code = exec_result.exit_code
            iter_record.execution_stdout = exec_result.stdout[:2000]
            iter_record.execution_stderr = exec_result.stderr[:2000]
            iter_record.execution_timed_out = exec_result.timed_out
            iter_record.execution_success = exec_result.success

            self.logger.record_iteration(iter_record)

            # ── Step 6: Success check ──
            if exec_result.success:
                self.logger.info(f"  Tests passed on iteration {iteration}")
                self._commit_changes(iteration, [c.path for c in last_changes])
                outcome = "success"
                break

            # ── Step 7: Handle timeout specifically ──
            if exec_result.timed_out:
                self.logger.warning(
                    f"  Execution timed out after {self.sandbox.timeout}s. "
                    f"Consider increasing timeout or fixing infinite loops."
                )

            # ── More iterations needed ──
            if iteration < cfg.max_iterations:
                self.logger.info(f"  Feeding error back to LLM (iteration {iteration + 1} of {cfg.max_iterations})")
            else:
                self.logger.warning(f"  Max iterations ({cfg.max_iterations}) reached.")
                outcome = "max_retries"

        # ── Post-loop: Git push + PR ──
        if outcome == "success" and self.git and self.branch_name:
            if cfg.git_push:
                push_result = self.git.push(self.branch_name)
                self.logger.log_git(f"push '{self.branch_name}'", push_result)

                if push_result.success and cfg.git_create_pr:
                    diff_stat = self.git.diff_staged() or "See commit for changes."
                    pr_url = self.git.create_github_pr(
                        title=f"[Agent] {cfg.task[:72]}",
                        body=(
                            f"## Autonomous Agent PR\n\n"
                            f"**Task:** {cfg.task}\n\n"
                            f"**Run ID:** `{self.run_id}`\n\n"
                            f"**Changes:**\n```\n{diff_stat}\n```"
                        ),
                        head_branch=self.branch_name,
                        base_branch=cfg.git_base_branch,
                    )
                    if pr_url:
                        self.logger.info(f"  PR created: {pr_url}")

        # ── Rollback on failure if rollback_on_failure ──
        if outcome in ("failed", "max_retries", "error") and last_apply_results:
            self.logger.warning("  Rolling back file changes due to failed run...")
            restored = self.modifier.rollback(last_apply_results)
            self.logger.info(f"  Rolled back {len(restored)} file(s): {restored}")

        final_message = {
            "success": "Task completed successfully. Tests pass.",
            "failed": "Task could not be completed. Check logs.",
            "max_retries": f"Exhausted {cfg.max_iterations} iterations without passing tests.",
            "error": "Agent encountered an unrecoverable error.",
        }.get(outcome, "Unknown outcome")

        self.logger.finish_run(outcome, self.branch_name, pr_url)

        return AgentRunResult(
            run_id=self.run_id,
            outcome=outcome,
            branch_name=self.branch_name,
            pr_url=pr_url,
            iterations_used=iterations_used,
            final_message=final_message,
        )
