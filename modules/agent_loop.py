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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from modules.repo_ingestion import ingest_repository, ingest_repository_parallel, Repository
from modules.context_builder import build_context, budget_for_model
from modules.llm_client import (
    BudgetExceededError,
    FileChange,
    LLMClient,
    LLMResponse,
)
from modules.code_modifier import CodeModificationEngine, ApplyResult
from modules.sandbox import (
    ExecutionResult,
    SubprocessSandbox,
    coverage_args,
    parse_coverage_percent,
)
from modules.git_integration import GitIntegration
from modules.doc_lookup import perform_lookups, render_lookups
from modules.run_state import check_resumable, clear_state, load_state, save_state
from modules.logger import AgentLogger, IterationRecord
from modules.secret_scanner import scan_directory, format_findings
from modules.token_tracker import TokenTracker


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class AgentConfig:
    repo_root: str
    task: str
    dry_run: bool = False
    context_only: bool = False             # print the compiled context and stop
    yes: bool = False
    interactive: bool = False   # pause for review after tests pass, before commit

    # Execution
    skip_tests: bool = False               # accept the LLM's own verdict instead
    plan_first: bool = False               # ask for an approach before coding
    test_runner: str = "pytest"            # pytest | npm_test | go | cargo | ...
    test_args: Optional[list] = None       # extra args to pass to runner
    skip_tests: bool = False               # accept the LLM's own verdict instead
    run_file: Optional[str] = None         # run a specific file instead of tests
    run_file_runner: str = "python"
    coverage: bool = False                 # measure coverage and feed drops back
    coverage_source: str = "."             # what --cov points at
    lint_runner: Optional[str] = None      # run a linter before the test suite
    lint_args: list[str] = field(default_factory=list)
    timeout_seconds: int = 120

    # Loop control
    max_iterations: int = 5
    min_confidence_to_apply: float = 0.25  # skip changes below this
    success_on_llm_done: bool = False       # trust LLM's "done=true" without running?

    # Git
    git_enabled: bool = True
    git_branch_prefix: str = "agent"
    git_base_branch: str = "main"
    git_commit_author: str = "RepoPilot Agent <agent@repopilot.local>"

    # CLI behavior
    yes: bool = False
    git_push: bool = False                  # push to remote?
    git_create_pr: bool = False             # create GitHub PR?

    # Directories
    backup_dir: str = "backups"
    log_dir: str = "logs"

    # LLM
    anthropic_api_key: Optional[str] = None
    resume_from: Optional[str] = None      # run_id to continue
    verbose_payloads: bool = False         # dump raw LLM request/response
    model: Optional[str] = None
    context_budget: Optional[int] = None   # chars; derived from the model if unset
    provider: str = "anthropic"

    # Context
    force_include_paths: Optional[list] = None  # always include these files

    # Parallel Processing
    parallel: bool = False
    workers: int = 10

@dataclass
class AgentRunResult:
    run_id: str
    outcome: str        # success | failed | max_retries | error | aborted
                        #   | dry_run | context_only
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
        self.pr_url: Optional[str] = None

        # Mirrors the loop's local `last_apply_results`. run() needs it to roll
        # back after a KeyboardInterrupt, which unwinds past the local.
        self._applied: list = []

        # Resolve paths
        cfg = self.config
        self.backup_dir = os.path.abspath(os.path.join(cfg.repo_root, cfg.backup_dir))
        self.log_dir = os.path.abspath(os.path.join(cfg.repo_root, cfg.log_dir))

        # Instantiate modules
        self.logger = AgentLogger(self.log_dir, self.run_id, verbose=True)
        # Built on first use rather than here. --context-only returns before any
        # request is made, so constructing a client eagerly made a flag whose
        # whole point is "no API call, no cost" fail without the provider SDK
        # installed and a valid key present.
        self._llm: Optional[LLMClient] = None
        self.modifier = CodeModificationEngine(cfg.repo_root, self.backup_dir)
        self.sandbox = SubprocessSandbox(cfg.repo_root, timeout_seconds=cfg.timeout_seconds)
        self.token_tracker = TokenTracker()

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

    def _skipped_execution(self, confidence: float) -> ExecutionResult:
        """
        Stand-in result for --skip-tests.

        Deliberately not dressed up as a passing test run: `command` says what
        happened and the stdout records the confidence the decision rested on,
        so a log cannot be mistaken for evidence that a suite went green.
        """
        return ExecutionResult(
            command="(skipped - --skip-tests; no test suite was run)",
            exit_code=0,
            stdout=(
                f"Tests were not run. Accepted on the model's own confidence of "
                f"{confidence:.2f}. No suite verified this change."
            ),
            stderr="",
            timed_out=False,
            duration_seconds=0.0,
        )

    @property
    def llm(self) -> LLMClient:
        """The provider client, constructed on first access."""
        if self._llm is None:
            cfg = self.config
            self._llm = LLMClient(
                api_key=cfg.anthropic_api_key,
                model=cfg.model,
                provider=cfg.provider,
                verbose=cfg.verbose_payloads,
            )
        return self._llm

    def _run_execution(self) -> ExecutionResult:
        """Run tests or the specified file in the sandbox."""
        cfg = self.config
        if cfg.run_file:
            # --run-file executes one script; --cov would report on the wrong
            # thing, so coverage is deliberately not applied here.
            return self.sandbox.run_file(cfg.run_file, cfg.run_file_runner)

        extra = list(cfg.test_args or [])
        if cfg.coverage and not cfg.run_file:
            extra += coverage_args(cfg.coverage_source)
        return self.sandbox.run_tests(cfg.test_runner, extra)

    # Cap the review diff so a large refactor cannot flood the terminal.
    _MAX_DIFF_LINES = 400

    def _confirm_commit(self, changed_paths: list[str]) -> bool:
        """
        Show what the agent wrote and ask before committing.

        Only runs under --interactive. --yes bypasses it (and main.py sets that
        automatically when CI=true), matching how the pre-apply prompt behaves,
        so unattended runs are never left waiting on stdin.
        """
        cfg = self.config
        if not cfg.interactive or cfg.yes:
            return True

        from modules.dry_run import ask_confirmation

        print("\n" + "=" * 60)
        print("  TESTS PASSED - REVIEW BEFORE COMMIT")
        print("=" * 60)

        diff = self.git.diff_unstaged() if self.git else ""
        if diff.strip():
            lines = diff.splitlines()
            for line in lines[: self._MAX_DIFF_LINES]:
                print(line)
            if len(lines) > self._MAX_DIFF_LINES:
                print(
                    f"\n  ... diff truncated at {self._MAX_DIFF_LINES} lines "
                    f"({len(lines) - self._MAX_DIFF_LINES} more). "
                    f"Run `git diff` in another shell for the full text."
                )
        else:
            # No git, or git reported nothing. Fall back to the file list so the
            # prompt is never shown without context.
            reason = "Git is disabled" if not self.git else "Git reported no diff"
            print(f"  {reason}. Files the agent changed:")
            for path in changed_paths:
                print(f"    {path}")

        print("=" * 60)
        return ask_confirmation(len(changed_paths), action="Commit")

    def _commit_changes(self, iteration: int, changed_paths: list[str]) -> bool:
        """Stage and commit the modified files."""
        if not self.git:
            return True  # No git; just pretend it worked.

        # Secret scan before committing
        findings = scan_directory(self.config.repo_root, paths=changed_paths)
        if findings:
            report = format_findings(findings)
            self.logger.warning(report)
            high_findings = [f for f in findings if f.severity == "high"]
            if high_findings:
                self.logger.warning(
                    f"  {len(high_findings)} HIGH severity secret(s) detected! "
                    f"Review before pushing."
                )

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
        """
        Execute the agent loop, rolling back applied files if interrupted.

        Ctrl+C during `apply_changes` used to unwind straight out of the loop,
        past the rollback at the end of `_run`, leaving half-written files in
        the working tree. The body is delegated so the interrupt can be caught
        here and the same rollback applied.
        """
        try:
            return self._run()
        except KeyboardInterrupt:
            return self._handle_interrupt()

    def _handle_interrupt(self) -> AgentRunResult:
        """Restore backed-up files after Ctrl+C and report an aborted run."""
        self.logger.warning("\n  Interrupted. Rolling back applied changes...")

        restored: list[str] = []
        if self._applied:
            try:
                restored = self.modifier.rollback(self._applied)
                self.logger.info(f"  Restored {len(restored)} file(s): {restored}")
            except Exception as exc:
                # Report rather than mask the interrupt with a second failure.
                self.logger.error(f"  Rollback failed after interrupt: {exc}")
        else:
            self.logger.info("  No files had been modified yet; nothing to restore.")

        self._applied = []

        try:
            self.logger.finish_run("aborted", self.branch_name, None)
        except Exception:  # pragma: no cover - logging must not mask the abort
            pass

        return AgentRunResult(
            run_id=self.run_id,
            outcome="aborted",
            branch_name=self.branch_name,
            pr_url=None,
            iterations_used=0,
            final_message=(
                f"Interrupted by user. Rolled back {len(restored)} file(s); "
                f"the working tree is back to its pre-run state."
                if restored else
                "Interrupted by user. No file changes needed rolling back."
            ),
        )

    def _run(self) -> AgentRunResult:
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
        start_iteration = 1

        if cfg.resume_from:
            state = load_state(cfg.log_dir, cfg.resume_from)
            check_resumable(state, cfg.repo_root, cfg.task)
            start_iteration = state.iteration + 1
            self.branch_name = state.branch_name or self.branch_name
            last_changes = [FileChange(**c) for c in state.last_changes]
            if state.last_exit_code is not None:
                # Rebuilt rather than stored whole: the retry prompt only reads
                # these four fields, and persisting a full ExecutionResult would
                # tie the state file to that dataclass's shape.
                last_exec = ExecutionResult(
                    command="(restored from a previous run)",
                    exit_code=state.last_exit_code,
                    stdout=state.last_stdout,
                    stderr=state.last_stderr,
                    timed_out=False,
                    duration_seconds=0.0,
                )
            self.logger.info(
                f"  Resuming '{cfg.resume_from}' at iteration {start_iteration} "
                f"({len(last_changes)} change(s) from the previous attempt)."
            )

        outcome = "failed"
        self.pr_url = None
        iterations_used = 0

        baseline_coverage: Optional[float] = None
        if cfg.coverage:
            # Measured before the model touches anything, so a first-iteration
            # regression is still caught.
            baseline_run = self._run_execution()
            baseline_coverage = parse_coverage_percent(
                baseline_run.stdout + baseline_run.stderr
            )
            if baseline_coverage is None:
                self.logger.warning(
                    "  --coverage is on but no coverage total was found. Is "
                    "pytest-cov installed? Continuing without the gate."
                )
            else:
                self.logger.info(f"  Baseline coverage: {baseline_coverage:.1f}%")

        for iteration in range(1, cfg.max_iterations + 1):
            iterations_used = iteration
            self.logger.start_iteration(iteration)

            # ── Step 1: Ingest repo (re-read to pick up changes from previous iter) ──
            try:
                if cfg.parallel:
                    repo: Repository = ingest_repository_parallel(cfg.repo_root, max_workers=cfg.workers)
                else:
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
                char_budget=(
                    cfg.context_budget
                    if cfg.context_budget
                    else budget_for_model(cfg.model)
                ),
            )
            self.logger.log_context([f.path for f in context.files], context.total_chars)
            context_str = context.render()

            if cfg.context_only:
                # Before the LLM call, so this costs nothing. --dry-run already
                # exists but shows the *changes* the model proposed, which means
                # it has already been paid for by the time you see anything.
                print(context_str)
                summary = (
                    f"{len(context.files)} file(s), {context.total_chars:,} chars"
                )
                if getattr(context, "outlined", None):
                    summary += f", {len(context.outlined)} outlined"
                self.logger.info(f"  Context only: {summary}. No LLM call made.")
                self.logger.finish_run("context_only", self.branch_name, None)
                return AgentRunResult(
                    run_id=self.run_id,
                    outcome="context_only",
                    branch_name=self.branch_name,
                    pr_url=None,
                    iterations_used=0,
                    final_message=(
                        f"Compiled context printed ({summary}). "
                        f"No API call was made and no files were touched."
                    ),
                )

            # ── Step 3: Call LLM ──
            try:
                if iteration == 1 or not last_exec:
                    plan = None
                    if cfg.plan_first and iteration == 1:
                        # Only on the first iteration. Later ones already carry
                        # the strongest possible signal -- real test output --
                        # and re-planning against it would cost a call to
                        # restate what the failure already says.
                        plan = self.llm.plan_request(cfg.task, context_str)
                        if plan.usable:
                            self.logger.info(
                                f"  Plan ({len(plan.steps)} steps, "
                                f"confidence {plan.confidence:.2f}):"
                            )
                            for number, step in enumerate(plan.steps, 1):
                                self.logger.info(f"    {number}. {step}")
                            for risk in plan.risks:
                                self.logger.warning(f"    risk: {risk}")
                        else:
                            self.logger.warning(
                                f"  Planning pass unusable "
                                f"({plan.parse_error or 'no steps returned'}); "
                                f"continuing without it."
                            )
                    llm_resp: LLMResponse = self.llm.initial_request(
                        cfg.task, context_str, plan=plan
                    )
                else:
                    llm_resp = self.llm.retry_request(
                        task=cfg.task,
                        context_str=context_str,
                        previous_changes=last_changes,
                        stdout=last_exec.stdout,
                        stderr=last_exec.stderr,
                        exit_code=last_exec.exit_code,
                    )
            except BudgetExceededError as e:
                # A deliberate stop, not a failure. Reported separately so the
                # run log distinguishes "ran out of money" from "the API broke".
                self.logger.warning(f"  {e}")
                outcome = "budget_exceeded"
                break
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

                from modules.dry_run import print_manifest, save_manifest, ask_confirmation

                changes_list = [
                    c.__dict__ if hasattr(c, '__dict__') else c
                    for c in llm_resp.changes
                ]

                print_manifest(changes_list)

                if self.config.dry_run:
                    manifest_path = save_manifest(changes_list, self.config.log_dir, self.run_id)
                    self.logger.info("[DRY RUN] No files were modified.")
                    self.logger.info(f"[DRY RUN] Manifest saved to: {manifest_path}")
                    return AgentRunResult(
                        outcome="dry_run",
                        run_id=self.run_id,
                        iterations_used=iteration,
                        final_message="Dry run complete. Review the manifest before applying.",
                        branch_name=None,
                        pr_url=None,
                    )

                if not (self.config.yes or ask_confirmation(len(llm_resp.changes))):
                    return AgentRunResult(
                        outcome="aborted",
                        run_id=self.run_id,
                        iterations_used=iteration,
                        final_message="Aborted by user at confirmation prompt.",
                        branch_name=None,
                        pr_url=None,
                    )

                self.modifier.git_stash_before_apply(self.config.repo_root)
                if cfg.parallel and len(valid_changes) > 1:
                    apply_results = self.modifier.apply_changes_parallel(valid_changes, max_workers=cfg.workers)
                else:
                    apply_results = self.modifier.apply_changes(valid_changes)
                self.logger.log_apply_results(apply_results)
                last_changes = valid_changes
                last_apply_results = apply_results
                self._applied = apply_results

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
                self._applied = []

            # ── Early exit if LLM says done (optional) ──
            if cfg.success_on_llm_done and llm_resp.done and llm_resp.confidence >= 0.8:
                self.logger.info("  LLM reports task complete. Skipping execution (success_on_llm_done=True).")
                changed_paths = [c.path for c in last_changes]
                if not self._confirm_commit(changed_paths):
                    self.logger.warning(
                        "  Commit declined. Changes are left in the working tree."
                    )
                    outcome = "aborted"
                    self.logger.record_iteration(iter_record)
                    break

                self._commit_changes(iteration, changed_paths)
                outcome = "success"
                iter_record.execution_success = True
                self.logger.record_iteration(iter_record)
                break

            # ── Step 5: Execute ──
            # Lint first when configured. A syntax error surfaces in under a
            # second with a precise location, instead of arriving as a pytest
            # collection error several seconds later.
            if cfg.lint_runner:
                lint_result = self.sandbox.run_lint(cfg.lint_runner, cfg.lint_args)
                self.logger.log_execution(lint_result)
                if not lint_result.success:
                    self.logger.warning(
                        f"  Lint failed ({cfg.lint_runner}); "
                        f"skipping tests and retrying with the lint output."
                    )
                    last_exec = lint_result
                    iter_record.execution_command = lint_result.command
                    iter_record.execution_exit_code = lint_result.exit_code
                    iter_record.execution_stdout = lint_result.stdout[:2000]
                    iter_record.execution_stderr = lint_result.stderr[:2000]
                    iter_record.execution_success = False
                    self.logger.record_iteration(iter_record)
                    continue

            if cfg.skip_tests:
                exec_result = self._skipped_execution(llm_resp.confidence)
            else:
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

            if cfg.coverage and exec_result.success:
                current = parse_coverage_percent(
                    exec_result.stdout + exec_result.stderr
                )
                drop = self._coverage_feedback(baseline_coverage, current)
                if drop:
                    self.logger.warning(f"  {drop}")
                    exec_result = ExecutionResult(
                        command=exec_result.command + "  [coverage gate]",
                        exit_code=1,
                        stdout=exec_result.stdout,
                        stderr=(exec_result.stderr + "\n\n" + drop).strip(),
                        timed_out=False,
                        duration_seconds=exec_result.duration_seconds,
                    )
                elif current is not None:
                    self.logger.info(f"  Coverage: {current:.1f}%")

            # ── Step 6: Success check ──
            if exec_result.success:
                if cfg.skip_tests:
                    self.logger.warning(
                        f"  Accepting iteration {iteration} without running tests "
                        f"(--skip-tests, confidence {llm_resp.confidence:.2f})."
                    )
                else:
                    self.logger.info(f"  Tests passed on iteration {iteration}")

                changed_paths = [c.path for c in last_changes]
                if not self._confirm_commit(changed_paths):
                    self.logger.warning(
                        "  Commit declined. Changes are left in the working tree."
                    )
                    outcome = "aborted"
                    break

                self._commit_changes(iteration, changed_paths)
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
                    self.pr_url = self.git.create_github_pr(
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
                    if self.pr_url:
                        self.logger.info(f"  PR created: {self.pr_url}")

        # ── Rollback on failure if rollback_on_failure ──
        if (
            outcome in ("failed", "max_retries", "error", "budget_exceeded")
            and last_apply_results
        ):
            self.logger.warning("  Rolling back file changes due to failed run...")
            restored = self.modifier.rollback(last_apply_results)
            self.logger.info(f"  Rolled back {len(restored)} file(s): {restored}")

        if outcome not in ("failed", "max_retries", "error"):
            # A run that ended deliberately has nothing to resume. Leaving the
            # checkpoint would offer to continue a finished run.
            clear_state(cfg.log_dir, self.run_id)

        final_message = {
            "budget_exceeded": (
                f"Stopped at the --max-cost limit after {self.llm.usage.summary()}. "
                f"Any applied changes were rolled back."
            ),
            "success": "Task completed successfully. Tests pass.",
            "failed": "Task could not be completed. Check logs.",
            "max_retries": f"Exhausted {cfg.max_iterations} iterations without passing tests.",
            "error": "Agent encountered an unrecoverable error.",
            "aborted": (
                "Commit declined at the review prompt. Tests passed and the "
                "changes are still in the working tree - commit them manually, "
                "or run with --rollback to discard them."
            ),
        }.get(outcome, "Unknown outcome")

        self.logger.finish_run(outcome, self.branch_name, self.pr_url)

        return AgentRunResult(
            run_id=self.run_id,
            outcome=outcome,
            branch_name=self.branch_name,
            pr_url=self.pr_url,
            iterations_used=iterations_used,
            final_message=final_message,
        )
