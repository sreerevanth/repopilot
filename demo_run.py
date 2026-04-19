"""
demo_run.py
End-to-end demonstration of the agent loop using a MockLLMClient
that returns a real, correct fix on the first call.

This proves every layer works:
  ingestion → context → (mock LLM) → modification → sandbox execution → git → logging
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.repo_ingestion import ingest_repository
from modules.context_builder import build_context
from modules.llm_client import LLMResponse, FileChange
from modules.code_modifier import CodeModificationEngine
from modules.sandbox import SubprocessSandbox
from modules.git_integration import GitIntegration
from modules.logger import AgentLogger, IterationRecord
from datetime import datetime
import uuid

# ─────────────────────────────────────────────────────────────
# The correct fix the LLM would produce (pre-computed)
# ─────────────────────────────────────────────────────────────
FIXED_UTILS_CONTENT = '''"""Utility functions for the sample application."""


def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot calculate average of an empty list")
    total = sum(numbers)
    return total / len(numbers)


def find_max(numbers):
    """Find the maximum value in a list."""
    if not numbers:
        raise ValueError("Cannot find max of an empty list")
    return max(numbers)


def normalize(numbers):
    """Normalize a list of numbers to [0, 1] range."""
    if not numbers:
        raise ValueError("Cannot normalize an empty list")
    min_val = min(numbers)
    max_val = max(numbers)
    if max_val == min_val:
        raise ValueError("Cannot normalize a constant list (all values are equal)")
    return [(x - min_val) / (max_val - min_val) for x in numbers]


def parse_int_list(s):
    """Parse a comma-separated string of integers."""
    if not s or not s.strip():
        raise ValueError("Input string is empty")
    try:
        return [int(x.strip()) for x in s.split(",")]
    except ValueError as e:
        raise ValueError(f"invalid integer in input: {e}") from e
'''

MOCK_LLM_RESPONSE = LLMResponse(
    raw="{}",
    analysis=(
        "The utils.py functions lack input validation. "
        "I need to add empty-list guards to calculate_average and find_max, "
        "add a constant-check to normalize, and add proper error handling to parse_int_list."
    ),
    changes=[
        FileChange(
            path="utils.py",
            action="modify",
            content=FIXED_UTILS_CONTENT,
            explanation="Added ValueError guards for all four functions per test expectations",
        )
    ],
    confidence=0.97,
    done=True,
)


class MockLLMClient:
    """Replaces the real LLMClient for offline demo/testing."""

    def __init__(self):
        self.call_count = 0

    def initial_request(self, task: str, context_str: str) -> LLMResponse:
        self.call_count += 1
        print(f"  [MockLLM] initial_request called (call #{self.call_count})")
        print(f"  [MockLLM] Context size: {len(context_str)} chars")
        return MOCK_LLM_RESPONSE

    def retry_request(self, task, context_str, previous_changes, stdout, stderr, exit_code):
        self.call_count += 1
        print(f"  [MockLLM] retry_request called (call #{self.call_count})")
        return MOCK_LLM_RESPONSE


# ─────────────────────────────────────────────────────────────
# Manual agent loop (same logic as agent_loop.py, but with mock LLM)
# ─────────────────────────────────────────────────────────────

def run_demo(repo_root: str, task: str, max_iterations: int = 3):
    repo_root = os.path.abspath(repo_root)
    run_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    log_dir = os.path.join(repo_root, "logs")
    backup_dir = os.path.join(repo_root, "backups")

    logger = AgentLogger(log_dir, run_id, verbose=True)
    llm = MockLLMClient()
    modifier = CodeModificationEngine(repo_root, backup_dir)
    sandbox = SubprocessSandbox(repo_root, timeout_seconds=60)

    git = None
    branch_name = None
    try:
        git = GitIntegration(repo_root)
        branch_name = f"agent/demo-fix-{run_id[-6:]}"
        res = git.create_branch(branch_name, "master")
        logger.log_git(f"create branch '{branch_name}'", res)
    except Exception as e:
        logger.warning(f"Git init failed (non-fatal): {e}")

    logger.start_run(task, repo_root)

    last_exec = None
    last_changes = []
    last_apply = []
    outcome = "failed"

    for iteration in range(1, max_iterations + 1):
        logger.start_iteration(iteration)

        # 1. Ingest
        repo = ingest_repository(repo_root)

        # 2. Context
        ctx = build_context(repo, task)
        logger.log_context([f.path for f in ctx.files], ctx.total_chars)
        context_str = ctx.render()

        # 3. LLM call
        if iteration == 1 or last_exec is None:
            resp = llm.initial_request(task, context_str)
        else:
            resp = llm.retry_request(
                task, context_str, last_changes,
                last_exec.stdout, last_exec.stderr, last_exec.exit_code
            )

        logger.log_llm_response(
            resp.analysis, resp.confidence, resp.changes, resp.done, resp.parse_error
        )

        # 4. Validate + Apply
        errors = modifier.verify_changes(resp.changes)
        if errors:
            logger.warning(f"Validation errors: {errors}")

        apply_results = modifier.apply_changes(resp.changes)
        logger.log_apply_results(apply_results)
        last_changes = resp.changes
        last_apply = apply_results

        # 5. Execute
        exec_result = sandbox.run_tests("pytest")
        logger.log_execution(exec_result)
        last_exec = exec_result

        # Build iteration record
        iter_record = IterationRecord(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            context_files=[f.path for f in ctx.files],
            context_chars=ctx.total_chars,
            llm_analysis=resp.analysis,
            llm_confidence=resp.confidence,
            llm_done=resp.done,
            changes_attempted=[
                {"path": c.path, "action": c.action, "explanation": c.explanation}
                for c in resp.changes
            ],
            apply_results=[
                {"path": r.path, "action": r.action, "success": r.success, "error": r.error}
                for r in apply_results
            ],
            execution_command=exec_result.command,
            execution_exit_code=exec_result.exit_code,
            execution_stdout=exec_result.stdout[:2000],
            execution_stderr=exec_result.stderr[:2000],
            execution_timed_out=exec_result.timed_out,
            execution_success=exec_result.success,
            parse_error=resp.parse_error,
        )
        logger.record_iteration(iter_record)

        # 6. Success?
        if exec_result.success:
            logger.info(f"\nTests passed on iteration {iteration}!")

            # Commit
            if git:
                git.stage_all()
                commit_res = git.commit(
                    f"agent: fix utils.py validation\n\nrun_id: {run_id}",
                    author="Agent Bot <agent@autonomous.dev>"
                )
                logger.log_git("commit", commit_res)

            outcome = "success"
            break

        if iteration == max_iterations:
            outcome = "max_retries"

    if outcome != "success" and last_apply:
        logger.warning("Rolling back changes...")
        restored = modifier.rollback(last_apply)
        logger.info(f"Rolled back: {restored}")

    logger.finish_run(outcome, branch_name, None)

    print(f"\n{'='*60}")
    print(f"FINAL OUTCOME : {outcome.upper()}")
    print(f"RUN ID        : {run_id}")
    print(f"LLM CALLS     : {llm.call_count}")
    if branch_name:
        print(f"GIT BRANCH    : {branch_name}")
    print(f"LOGS DIR      : {log_dir}")
    print(f"{'='*60}")
    return outcome


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "../sample_repo"
    task = (
        "Fix all failing tests in test_utils.py. "
        "Add proper ValueError guards to calculate_average, find_max, normalize, and parse_int_list."
    )
    result = run_demo(repo, task)
    sys.exit(0 if result == "success" else 1)
