"""
Module 8: Logging + Observability
Structured per-run logging with iteration tracking.
Stores prompts, LLM outputs, execution results, and final summary.
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional


def _configure_text_stream(stream):
    """Make console logging tolerant of Unicode on Windows terminals."""
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    return stream


@dataclass
class IterationRecord:
    iteration: int
    timestamp: str
    context_files: list[str]
    context_chars: int
    llm_analysis: str
    llm_confidence: float
    llm_done: bool
    changes_attempted: list[dict]
    apply_results: list[dict]
    execution_command: Optional[str]
    execution_exit_code: Optional[int]
    execution_stdout: Optional[str]
    execution_stderr: Optional[str]
    execution_timed_out: bool
    execution_success: bool
    parse_error: Optional[str]

    # Per-phase wall time. Defaulted so every existing construction site keeps
    # working -- there are several, and a required field would break them all.
    duration_ingest: float = 0.0
    duration_context: float = 0.0
    duration_llm: float = 0.0
    duration_apply: float = 0.0
    duration_execution: float = 0.0
    duration_total: float = 0.0


@dataclass
class RunRecord:
    run_id: str
    task: str
    repo_root: str
    started_at: str
    finished_at: Optional[str] = None
    outcome: str = "in_progress"  # success | failed | max_retries | error
    branch_name: Optional[str] = None
    pr_url: Optional[str] = None
    total_iterations: int = 0
    iterations: list[IterationRecord] = field(default_factory=list)
    final_error: Optional[str] = None


class _ReadableFormatter(logging.Formatter):
    """
    Timestamps ordinary lines, leaves structural ones bare.

    A separator rendered through the standard formatter comes out as

        2026-04-19 18:31:48 [INFO] ------------------------------------

    which is indented by thirty characters of metadata and no longer separates
    anything. Headers and rules are emitted without the prefix so the eye can
    find them, and everything else keeps its timestamp and level.
    """

    def __init__(self, datefmt: str):
        super().__init__("%(asctime)s [%(levelname)s] %(message)s", datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "bare", False):
            return record.getMessage()
        return super().format(record)


class AgentLogger:
    def __init__(self, log_dir: str, run_id: str, verbose: bool = True):
        self.log_dir = os.path.abspath(log_dir)
        self.run_id = run_id
        self.verbose = verbose
        os.makedirs(self.log_dir, exist_ok=True)

        self.run_log_path = os.path.join(self.log_dir, f"{run_id}.jsonl")
        self.summary_path = os.path.join(self.log_dir, f"{run_id}_summary.json")
        self.human_log_path = os.path.join(self.log_dir, f"{run_id}_human.log")

        self._logger = logging.getLogger(f"agent.{run_id}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.handlers.clear()

        console_handler = logging.StreamHandler(_configure_text_stream(sys.stdout))
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(_ReadableFormatter(datefmt="%H:%M:%S"))
        self._logger.addHandler(console_handler)

        file_handler = logging.FileHandler(self.human_log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_ReadableFormatter(datefmt="%H:%M:%S"))
        self._logger.addHandler(file_handler)

        self._run_record: Optional[RunRecord] = None

    def _ts(self) -> str:
        return datetime.now().isoformat()

    def _append_jsonl(self, data: dict):
        with open(self.run_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def start_run(self, task: str, repo_root: str):
        self._run_record = RunRecord(
            run_id=self.run_id,
            task=task,
            repo_root=repo_root,
            started_at=self._ts(),
        )
        self._logger.info(f"Run started: {self.run_id}")
        self._logger.info(f"  Task: {task[:120]}")
        self._logger.info(f"  Repo: {repo_root}")

    def section(self, title: str, rule: str = "-") -> None:
        """A header with no timestamp prefix, so it reads as a boundary."""
        self._logger.info("", extra={"bare": True})
        self._logger.info(rule * 60, extra={"bare": True})
        self._logger.info(title, extra={"bare": True})
        self._logger.info(rule * 60, extra={"bare": True})

    def start_iteration(self, iteration: int):
        # The date moved here from every line: a run happens on one day, and
        # repeating it 200 times pushed the actual message off to the right.
        self.section(
            f"Iteration {iteration}"
            f"  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
        )

    def log_context(self, files: list[str], total_chars: int):
        self._logger.debug(f"  Context: {len(files)} files, {total_chars} chars")
        for path in files[:8]:
            self._logger.debug(f"    - {path}")
        if len(files) > 8:
            self._logger.debug(f"    ... and {len(files) - 8} more")

    def log_llm_response(
        self,
        analysis: str,
        confidence: float,
        changes: list,
        done: bool,
        parse_error: Optional[str],
    ):
        self._logger.info(f"  LLM analysis: {analysis[:200]}")
        self._logger.info(f"     Confidence: {confidence:.2f} | Done: {done} | Changes: {len(changes)}")
        for change in changes:
            self._logger.info(f"     [{change.action}] {change.path}")
        if parse_error:
            self._logger.warning(f"  LLM parse error: {parse_error}")

    def log_apply_results(self, results: list):
        for result in results:
            status = "OK" if result.success else "FAIL"
            self._logger.info(f"  {status} Apply [{result.action}] {result.path}")
            if result.error:
                self._logger.warning(f"     Error: {result.error}")
            if result.backup_path:
                self._logger.debug(f"     Backup: {result.backup_path}")

    def log_execution(self, exec_result):
        status = "PASS" if exec_result.success else ("TIMEOUT" if exec_result.timed_out else "FAIL")
        sandbox = getattr(exec_result, "sandbox", "unknown")
        self._logger.info(
            f"  {status} | exit={exec_result.exit_code} | "
            f"{exec_result.duration_seconds:.1f}s | sandbox={sandbox}"
        )
        self._logger.info(f"     cmd: {exec_result.command}")
        if sandbox == "subprocess-fallback":
            self._logger.warning(
                "     This run was NOT isolated - Docker was unavailable and the "
                "command ran directly on the host."
            )
        if exec_result.stdout.strip():
            self._logger.debug(f"     stdout:\n{exec_result.stdout[:1000]}")
        if exec_result.stderr.strip():
            self._logger.debug(f"     stderr:\n{exec_result.stderr[:1000]}")

    def log_git(self, action: str, result):
        status = "OK" if result.success else "FAIL"
        self._logger.info(f"  {status} Git {action}: {result.output[:100] or result.error[:100]}")

    def record_iteration(self, record: IterationRecord):
        if record.duration_total:
            # One line rather than six: the breakdown is in the JSONL for
            # anyone measuring, and the terminal only needs the shape of it.
            self.info(
                f"  Timing: llm {record.duration_llm:.1f}s | "
                f"tests {record.duration_execution:.1f}s | "
                f"ingest {record.duration_ingest:.1f}s | "
                f"total {record.duration_total:.1f}s"
            )
        if self._run_record:
            self._run_record.iterations.append(record)
            self._run_record.total_iterations += 1
        self._append_jsonl({"type": "iteration", **asdict(record)})

    def finish_run(
        self,
        outcome: str,
        branch_name: Optional[str] = None,
        pr_url: Optional[str] = None,
        final_error: Optional[str] = None,
    ):
        if self._run_record:
            self._run_record.finished_at = self._ts()
            self._run_record.outcome = outcome
            self._run_record.branch_name = branch_name
            self._run_record.pr_url = pr_url
            self._run_record.final_error = final_error

            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self._run_record), f, indent=2, ensure_ascii=False)

        self._logger.info(f"\nRun finished: {outcome}")
        if branch_name:
            self._logger.info(f"  Branch: {branch_name}")
        if pr_url:
            self._logger.info(f"  PR: {pr_url}")
        if final_error:
            self._logger.error(f"  Error: {final_error}")
        self._logger.info(f"  Logs: {self.human_log_path}")
        self._logger.info(f"  Summary: {self.summary_path}")

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)
