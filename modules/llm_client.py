"""
Module 3: LLM Interaction Layer
Clean prompt engineering, structured I/O, token budget handling,
iterative refinement support. Uses Anthropic Claude API.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192

# USD per million tokens. Published rates, kept in one place so a price change
# is a one-line edit rather than a hunt. An unknown model falls back to the
# default rather than silently costing nothing -- a budget that reports $0.00
# for an unrecognised model is worse than no budget at all.
PRICING_PER_MTOK = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


class BudgetExceededError(RuntimeError):
    """Raised when the accumulated spend has reached --max-cost."""


@dataclass
class UsageTracker:
    """Running token and cost total for one client."""

    model: str = MODEL
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def pricing(self) -> dict:
        return PRICING_PER_MTOK.get(self.model, DEFAULT_PRICING)

    @property
    def cost_usd(self) -> float:
        rates = self.pricing
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
        ) / 1_000_000

    def record(self, response) -> None:
        """Add one API response. Tolerates a client that reports no usage."""
        usage = getattr(response, "usage", None)
        self.calls += 1
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def summary(self) -> str:
        return (
            f"{self.calls} call(s), {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out tokens, ${self.cost_usd:.4f}"
        )

# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert software engineer embedded in an autonomous code modification pipeline.
You receive:
1. A task description
2. Relevant source files from the repository
3. (Optionally) error output from a previous execution attempt

Your job is to produce ONLY the file changes required to complete the task.

OUTPUT FORMAT (STRICT — machine-parsed):
Return a JSON object with this exact schema:

{
  "analysis": "<brief explanation of what you understood and what changes are needed>",
  "changes": [
    {
      "path": "<relative file path from repo root>",
      "action": "modify" | "create" | "delete",
      "content": "<full new file content (for modify/create)>",
      "explanation": "<why this change>"
    }
  ],
  "confidence": <0.0-1.0 float>,
  "done": <true if you believe the task is complete, false if more iterations needed>
}

RULES:
- For "modify" and "create": always provide the COMPLETE file content, not diffs or snippets.
- For "delete": omit "content".
- Do NOT include markdown fences, explanation text, or anything outside the JSON object.
- Paths must be relative (e.g., "src/utils.py"), never absolute.
- If you cannot determine a fix, set confidence < 0.3 and done=false with a clear analysis.
- Preserve existing code style, indentation, and conventions.
"""

TASK_PROMPT_TEMPLATE = """\
## Task
{task}

## Repository Context
{context}

## Instructions
Analyze the code and produce the minimal changes needed to complete the task.
Return ONLY the JSON object specified in the system prompt.
"""

RETRY_PROMPT_TEMPLATE = """\
## Task
{task}

## Previous Changes Applied
The following files were modified in the previous iteration:
{previous_changes_summary}

## Execution Result (FAILED)
Exit code: {exit_code}

### stdout:
{stdout}

### stderr:
{stderr}

## Current Repository Context
{context}

## Instructions
The previous attempt failed. Analyze the error output carefully and produce corrected changes.
Focus on the root cause of the failure. Return ONLY the JSON object.
"""


# ─────────────────────────────────────────────
# Response types
# ─────────────────────────────────────────────

@dataclass
class FileChange:
    path: str
    action: str      # "modify" | "create" | "delete"
    content: str     # full file content
    explanation: str


@dataclass
class LLMResponse:
    raw: str
    analysis: str
    changes: list[FileChange]
    confidence: float
    done: bool
    parse_error: Optional[str] = None


# ─────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────

class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_cost_usd: Optional[float] = None,
        model: str = MODEL,
    ):
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_cost_usd = max_cost_usd
        self.usage = UsageTracker(model=model)

    def _check_budget(self) -> None:
        """
        Stop before the next call once the limit is reached.

        The cost of a call is only known after it returns, so this is a stop
        condition rather than a pre-authorisation: spend can overshoot the limit
        by at most one call. Set the limit slightly below what you can actually
        afford.
        """
        if self.max_cost_usd is None:
            return
        if self.usage.cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"Cost limit reached: spent ${self.usage.cost_usd:.4f} of "
                f"${self.max_cost_usd:.2f} after {self.usage.summary()}. "
                f"Stopping before the next call."
            )

    def _call(self, prompt: str, retries: int = 3) -> str:
        """Raw API call with retry on transient errors."""
        self._check_budget()
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.usage.record(response)
                return response.content[0].text
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError("LLM call failed after retries")

    def _parse_response(self, raw: str) -> LLMResponse:
        """Extract and parse JSON from LLM output."""
        # Strip any accidental markdown fences
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Find the outermost JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return LLMResponse(
                raw=raw, analysis="", changes=[], confidence=0.0, done=False,
                parse_error=f"No JSON object found in response: {raw[:300]}"
            )

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            return LLMResponse(
                raw=raw, analysis="", changes=[], confidence=0.0, done=False,
                parse_error=f"JSON parse error: {e}\nText: {text[start:end][:500]}"
            )

        changes = []
        for c in data.get("changes", []):
            changes.append(FileChange(
                path=c.get("path", ""),
                action=c.get("action", "modify"),
                content=c.get("content", ""),
                explanation=c.get("explanation", ""),
            ))

        return LLMResponse(
            raw=raw,
            analysis=data.get("analysis", ""),
            changes=changes,
            confidence=float(data.get("confidence", 0.5)),
            done=bool(data.get("done", False)),
        )

    def initial_request(self, task: str, context_str: str) -> LLMResponse:
        """First-pass: analyze task and produce code changes."""
        prompt = TASK_PROMPT_TEMPLATE.format(task=task, context=context_str)
        raw = self._call(prompt)
        return self._parse_response(raw)

    def retry_request(
        self,
        task: str,
        context_str: str,
        previous_changes: list[FileChange],
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> LLMResponse:
        """Retry after a failed execution — feed error output back."""
        prev_summary = "\n".join(
            f"  - [{c.action}] {c.path}: {c.explanation}"
            for c in previous_changes
        )

        prompt = RETRY_PROMPT_TEMPLATE.format(
            task=task,
            previous_changes_summary=prev_summary or "  (none)",
            exit_code=exit_code,
            stdout=stdout[:4000] if stdout else "(empty)",
            stderr=stderr[:4000] if stderr else "(empty)",
            context=context_str,
        )
        raw = self._call(prompt)
        return self._parse_response(raw)
