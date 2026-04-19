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
    def __init__(self, api_key: Optional[str] = None):
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)

    def _call(self, prompt: str, retries: int = 3) -> str:
        """Raw API call with retry on transient errors."""
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
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
