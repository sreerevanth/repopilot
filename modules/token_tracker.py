"""
Module: Token Tracker
Tracks token usage and estimates costs for LLM API calls.
"""

from dataclasses import dataclass

# Cost per 1k tokens (input, output) in USD
PRICING_TABLE = {
    # Anthropic
    "claude-3-opus-20240229": (15.00 / 1000, 75.00 / 1000),
    "claude-3-5-sonnet-20240620": (3.00 / 1000, 15.00 / 1000),
    "claude-sonnet-4-20250514": (3.00 / 1000, 15.00 / 1000),  # Assuming same as 3.5
    "claude-3-haiku-20240307": (0.25 / 1000, 1.25 / 1000),
    
    # OpenAI
    "gpt-4o": (5.00 / 1000, 15.00 / 1000),
    "gpt-4-turbo": (10.00 / 1000, 30.00 / 1000),
    "gpt-3.5-turbo": (0.50 / 1000, 1.50 / 1000),
    
    # Gemini
    "gemini-1.5-pro": (3.50 / 1000, 10.50 / 1000),
    "gemini-1.5-flash": (0.35 / 1000, 1.05 / 1000),
    "gemini-2.0-flash": (0.35 / 1000, 1.05 / 1000), # Assuming same as 1.5 flash
}


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TokenTracker:
    """Tracks token usage and estimates costs across a session."""
    
    def __init__(self):
        self.total_usage = TokenUsage()
        self.calls = 0
        self.cost_usd = 0.0
        
    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Add usage from a single API call and update cost estimates."""
        self.calls += 1
        self.total_usage.input_tokens += input_tokens
        self.total_usage.output_tokens += output_tokens
        self.total_usage.total_tokens += (input_tokens + output_tokens)
        
        # Calculate cost
        pricing = PRICING_TABLE.get(model)
        if pricing:
            in_cost_per_1k, out_cost_per_1k = pricing
            self.cost_usd += (input_tokens / 1000.0) * in_cost_per_1k
            self.cost_usd += (output_tokens / 1000.0) * out_cost_per_1k
            
    def get_summary(self) -> str:
        """Get a formatted string summarizing usage and cost."""
        lines = [
            "── Token Usage Summary ──",
            f"API Calls    : {self.calls}",
            f"Input Tokens : {self.total_usage.input_tokens:,}",
            f"Output Tokens: {self.total_usage.output_tokens:,}",
            f"Total Tokens : {self.total_usage.total_tokens:,}",
        ]
        
        if self.cost_usd > 0:
            lines.append(f"Est. Cost    : ${self.cost_usd:.4f}")
        else:
            lines.append("Est. Cost    : Unknown model / Local")
            
        return "\n".join(lines)
