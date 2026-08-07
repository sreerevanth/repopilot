"""Utility functions for the sample application."""


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
