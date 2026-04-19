"""Tests for utils.py — these tests FAIL until the bugs are fixed."""

import pytest
from utils import calculate_average, find_max, normalize, parse_int_list


class TestCalculateAverage:
    def test_basic(self):
        assert calculate_average([1, 2, 3, 4, 5]) == 3.0

    def test_single(self):
        assert calculate_average([42]) == 42.0

    def test_empty_list(self):
        """Should raise ValueError, not ZeroDivisionError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_average([])

    def test_floats(self):
        assert abs(calculate_average([1.5, 2.5, 3.0]) - 2.333) < 0.01


class TestFindMax:
    def test_basic(self):
        assert find_max([3, 1, 4, 1, 5, 9]) == 9

    def test_negative(self):
        assert find_max([-10, -3, -7]) == -3

    def test_empty_list(self):
        """Should raise ValueError with a helpful message."""
        with pytest.raises(ValueError, match="empty"):
            find_max([])


class TestNormalize:
    def test_basic(self):
        result = normalize([0, 5, 10])
        assert result == [0.0, 0.5, 1.0]

    def test_all_same(self):
        """Should raise ValueError when all values are identical."""
        with pytest.raises(ValueError, match="constant"):
            normalize([5, 5, 5])

    def test_single(self):
        """Single-element list: already normalized."""
        with pytest.raises(ValueError, match="constant"):
            normalize([7])


class TestParseIntList:
    def test_basic(self):
        assert parse_int_list("1,2,3") == [1, 2, 3]

    def test_spaces(self):
        assert parse_int_list("10, 20, 30") == [10, 20, 30]

    def test_invalid_input(self):
        """Should raise ValueError with a clear message."""
        with pytest.raises(ValueError, match="invalid"):
            parse_int_list("1,two,3")

    def test_empty_string(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_int_list("")
