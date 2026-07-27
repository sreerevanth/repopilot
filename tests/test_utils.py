import pytest
from utils import calculate_average, find_max, normalize, parse_int_list

def test_calculate_average():
    assert calculate_average([1, 2, 3]) == 2.0
    assert calculate_average([10, 20]) == 15.0
    with pytest.raises(ValueError, match="Cannot calculate average of an empty list"):
        calculate_average([])

def test_find_max():
    assert find_max([1, 5, 3]) == 5
    assert find_max([-10, -5, -20]) == -5
    with pytest.raises(ValueError, match="Cannot find max of an empty list"):
        find_max([])

def test_normalize():
    assert normalize([0, 5, 10]) == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="Cannot normalize an empty list"):
        normalize([])
    with pytest.raises(ValueError, match="Cannot normalize a constant list"):
        normalize([5, 5, 5])

def test_parse_int_list():
    assert parse_int_list("1, 2, 3") == [1, 2, 3]
    assert parse_int_list("10") == [10]
    with pytest.raises(ValueError, match="Input string is empty"):
        parse_int_list("")
    with pytest.raises(ValueError, match="Input string is empty"):
        parse_int_list("   ")
    with pytest.raises(ValueError, match="invalid integer"):
        parse_int_list("1, a, 3")
