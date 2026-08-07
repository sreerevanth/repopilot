import pytest
from utils import calculate_average, find_max, normalize, parse_int_list

def test_calculate_average():
    assert calculate_average([1, 2, 3, 4, 5]) == 3.0
    assert calculate_average([10, 20]) == 15.0
    with pytest.raises(ValueError, match="Cannot calculate average of an empty list"):
        calculate_average([])

def test_find_max():
    assert find_max([1, 5, 3, 9, 2]) == 9
    assert find_max([-10, -5, -20]) == -5
    with pytest.raises(ValueError, match="Cannot find max of an empty list"):
        find_max([])

def test_normalize():
    assert normalize([0, 5, 10]) == [0.0, 0.5, 1.0]
    assert normalize([10, 20, 30, 40, 50]) == [0.0, 0.25, 0.5, 0.75, 1.0]
    
    with pytest.raises(ValueError, match="Cannot normalize an empty list"):
        normalize([])
        
    with pytest.raises(ValueError, match="Cannot normalize a constant list"):
        normalize([5, 5, 5])

def test_parse_int_list():
    assert parse_int_list("1,2,3,4") == [1, 2, 3, 4]
    assert parse_int_list(" 10 , 20, 30 ") == [10, 20, 30]
    assert parse_int_list("-5,0,5") == [-5, 0, 5]
    
    with pytest.raises(ValueError, match="Input string is empty"):
        parse_int_list("")
        
    with pytest.raises(ValueError, match="Input string is empty"):
        parse_int_list("   ")
        
    with pytest.raises(ValueError, match="invalid integer in input"):
        parse_int_list("1,2,abc,4")
