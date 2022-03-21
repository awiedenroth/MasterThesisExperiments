import pytest
from average_calculater import calculate_average
from average_calculater import calculate_conf_average

@pytest.fixture()
def data():
    return [{"a":1, "b":2, "c":3},{"a":3, "b":4, "c":5}]

@pytest.fixture()
def complex_data():
    return [[{"a":1, "b":2, "c":3}, {"c":5, "d":6, "e":7}], [{"a":3, "b":4, "c":5}, {"c":7, "d":8, "e":9}]]

def test_average_calculater(data):
    assert calculate_average(data) == {"a":2, "b":3, "c":4}

def test_complex_average(complex_data):
    assert [{"a":2, "b":3, "c":4}, {"c":6, "d":7, "e":8}] == calculate_conf_average(complex_data)