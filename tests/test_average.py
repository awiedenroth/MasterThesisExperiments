import pytest
from average_calculater import calculate_average
from average_calculater import calculate_conf_average
from average_calculater import calculate_average_report

@pytest.fixture()
def data():
    return [{"a":1, "b":2, "c":3},{"a":3, "b":4, "c":5}]

@pytest.fixture()
def complex_data():
    return [[{"a":1, "b":2, "c":3}, {"c":5, "d":6, "e":7}], [{"a":3, "b":4, "c":5}, {"c":7, "d":8, "e":9}]]

@pytest.fixture()
def report_data():
    return [{'0':{"a":1, "b":2, "c":3}, '1':{"a":3, "b":4, "c":5}}, {'0':{"a":3, "b":4, "c":5}, '1':{"a":5, "b":6, "c":7}}]

def test_average_calculater(data):
    assert calculate_average(data) == {"a":2, "b":3, "c":4}

def test_complex_average(complex_data):
    assert [{"a":2, "b":3, "c":4}, {"c":6, "d":7, "e":8}] == calculate_conf_average(complex_data)

def test_report_data(report_data):
    assert {'0':{"a":2, "b":3, "c":4}, '1':{"a":4, "b":5, "c":6}} == calculate_average_report(report_data)