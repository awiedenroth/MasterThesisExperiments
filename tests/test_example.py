import pytest

@pytest.fixture()
def x(x0):
    return x0+ 0.2

@pytest.fixture()
def y():
    return 0.1

@pytest.fixture()
def x0():
    return 0

def test_sth(x,y):
    assert x+y==pytest.approx(0.3)

def test_sth2(x,y):
    with pytest.raises(AssertionError):
        assert x+y==pytest.approx(0.6)
