import pytest
import pandas as pd
from shutil import rmtree

@pytest.fixture()
def x(x0):
    return x0+ 0.2

@pytest.fixture()
def y():
    return 0.1

@pytest.fixture()
def x0():
    return 0

@pytest.fixture()
def file(tmp_path):
    tmp_path.mkdir()
    filename = tmp_path/"file.csv"
    pd.DataFrame.from_dict({"col": [1,2,3], "col2": [1.0, 2.0, 3.0]}).to_csv(filename, sep=";")
    yield str(filename)
    tmp_path.rmdir()

def test_sth(x,y):
    assert x+y==pytest.approx(0.3)

def test_sth2(x,y):
    with pytest.raises(AssertionError):
        assert x+y==pytest.approx(0.6)

# ungleichheit abfragen
# bei komplizierten funktionen dumm gefakte daten testen
# man kann sklearn usw funktionen vertrauen generell
# dumme tests generell
# wenn was random passieren soll: mit forschleife testen ob jedes mal was anderes rauskommt, mindest 3 mal durchlaufen lassen