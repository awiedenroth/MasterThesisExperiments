import pytest
from data_cleaning import clean_data
from data_cleaning import remove_punctuation
from data_cleaning import remove_stopwords
from data_cleaning import remove_numbers

@pytest.fixture()
def text():
    return "Ich arbeite (freiberuflich) 3 mal die Woche in der Hochschule!"

def test_stopwords(text):
    assert remove_stopwords(text) == "Ich arbeite (freiberuflich) 3 mal Woche Hochschule!"

def test_numbers(text):
    assert remove_numbers(text) == "Ich arbeite (freiberuflich)  mal die Woche in der Hochschule!"

def test_punctuation(text):
    assert remove_punctuation(text) == "Ich arbeite freiberuflich 3 mal die Woche in der Hochschule"
