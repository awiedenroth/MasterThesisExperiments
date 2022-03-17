import pytest
from data_cleaning import clean_data
from data_cleaning import remove_punctuation
from data_cleaning import remove_stopwords
from data_cleaning import remove_numbers
import pandas as pd

@pytest.fixture()
def config():
    return {

        "lowercase": True,
        "remove_stopwords": True,
        "remove_numbers": True,
        "remove_punctuation": True,
        "keyboard_aug": True,
        "random_seed": 42,
    }

@pytest.fixture()
def df():
    return pd.read_csv("../Daten/wic_beruf-w1_data.csv", sep=";").head(5)

#Todo: fehlenden Datenpunkt hinzufügen, der wird ja nicht rausgeschmissen weil es direkt eingelesen wird
@pytest.fixture()
def fasttext_df():
    return pd.DataFrame.from_dict({"taetigk": ["bibliothekar", "projektleitung", "büro handwerksfirma", "marketingforscherin planerin"],"oesch8": ["7", "6", "6", "5"],
                            })

@pytest.fixture()
def text():
    return "Ich arbeite (freiberuflich) 3 mal die Woche in der Hochschule!"

def test_stopwords(text):
    assert remove_stopwords(text) == "Ich arbeite (freiberuflich) 3 mal Woche Hochschule!"

def test_numbers(text):
    assert remove_numbers(text) == "Ich arbeite (freiberuflich)  mal die Woche in der Hochschule!"

def test_punctuation(text):
    assert remove_punctuation(text) == "Ich arbeite freiberuflich 3 mal die Woche in der Hochschule"

def test_cleaning(df, config):
    assert clean_data(df, config)
