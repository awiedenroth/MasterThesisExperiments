import pytest
from zusatzdatengenerierung import Zusatzdatengenerierer
from pandas.testing import assert_frame_equal
import pandas as pd



@pytest.fixture()
def config():
    return {
    "fasttext_zusatzdaten": False,
    "meta_zusatzdaten" : False,
    "selbstständige" : "ohne",
    "oesch" : "oesch8",
    "lowercase" : True,
    "remove_stopwords": True,
    "remove_numbers": True,
    "remove_punctuation": True,
    "keyboard_aug" : True,
    "random_seed": 42,
    "path_welle1": "../Daten/wic_beruf-w1_data.csv",
    "path_welle2": "../Daten/wic_beruf-w2_data.csv",
    "path_welle3": "../Daten/wic_beruf-w3_data.csv",
    "path_wb": "../Wörterbücher/wic_wörterbuch_aufbereitet_oesch.csv"
    }

def test_shapes_equal_make_dataset(config):
    trainingsdaten_ft, X_w1_meta, y_w1_meta = Zusatzdatengenerierer.make_dataset(config)
    assert X_w1_meta.shape[0] == y_w1_meta.shape[0]

def test_shape_ft_wb(config):
    trainingsdaten_ft, X_w1_meta, y_w1_meta = Zusatzdatengenerierer.make_dataset(config)
    assert type(trainingsdaten_ft) == pd.DataFrame