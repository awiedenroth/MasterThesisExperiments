import pytest
from zusatzdatengenerierung import Zusatzdatengenerierer
from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np

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
    fasttext_wb_df, X_meta, y_meta= Zusatzdatengenerierer.make_dataset(config)
    assert X_meta.shape[0] == y_meta.shape[0]

def test_shape_ft_wb(config):
    fasttext_wb_df, X_meta, y_meta= Zusatzdatengenerierer.make_dataset(config)
    if config["selbstständige"] == "ohne":
        assert type(fasttext_wb_df) == pd.DataFrame

def test_dtype_meta(config):
    fasttext_wb_df, X_meta, y_meta = Zusatzdatengenerierer.make_dataset(config)
    assert isinstance(X_meta, np.ndarray)
    assert isinstance(y_meta, np.ndarray)
    assert X_meta.dtype == "int64"
    assert y_meta.dtype == "int32"

def test_content_of_wb_oesch8(config):
    config["oesch"] = "oesch8"
    config["selbstständige"] = "ohne"
    fasttext_wb_df, X_meta, y_meta = Zusatzdatengenerierer.make_dataset(config)
    assert fasttext_wb_df[config["oesch"]].dtypes == "int64"
    assert fasttext_wb_df.oesch8.isin([3,4,5,6,7,8]).all()

def test_content_of_wb_oesch16(config):
    config["oesch"] = "oesch16"
    config["selbstständige"] = "ohne"
    fasttext_wb_df, X_meta, y_meta = Zusatzdatengenerierer.make_dataset(config)
    assert fasttext_wb_df[config["oesch"]].dtypes == "int64"
    assert fasttext_wb_df.oesch16.isin([5,6,7,8,9,10,11,12,13,14,15,16]).all()