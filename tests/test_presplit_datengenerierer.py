from shutil import rmtree

import pytest
import pandas as pd
from presplit_datengenerierung import Datengenerierer
from pandas.testing import assert_frame_equal
import numpy as np

@pytest.fixture()
def filepath(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    filename = tmp_path/"file.csv"
    temp_df = pd.read_csv("../Daten/wic_beruf-w1_data.csv", sep=";").head(5)
    temp_df.to_csv(filename, sep=";", index = False)
    yield str(filename)
    rmtree(tmp_path)

@pytest.fixture()
def fasttext_df():
    return pd.DataFrame.from_dict({"taetigk": ["Bibliothekar", "Projektleitung", "Büro in handwerksfirma", "Marketingforscherin/ Planerin"],"oesch8": ["7", "6", "6", "5"],
                            })

@pytest.fixture()
def X_meta():
    #['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5', 'beab', 'einkommen', 'besch_arbzeit']
    return np.array([[-1, 2, 2, 5, 2, 2, 2, 7, 8, 1], [19, 2, 1, 5, 5, 1, 1, 7, 4, 1], [6, 2, 2, 5, 4, 2, 3, -1, -1, -1],
                       [4, 1, 1, 5, -1, -1, 1, 7, 7, 1]])

@pytest.fixture()
def y_meta():
    return np.array([7,6,6,5])

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

@pytest.fixture()
def df():
    return pd.read_csv("../Daten/wic_beruf-w1_data.csv", sep=";").head(5)

def test_korrektes_einlesen_8_ohne(config):
    act_df = Datengenerierer._read_in_correct_data(config)
    assert not act_df.oesch8.isin(["1","2"]).any()

def test_korrektes_einlesen_8_nur():
    config = {"oesch": "oesch8", "selbstständige": "nur", "path_welle1": "../Daten/wic_beruf-w1_data.csv"}
    act_df = Datengenerierer._read_in_correct_data(config)
    assert act_df.oesch8.isin(["1", "2"]).all()

def test_korrektes_einlesen_16_ohne():
    config = {"oesch": "oesch16", "selbstständige": "ohne", "path_welle1": "../Daten/wic_beruf-w1_data.csv"}
    act_df = Datengenerierer._read_in_correct_data(config)
    assert not act_df.oesch16.isin(["1","2","3","4"]).any()

def test_korrektes_einlesen_16_nur():
    config = {"oesch": "oesch8", "selbstständige": "nur", "path_welle1": "../Daten/wic_beruf-w1_data.csv"}
    act_df = Datengenerierer._read_in_correct_data(config)
    assert act_df.oesch8.isin(["1", "2", "3", "4"]).all()

def test_nur_int(df):
    config = {"selbstständige": "nur", "path_welle1": "../Daten/wic_beruf-w1_data.csv"}
    clean_df = Datengenerierer._cast_to_int_and_replace_missing(config, df)
    assert clean_df[['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']].dtypes.all() == "int64"

def test_nur_int_ohne(df, config):
    clean_df = Datengenerierer._cast_to_int_and_replace_missing(config, df)
    assert clean_df[['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].dtypes.all() == "int64"

def test_select_columns_for_ft_type(df, config):
    assert type(Datengenerierer._select_columns_for_ft(config, df)) == pd.DataFrame


def test_select_columns_for_ft_sideeffects(df, config):
    df2 = df.copy()
    df1= Datengenerierer._select_columns_for_ft(config, df)
    assert_frame_equal(df, df2)
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)

def test_dtype_make_meta(df, config):
    clean_df = Datengenerierer._cast_to_int_and_replace_missing(config, df)
    X, y = Datengenerierer._make_meta_numpy(config, clean_df)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype == "int64"
    assert y.dtype == "int32"

def test_shape_make_meta(df, config):
    X, y = Datengenerierer._make_meta_numpy(config, df)
    assert X.shape[0] == y.shape[0]

def test_full_make_dataset(fasttext_df, X_meta, y_meta, filepath):
    config = {"oesch": "oesch8", "selbstständige": "ohne", "path_welle1": filepath}
    trainingsdaten_ft, X_w1_meta, y_w1_meta = Datengenerierer.make_dataset(config)
    assert_frame_equal(trainingsdaten_ft.reset_index(drop=True), fasttext_df.reset_index(drop=True), check_dtype=False)
    assert X_w1_meta.all() == X_meta.all()
    assert y_w1_meta.all() == y_meta.all()

def test_shapes_make_dataset(config):
    trainingsdaten_ft, X_w1_meta, y_w1_meta = Datengenerierer.make_dataset(config)
    assert trainingsdaten_ft.shape[0] == X_w1_meta.shape[0]
    assert trainingsdaten_ft.shape[0] == y_w1_meta.shape[0]