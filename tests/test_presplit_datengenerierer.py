import pytest
import pandas as pd
from presplit_datengenerierung import Datengenerierer

@pytest.fixture()
def df():
    return pd.read_csv("../Daten/wic_beruf-w1_data.csv", sep=";").head(10)


def test_korrektes_einlesen_8_ohne():
    config = {"oesch": "oesch8", "selbstständige": "ohne"}
    act_df = Datengenerierer._read_in_correct_data(config, path = "../Daten/wic_beruf-w1_data.csv")
    assert not act_df.oesch8.isin(["1","2"]).any()

def test_korrektes_einlesen_8_nur():
    config = {"oesch": "oesch8", "selbstständige": "nur"}
    act_df = Datengenerierer._read_in_correct_data(config, path="../Daten/wic_beruf-w1_data.csv")
    assert act_df.oesch8.isin(["1", "2"]).all()

def test_korrektes_einlesen_16_ohne():
    config = {"oesch": "oesch16", "selbstständige": "ohne"}
    act_df = Datengenerierer._read_in_correct_data(config, path = "../Daten/wic_beruf-w1_data.csv")
    assert not act_df.oesch16.isin(["1","2","3","4"]).any()

def test_korrektes_einlesen_16_nur():
    config = {"oesch": "oesch8", "selbstständige": "nur"}
    act_df = Datengenerierer._read_in_correct_data(config, path="../Daten/wic_beruf-w1_data.csv")
    assert act_df.oesch8.isin(["1", "2", "3", "4"]).all()

def test_nur_int(df):
    config = {"selbstständige" : "nur"}
    clean_df = Datengenerierer._cast_to_int_and_replace_missing(config, df)
    assert clean_df[['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']].dtypes.all() == "int64"

def test_nur_int_ohne(df):
    config = {"selbstständige" : "ohne"}
    clean_df = Datengenerierer._cast_to_int_and_replace_missing(config, df)
    assert clean_df[['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].dtypes.all() == "int64"