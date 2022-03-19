import pytest
from postsplit_datengenerierung import PS_Datengenerierer
import pandas as pd
import numpy as np

@pytest.fixture()
def fasttext_df():
    return pd.DataFrame.from_dict({"taetigk": ["Bibliothekar", "Projektleitung", "Büro in handwerksfirma", "Marketingforscherin/ Planerin"],"oesch8": ["7", "6", "6", "5"],
                            })
@pytest.fixture()
def fasttext_wb_df():
    return pd.DataFrame.from_dict({"taetigk": ["Bibliothekarin", "Projektleitungsfrau", "Büro in handwerksfirmen", "Marketingforscher/ Planer"],"oesch8": ["7", "6", "6", "5"],
                            })

@pytest.fixture()
def X_meta():
    #['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5', 'beab', 'einkommen', 'besch_arbzeit']
    return np.array([[-1, 2, 2, 5, 2, 2, 2, 7, 8, 1], [19, 2, 1, 5, 5, 1, 1, 7, 4, 1], [6, 2, 2, 5, 4, 2, 3, -1, -1, -1],
                       [4, 1, 1, 5, -1, -1, 1, 7, 7, 1]])

@pytest.fixture()
def X_meta_z():
    #['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5', 'beab', 'einkommen', 'besch_arbzeit']
    return np.array([[-1, 2, 2, 5, 2, 2, 3, 7, 8, 1], [19, 2, 1, 2, 5, 1, 1, 7, 4, 1], [6, 2, 2, 5, 4, 4, 3, -1, -1, -1],
                       [4, 1, 3, 5, -1, -1, 1, 7, 7, 1]])

@pytest.fixture()
def y_meta():
    return np.array([7,6,6,5])

@pytest.fixture()
def y_meta_z():
    return np.array([5,7,7,6])


@pytest.fixture()
def configuration():
    return {
        "fasttext_zusatzdaten": True,
        "meta_zusatzdaten": True,
        "selbstständige": "ohne",
        "oesch": "oesch8",
        "lowercase": True,
        "remove_stopwords": True,
        "remove_numbers": True,
        "remove_punctuation": True,
        "keyboard_aug": True,
        "random_seed": 42,
        "path_welle1": "../Daten/wic_beruf-w1_data.csv",
        "path_welle2": "../Daten/wic_beruf-w2_data.csv",
        "path_welle3": "../Daten/wic_beruf-w3_data.csv",
        "path_wb": "../Wörterbücher/wic_wörterbuch_aufbereitet_oesch.csv",
        "path_pretrained_fasttext_model": "../cc.de.300.bin"
    }

@pytest.fixture()
def train_index():
    return [1,3]

@pytest.fixture()
def test_index():
    return [0,2]



def test_meta(X_meta, y_meta, X_meta_z, y_meta_z, configuration, train_index, test_index):

    X_train_meta, y_train_meta, X_test_meta, y_test_meta= PS_Datengenerierer.make_ps_data_meta(X_meta, y_meta, X_meta_z, y_meta_z, configuration, train_index, test_index)

    assert X_train_meta.shape[0] == y_train_meta.shape[0]
    assert X_test_meta.shape[0] == y_test_meta.shape[0]

    for i in range(5):
        configuration["random_seed"] = i
        X_train_meta_n, y_train_meta_n, X_test_meta, y_test_meta = PS_Datengenerierer.make_ps_data_meta(X_meta, y_meta,
                                                X_meta_z, y_meta_z, configuration, train_index, test_index)
        with pytest.raises(AssertionError):
            assert (X_train_meta_n == X_train_meta).all()


def test_meta(fasttext_df, fasttext_wb_df, configuration, train_index, test_index):

    X_train_fasttext, y_train_fasttext, X_test_fasttext, y_test_fasttext= PS_Datengenerierer.make_ps_data_ft(fasttext_df, fasttext_wb_df, configuration, train_index, test_index)

    assert X_train_fasttext.shape[0] == y_train_fasttext.shape[0]
    assert X_test_fasttext.shape[0] == y_test_fasttext.shape[0]

    for i in range(5):
        configuration["random_seed"] = i
        X_train_fasttext_n, y_train_fasttext_n, X_test_fasttext, y_test_fasttext = PS_Datengenerierer.make_ps_data_ft(fasttext_df, fasttext_wb_df, configuration, train_index, test_index)
        with pytest.raises(AssertionError):
            assert (X_train_fasttext_n == X_train_fasttext).all()

# ungleichheit abfragen
# bei komplizierten funktionen dumm gefakte daten testen
# man kann sklearn usw funktionen vertrauen generell
# dumme tests generell
# wenn was random passieren soll: mit forschleife testen ob jedes mal was anderes rauskommt, mindest 3 mal durchlaufen lassen