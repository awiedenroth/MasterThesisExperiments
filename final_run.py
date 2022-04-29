from typing import Union, Dict, Any
import time

from matplotlib import pyplot as plt
from tqdm import tqdm
from postsplit_datengenerierung import PS_Datengenerierer
from presplit_datengenerierung import Datengenerierer
from final_zusatzdatengenerierung import Zusatzdatengenerierer
from data_cleaning import clean_data
from modelltraining import train_ft
from modelltraining import train_meta
from modelltraining import train_combi
from evaluation import Evaluierer
from sklearn.model_selection import KFold
import sys
import warnings
from pprint import pprint
import wandb
import pickle
import json
from caching import mem
from average_calculater import calculate_average
from average_calculater import calculate_conf_average
from average_calculater import calculate_average_report
import pandas as pd
import numpy as np

import os
WANDB_API_KEY = "5a0c9d2c04e141e969e5f63e2c722fe7737846c2"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

if not sys.warnoptions:
    warnings.simplefilter("ignore")

DEFAULT_CONFIG = {
    "fasttext_zusatzdaten": False,
    "meta_zusatzdaten" : True,
    "selbstständige" : "nur",
    "oesch" : "oesch8",
    "lowercase" : False,
    "remove_stopwords": True,
    "remove_num_punc": False,
    "keyboard_aug" : True,
    "random_seed": 444,
    "path_welle1": "./Daten/welle1_und_3.csv",
    "path_welle2": "./Daten/wic_beruf-w2_data.csv",
    #"path_welle3": "./Daten/wic_beruf-w4_data.csv",
    "path_wb": "./Wörterbücher/Wörterbuch_binär.csv",
    "path_pretrained_fasttext_model": "cc.de.300.bin",
    "k_fold_splits": 0,
    "ft_model": "nn",
    "meta_model": "xgboost",
    "combi_model": "linear" # "xgboost" oder "nn" oder "linear"
}


def main():
    run = wandb.init(project="test_data", entity="awiedenroth", config=DEFAULT_CONFIG)
    print(wandb.config)
    configuration = DEFAULT_CONFIG
    #configuration = {k:v for k,v in wandb.config.items()}
    assert isinstance(configuration["remove_stopwords"], bool)

    # ich erzeuge für fasttext und meta jeweils die grunddaten
    fasttext_df1, X_meta1, y_meta1 = Datengenerierer.make_dataset(configuration)
    conf2 = configuration.copy()
    conf2.update({"path_welle1": "./TestDataset/welle4.csv"})
    fasttext_df2, X_meta2, y_meta2 = Datengenerierer.make_dataset(conf2)
    train_len = len(fasttext_df1)
    test_len = len(fasttext_df2)
    assert train_len == len(X_meta1)
    assert test_len == len(X_meta2)
    fasttext_df = pd.concat([fasttext_df1, fasttext_df2])
    X_meta = np.concatenate((X_meta1, X_meta2), axis=0)
    y_meta = np.concatenate((y_meta1, y_meta2), axis=0)

    train_index = np.arange(train_len)
    test_index = np.arange(train_len, (test_len+train_len))
    assert fasttext_df.iloc[train_index].equals(fasttext_df1)
    assert fasttext_df.iloc[test_index].equals(fasttext_df2)
    assert np.array_equal(X_meta[train_index], X_meta1)
    assert np.array_equal(X_meta[test_index], X_meta2)
    assert np.array_equal(y_meta[train_index], y_meta1)
    assert np.array_equal(y_meta[test_index], y_meta2)

    # ich erzeuge die Zusatzdaten für fasttext und meta
    fasttext_wb_df, X_meta_z, y_meta_z = Zusatzdatengenerierer.make_dataset(configuration)

    fasttext_df = clean_data(fasttext_df, configuration)
    if configuration["fasttext_zusatzdaten"] == True and configuration["selbstständige"] == "ohne":
        fasttext_wb_df = clean_data(fasttext_wb_df, configuration)
    #dict zum abspeichern der ergebnisse
    ergebnisse = []
    meta_ergebnisse = []
    ft_ergebnisse = []
    combi_ergebnisse = []
    meta_ergebnisse_train = []
    ft_ergebnisse_train = []
    combi_ergebnisse_train = []
    meta_ergebnisse_val = []
    ft_ergebnisse_val = []
    combi_ergebnisse_val = []
    combi_conf_ergebnisse = []
    # index um zu tracken bei welchem durchgang man ist
    i = 0
    #kf = KFold(n_splits=configuration["k_fold_splits"], shuffle = True, random_state = configuration["random_seed"])

    # die for schleife geht k mal durch
    #for train_index, test_index in kf.split(fasttext_df):
    start_time = time.time()
    print("Durchgang ", i)

    X_train_fasttext, y_train_fasttext, X_test_fasttext, y_test_fasttext = \
        PS_Datengenerierer.make_ps_data_ft(fasttext_df, fasttext_wb_df, configuration, train_index, test_index)

    X_train_meta, y_train_meta, X_test_meta, y_test_meta = \
        PS_Datengenerierer.make_ps_data_meta(X_meta, y_meta, X_meta_z, y_meta_z, configuration, train_index, test_index)

    print("Gesamtanzahl Datenpunkte Grunddaten vor k-split = ", len(fasttext_df))
    print("Anzahl Trainingsdaten ohne Zusatzdaten", len(fasttext_df)-len(X_test_fasttext))
    print("Anzahl Fasttext Trainingsdaten inklusive Zusatzdaten", len(X_train_fasttext))
    print("Anzahl fasttext Validierungsdaten = ", len(X_test_fasttext))
    print("Anzahl meta Trainingsdaten inklusive Zusatzdaten = ", len(X_train_meta))
    print("Anzahl meta Validierungsdaten = ", len(X_test_meta))

    print("trainiere meta Modell")

    meta_model = train_meta(X_train_meta, y_train_meta, configuration)
    evaluation_meta, evaluation_meta_train, evaluation_meta_val = Evaluierer.make_evaluation(meta_model, X_train_meta, y_train_meta,
                                                     X_test_meta, y_test_meta, configuration, modelname = "meta", run = i)
    meta_ergebnisse.append(evaluation_meta)
    meta_ergebnisse_train.append(evaluation_meta_train)
    meta_ergebnisse_val.append(evaluation_meta_val)

    print("trainiere fasttext Modell")
    fasttext_model = train_ft(X_train_fasttext, y_train_fasttext, configuration)
    evaluation_fasttext, evaluation_fasttext_train, evaluation_fasttext_val = Evaluierer.make_evaluation(fasttext_model, X_train_fasttext, y_train_fasttext,
                               X_test_fasttext, y_test_fasttext, configuration, modelname = "fasttext", run = i)
    ft_ergebnisse.append(evaluation_fasttext)
    ft_ergebnisse_train.append(evaluation_fasttext_train)
    ft_ergebnisse_val.append(evaluation_fasttext_val)

    X_train_combi, y_train_combi, X_test_combi, y_test_combi = \
        PS_Datengenerierer.make_ps_data_combi(fasttext_df, X_meta, fasttext_model, meta_model, train_index, test_index, configuration)

    print("trainiere Combi Modell")

    combi_model = train_combi(X_train_combi, y_train_combi, configuration)
    evaluation_combi, evaluation_combi_train, evaluation_combi_val = Evaluierer.make_evaluation(combi_model, X_train_combi, y_train_combi,
                                                    X_test_combi, y_test_combi, configuration, modelname = "combi", run = i)
    combi_ergebnisse.append(evaluation_combi)
    combi_ergebnisse_train.append(evaluation_combi_train)
    combi_ergebnisse_val.append(evaluation_combi_val)

    evaluation_combi_confidence = []
    for confidence in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.992,0.994,0.995, 0.996, 0.997, 0.998, 0.999]:
        evaluation_combi_confidence.append(Evaluierer.make_evaluation_confidence(combi_model, X_train_combi, y_train_combi, X_test_combi, y_test_combi, confidence, run=i))
    combi_conf_ergebnisse.append(evaluation_combi_confidence)

    plt.clf()
    plt.cla()
    zeit = round(time.time() - start_time) / 60
    wandb.log({f"Dauer Durchlauf {i}min: ": zeit})
    i = i+1

    """
    # hier berechne ich die averages von den einzelnen modell-evaluationen und speicher diese weg, diese sind das wichtigste für meine Auswertung
    meta_average = calculate_average(meta_ergebnisse)
    ft_average = calculate_average(ft_ergebnisse)
    combi_average = calculate_average(combi_ergebnisse)

    meta_train_average = calculate_average_report(meta_ergebnisse_train)
    ft_train_average = calculate_average_report(ft_ergebnisse_train)
    combi_train_average = calculate_average_report(combi_ergebnisse_train)

    meta_val_average = calculate_average_report(meta_ergebnisse_val)
    ft_val_average = calculate_average_report(ft_ergebnisse_val)
    combi_val_average = calculate_average_report(combi_ergebnisse_val)
    """
    combi_conf_average = calculate_conf_average(combi_conf_ergebnisse)

    wandb.log({"meta average": evaluation_meta})
    wandb.log({"fasttext average": evaluation_fasttext})
    wandb.log({"combi average": evaluation_combi})

    wandb.log({"meta train report average": evaluation_meta_train})
    wandb.log({"fasttext train report average": evaluation_fasttext_train})
    wandb.log({"combi train report average": evaluation_combi_train})

    wandb.log({"meta val report average": evaluation_meta_val})
    wandb.log({"fasttext val report average": evaluation_fasttext_val})
    wandb.log({"combi val report average": evaluation_combi_val})

    for conf in combi_conf_average:
        wandb.log({f"confidence average @{conf['confidence: ']}": conf})

    wandb.log({"Anzahl Datenpunkte Grunddaten vor k-split = ": len(fasttext_df),
               "Anzahl Trainingsdaten ohne Zusatzdaten": len(fasttext_df) - len(X_test_fasttext),
               "Anzahl Trainingsdaten inklusive Zusatzdaten": len(X_train_fasttext),
               "Anzahl Validierungsdaten = ": len(X_test_fasttext),
               "Anzahl meta Trainingsdaten inklusive Zusatzdaten = ": len(X_train_meta),
               "Anzahl meta Validierungsdaten = ": len(X_test_meta)
               })

    # hier muss ich sicherheitshalber alles nochmal abspeichern!!
    # für jeden durchgang jeweils meta model, ft model, combi model, meta_train, meta_val, ft_train, ft_val, combi_train, combi_val, und jeweils ergebnisse und report ergebnisse
    with open("./TestDataset/oesch8_nur/meta_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)
    with open("./TestDataset/oesch8_nur/ft_model.pkl", "wb") as f:
        pickle.dump(fasttext_model, f)
    with open("./TestDataset/oesch8_nur/combi_model.pkl", "wb") as f:
        pickle.dump(combi_model, f)

    with open("./TestDataset/oesch8_nur/meta_Xtrain.pkl", "wb") as f:
        pickle.dump(X_train_meta, f)
    with open("./TestDataset/oesch8_nur/ft_Xtrain.pkl", "wb") as f:
        pickle.dump(X_train_fasttext, f)
    with open("./TestDataset/oesch8_nur/combi_Xtrain.pkl", "wb") as f:
        pickle.dump(X_train_combi, f)

    with open("./TestDataset/oesch8_nur/meta_ytrain.pkl", "wb") as f:
        pickle.dump(y_train_meta, f)
    with open("./TestDataset/oesch8_nur/ft_ytrain.pkl", "wb") as f:
        pickle.dump(y_train_fasttext, f)
    with open("./TestDataset/oesch8_nur/combi_ytrain.pkl", "wb") as f:
        pickle.dump(y_train_combi, f)

    with open("./TestDataset/oesch8_nur/meta_Xtest.pkl", "wb") as f:
        pickle.dump(X_test_meta, f)
    with open("./TestDataset/oesch8_nur/ft_Xtest.pkl", "wb") as f:
        pickle.dump(X_test_fasttext, f)
    with open("./TestDataset/oesch8_nur/combi_Xtest.pkl", "wb") as f:
        pickle.dump(X_test_combi, f)

    with open("./TestDataset/oesch8_nur/meta_ytest.pkl", "wb") as f:
        pickle.dump(y_test_meta, f)
    with open("./TestDataset/oesch8_nur/ft_ytest.pkl", "wb") as f:
        pickle.dump(y_test_fasttext, f)
    with open("./TestDataset/oesch8_nur/combi_ytest.pkl", "wb") as f:
        pickle.dump(y_test_combi, f)

    json.dump(evaluation_meta, open("./TestDataset/oesch8_nur/meta_result.json", 'w'))
    json.dump(evaluation_fasttext, open("./TestDataset/oesch8_nur/ft_result.json", 'w'))
    json.dump(evaluation_combi, open("./TestDataset/oesch8_nur/combi_result.json", 'w'))
    json.dump(evaluation_combi_confidence, open("./TestDataset/oesch8_nur/combi_conf_result.json", 'w'))

    json.dump(evaluation_meta_train, open("./TestDataset/oesch8_nur/meta_report_train.json", 'w'))
    json.dump(evaluation_fasttext_train, open("./TestDataset/oesch8_nur/ft_report_train.json", 'w'))
    json.dump(evaluation_combi_train, open("./TestDataset/oesch8_nur/combi_report_train.json", 'w'))

    json.dump(evaluation_meta_val, open("./TestDataset/oesch8_nur/meta_report_test.json", 'w'))
    json.dump(evaluation_fasttext_val, open("./TestDataset/oesch8_nur/ft_report_test.json", 'w'))
    json.dump(evaluation_combi_val, open("./TestDataset/oesch8_nur/combi_report_test.json", 'w'))


    pprint(meta_ergebnisse)
    pprint(ft_ergebnisse)
    pprint(combi_ergebnisse)
    pprint(combi_conf_ergebnisse)
    run.finish()

if __name__ == '__main__':
    main()