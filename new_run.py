from typing import Union, Dict, Any
from postsplit_datengenerierung import PS_Datengenerierer
from presplit_datengenerierung import Datengenerierer
from zusatzdatengenerierung import Zusatzdatengenerierer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_cleaning import clean_data
from modelltraining import train_ft
from modelltraining import train_meta
from modelltraining import train_combi
from evaluation import Evaluierer
from datenaugementierung import augment_data
from sklearn.model_selection import KFold
import pickle
import json
import sys
import warnings
from pprint import pprint
import numpy as np
import wandb
import pandas as pd
from caching import mem
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

if not sys.warnoptions:
    warnings.simplefilter("ignore")

configuration = {
    "fasttext_zusatzdaten": False,
    "meta_zusatzdaten" : True,
    "selbstständige" : "nur",
    "oesch" : "oesch8",
    "lowercase" : True,
    "remove_stopwords": True,
    "remove_numbers": True,
    "remove_punctuation": True,
    "remove_duplicates": True,
    "keyboard_aug" : True,
    "random_seed": 42,
    "path_welle1": "./Daten/wic_beruf-w1_data.csv",
    "path_welle2": "./Daten/wic_beruf-w2_data.csv",
    "path_welle3": "./Daten/wic_beruf-w3_data.csv",
    "path_wb": "./Wörterbücher/wic_wörterbuch_aufbereitet_oesch.csv",
    "path_pretrained_fasttext_model": "cc.de.300.bin",
    "k_fold_splits": 4,
    "ft_model": "nn",
    "meta_model": "nn",
    "combi_model": "nn" # "xgboost" oder "nn" oder "linear"
}

run = wandb.init(project="Masterarbeit", entity="awiedenroth", config=configuration)

# caching funktion zur Datensatzerstellung
@mem.cache
def instantiate_dataset(configuration: Dict[str, Union[bool,str]]) -> Any:
    # ich erzeuge für fasttext und meta jeweils die grunddaten
    fasttext_df, X_meta, y_meta = Datengenerierer.make_dataset(configuration)
    # ich erzeuge die Zusatzdaten für fasttext und meta
    fasttext_wb_df, X_meta_z, y_meta_z = Zusatzdatengenerierer.make_dataset(configuration)

    return fasttext_df, X_meta, y_meta, fasttext_wb_df, X_meta_z, y_meta_z

if __name__ == "__main__":

    fasttext_df, X_meta, y_meta, fasttext_wb_df, X_meta_z, y_meta_z = instantiate_dataset(configuration)

    # Todo: aufzeichnen wieviel prozent der Daten durch cleaning rausgechmissen werden, jeweils für wörterbuch und welle 1 daten
    fasttext_df = clean_data(fasttext_df, configuration)
    if configuration["fasttext_zusatzdaten"] == True:
        fasttext_wb_df = clean_data(fasttext_wb_df, configuration)
    #dict zum abspeichern der ergebnisse
    ergebnisse = {}
    # index um zu tracken bei welchem durchgang man ist
    i = 0
    kf = KFold(n_splits=configuration["k_fold_splits"], shuffle = True, random_state = configuration["random_seed"])
    # die for schleife geht k mal durch
    for train_index, test_index in kf.split(fasttext_df):
        print("Durchgang ", i)

        X_train_fasttext, y_train_fasttext, X_test_fasttext, y_test_fasttext = \
            PS_Datengenerierer.make_ps_data_ft(fasttext_df, fasttext_wb_df, configuration, train_index, test_index)

        X_train_meta, y_train_meta, X_test_meta, y_test_meta = \
            PS_Datengenerierer.make_ps_data_meta(X_meta, y_meta, X_meta_z, y_meta_z, configuration, train_index, test_index)

        print("Anteil Trainingsdaten = ", len(X_train_fasttext), "von", len(fasttext_df))
        print("Anzahl fasttext Trainingsdaten inklusive Zusatzdaten = ", len(X_train_fasttext))
        print("Anzahl fasttext Validierungsdaten = ", len(X_test_fasttext))
        print("Anzahl meta Trainingsdaten inklusive Zusatzdaten = ", len(X_train_meta))
        print("Anzahl meta Validierungsdaten = ", len(X_test_meta))

        # Todo: hier muss ich den evaluierer überarbeiten und so machen dass es gleich die richtigen plots gibt!

        print("trainiere meta Modell")

        meta_model = train_meta(X_train_meta, y_train_meta, configuration)
        evaluation_meta = Evaluierer.make_evaluation(meta_model, X_train_meta, y_train_meta,
                                                         X_test_meta, y_test_meta, modelname = "meta")
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(meta_model, X_test_meta, y_test_meta, normalize=None, ax=plt.gca())
        plt.tight_layout()
        wandb.log({f"meta model run{i}": plt})

        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(meta_model, X_test_meta, y_test_meta, normalize="true", ax=plt.gca())
        plt.tight_layout()
        wandb.log({f"meta model normalized run{i}": plt})

        print("Meta Modell: ")
        pprint(evaluation_meta)
        #with open("Trained_Models/meta_8_80_n_0.pkl", "wb") as f:
        #    pickle.dump(meta_model, f)
        #json.dump(evaluation_meta, open("Ergebnisse/meta_8_80_o_0.json", 'w'))


        print("trainiere fasttext Modell")
        fasttext_model = train_ft(X_train_fasttext, y_train_fasttext, configuration)
        evaluation_fasttext = Evaluierer.make_evaluation(fasttext_model, X_train_fasttext, y_train_fasttext,
                                   X_test_fasttext, y_test_fasttext, modelname = "fasttext")
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(fasttext_model, X_test_fasttext, y_test_fasttext, normalize=None, ax=plt.gca())
        plt.tight_layout()
        wandb.log({f"fasttext model run{i}": plt})
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(fasttext_model, X_test_fasttext, y_test_fasttext, normalize="true", ax=plt.gca())
        plt.tight_layout()
        wandb.log({f"fasttext model normalized run{i}": plt})
        pprint("Fasttext Modell: ")
        pprint(evaluation_fasttext)

        X_train_combi, y_train_combi, X_test_combi, y_test_combi = \
            PS_Datengenerierer.make_ps_data_combi(fasttext_df, X_meta, fasttext_model, meta_model, train_index, test_index, configuration)

        print("trainiere Combi Modell")

        combi_model = train_combi(X_train_combi, y_train_combi, configuration)
        evaluation_combi = Evaluierer.make_evaluation(combi_model, X_train_combi, y_train_combi,
                                                         X_test_combi, y_test_combi, modelname = "combi")
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(combi_model, X_test_combi, y_test_combi, normalize=None, ax=plt.gca())
        plt.tight_layout()
        wandb.log({f"combi model run{i}": plt})
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(combi_model, X_test_combi, y_test_combi, normalize="true", ax=plt.gca())
        plt.tight_layout()
        wandb.log({f"combi model normalized run{i}": plt})

        pprint("Combi Modell evaluation: ")
        pprint(evaluation_combi)

        for confidence in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9,0.91,0.92,0.93]:#,0.94,0.95,0.96,0.97,0.98,0.99,0.992,0.994,0.995, 0.996]:
            path = "Ergebnisse/combi_8_80_n_0_"
            evaluation_combi_confidence = Evaluierer.make_evaluation_confidence(combi_model, X_train_combi, y_train_combi, X_test_combi, y_test_combi, confidence)
            pprint("Combi Modell evaluation mit confidence")
            pprint((confidence, evaluation_combi_confidence))
            path += str(confidence) + ".json"
            json.dump(evaluation_combi_confidence, open(path, 'w'))
        ergebnisse[i] = {"meta":evaluation_meta, "fasttext": evaluation_fasttext, "combi": evaluation_combi}
        wandb.log({"cross validation Durchgang": i, "meta":evaluation_meta, "fasttext": evaluation_fasttext, "combi": evaluation_combi})
        i = i+1

    json.dump(ergebnisse, open("Ergebnisse/kfold_8_ohne_ohne_zusatz", 'w'))
    run.finish()
    pprint(ergebnisse)