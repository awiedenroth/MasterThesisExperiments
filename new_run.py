from typing import Union, Dict, Any

from presplit_datengenerierung import Datengenerierer
from zusatzdatengenerierung import Zusatzdatengenerierer
from sklearn.model_selection import train_test_split
from data_cleaning import clean_data
from modelltraining import Modelltrainer
from evaluation import Evaluierer
from datenaugementierung import augment_data
from finalize_dataset import finalize_data
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

if not sys.warnoptions:
    warnings.simplefilter("ignore")

configuration = {
    "fasttext_zusatzdaten": False,
    "meta_zusatzdaten" : False,
    "selbstständige" : "nur",
    "oesch" : "oesch8",
    "lowercase" : True,
    "remove_stopwords": True,
    "remove_numbers": False,
    "remove_punctuation": False,
    "keyboard_aug" : True,
    "random_seed": 42
}

wandb.init(project="Masterarbeit", entity="awiedenroth", config=configuration)

# caching funktion zur Datensatzerstellung
@mem.cache
def instantiate_dataset(configuration: Dict[str, Union[bool,str]]) -> Any:
    datengenerierer = Datengenerierer(configuration["oesch"], configuration["selbstständige"])
    zusatzdatengenerierer = Zusatzdatengenerierer(configuration["oesch"], configuration["selbstständige"])
    # ich erzeuge für fasttext und meta jeweils die grunddaten
    fasttext_df, X_meta, y_meta = datengenerierer.make_dataset()
    # ich erzeuge die Zusatzdaten für fasttext und meta
    fasttext_wb_df, X_meta_z, y_meta_z = zusatzdatengenerierer.make_dataset()

    return fasttext_df, X_meta, y_meta, fasttext_wb_df, X_meta_z, y_meta_z

if __name__ == "__main__":
    # hier muss ich angeben ob ich Oesch8 oder Oesch16 möchte und ob ich "nur" Selbstständige oder "ohne" Selbstständige haben möchte
    #Todo: hier könnte ich statt strings für oesch und selbstständige etwas eleganter Oesch=8 bzw Oesch=16 und Selbstständige = true/false machen
    """datengenerierer = Datengenerierer(configuration["oesch"],configuration["selbstständige"])
    zusatzdatengenerierer = Zusatzdatengenerierer(configuration["oesch"],configuration["selbstständige"])
    # ich erzeuge für fasttext und meta jeweils die grunddaten
    fasttext_df, X_meta, y_meta = datengenerierer.make_dataset()
    # ich erzeuge die Zusatzdaten für fasttext und meta
    fasttext_wb_df, X_meta_z, y_meta_z = zusatzdatengenerierer.make_dataset()"""
    fasttext_df, X_meta, y_meta, fasttext_wb_df, X_meta_z, y_meta_z = instantiate_dataset(configuration)

    fasttext_df = clean_data(fasttext_df, configuration)
    if configuration["fasttext_zusatzdaten"] == True:
        fasttext_wb_df = clean_data(fasttext_wb_df, configuration)
    #dict zum abspeichern der ergebnisse
    ergebnisse = {}
    # index um zu tracken bei welchem durchgang man ist
    i = 0
    kf = KFold(n_splits=8)
    # die for schleife geht k mal durch
    for train_index, test_index in kf.split(fasttext_df):
        print("Durchgang ", i)

        # erstelle die fasttext trainings und test daten
        X_train_fasttext, X_test_fasttext = fasttext_df.iloc[train_index], fasttext_df.iloc[test_index]
        #y_train_fasttext, y_test_fasttext = y_fasttext.iloc[train_index], y_fasttext.iloc[test_index]
        # erstelle die meta modell trainings und test daten
        X_train_meta, X_test_meta = X_meta[train_index], X_meta[test_index]
        y_train_meta, y_test_meta = y_meta[train_index], y_meta[test_index]

        print("Anteil Trainingsdaten = ", len(X_train_fasttext), "von", len(fasttext_df))

        # füge zu den trainingsdatensätzen die zusatzdaten hinzu falls gewünscht
        if configuration["fasttext_zusatzdaten"] == True:
            if configuration["selbstständige"] == "ohne":
                #X_train_fasttext = np.concatenate((X_train_fasttext, X_fasttext_z))
                #y_train_fasttext = np.concatenate((y_train_fasttext, y_fasttext_z))
                X_train_fasttext = pd.concat([X_train_fasttext, fasttext_wb_df])

        if configuration["meta_zusatzdaten"] == True:
            X_train_meta = np.concatenate((X_train_meta, X_meta_z))
            y_train_meta = np.concatenate((y_train_meta, y_meta_z))

        print("Anzahl fasttext Trainingsdaten inklusive Zusatzdaten = ", len(X_train_fasttext))
        print("Anzahl fasttext Validierungsdaten = ", len(X_test_fasttext))
        print("Anzahl meta Trainingsdaten inklusive Zusatzdaten = ", len(X_train_meta))
        print("Anzahl meta Validierungsdaten = ", len(X_test_meta))

        X_train_fasttext = augment_data(X_train_fasttext, configuration)

        # ich erzeuge aus training_df und test_df die embeddings bei den fasttext dingen und shuffle
        X_train_fasttext, y_train_fasttext = finalize_data(X_train_fasttext, configuration)
        X_test_fasttext, y_test_fasttext = finalize_data(X_test_fasttext, configuration)



        # hier füge ich die anderen Metriken hinzu
        modelltrainer = Modelltrainer(0, "merror")
        evaluierer = Evaluierer()

        print("trainiere meta Modell")
        # Todo: was zur fucking hölle ist hier los?!? es sagt immer "Wrong number of arguments for train_meta(self, X, y, hyperparameter):"
        meta_model = modelltrainer.train_meta(X_train_meta, y_train_meta, hyperparameter=None)
        evaluation_meta = evaluierer.make_evaluation(meta_model, X_train_meta, y_train_meta,
                                                         X_test_meta, y_test_meta)
        print("Meta Modell: ")
        pprint(evaluation_meta)
        #with open("Trained_Models/meta_8_80_n_0.pkl", "wb") as f:
        #    pickle.dump(meta_model, f)
        #json.dump(evaluation_meta, open("Ergebnisse/meta_8_80_o_0.json", 'w'))


        print("trainiere fasttext Modell")
        fasttext_model = modelltrainer.train_ft(X_train_fasttext, y_train_fasttext, hyperparameter=None)
        evaluation_fasttext = evaluierer.make_evaluation(fasttext_model, X_train_fasttext, y_train_fasttext,
                                   X_test_fasttext, y_test_fasttext)
        pprint("Fasttext Modell: ")
        pprint(evaluation_fasttext)
        #with open("Trained_Models/fasttext_8_80_n_0.pkl", "wb") as f:
            #pickle.dump(fasttext_model, f)
        #json.dump(evaluation_fasttext, open("Ergebnisse/fasttext_8_80_n_0.json", 'w'))

        #Todo: combimodell zum laufen bringen
        # erzeuge Daten für Combi model
        fasttext_proba = fasttext_model.predict_proba(X_train_fasttext)
        meta_proba = meta_model.predict_proba(X_train_meta)
        X_train_combi = np.concatenate((fasttext_proba, meta_proba), axis=1)
        y_train_combi = (y_meta[train_index])


        # erzeuge validierungsdaten für combi model
        fasttext_proba_test = fasttext_model.predict_proba(X_test_fasttext)
        meta_proba_test = meta_model.predict_proba(X_test_meta)
        X_test_combi = np.concatenate((fasttext_proba_test, meta_proba_test), axis=1)
        y_test_combi = (y_meta[test_index])

        print("trainiere Combi Modell")

        combi_model = modelltrainer.train_combi(X_train_combi, y_train_combi, hyperparameter=None)
        evaluation_combi = evaluierer.make_evaluation(combi_model, X_train_combi, y_train_combi,
                                                         X_test_combi, y_test_combi)
        print("Combi Modell evaluation: ", evaluation_combi)
        #with open("Trained_Models/combi_8_80_n_0.pkl", "wb") as f:
           # pickle.dump(combi_model, f)
        #json.dump(evaluation_combi, open("Ergebnisse/combi_8_80_n_0.json", 'w'))


        """for confidence in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.992,0.994,0.995, 0.996, 0.997]:
            path = "Ergebnisse/combi_8_80_n_0_"
            evaluation_combi_confidence = evaluierer.make_evaluation_confidence(combi_model, train[0], train[1],
                                                          val[0], val[1], confidence)
            print("Combi Modell evaluation mit confidence", confidence, evaluation_combi_confidence)
            path += str(confidence) + ".json"
            json.dump(evaluation_combi_confidence, open(path, 'w'))"""
        ergebnisse[i] = {"meta":evaluation_meta, "fasttext": evaluation_fasttext, "combi": evaluation_combi}
        wandb.log({"cross validation Durchgang": i, "meta":evaluation_meta, "fasttext": evaluation_fasttext, "combi": evaluation_combi})
        i = i+1



    json.dump(ergebnisse, open("Ergebnisse/kfold_8_nur", 'w'))
    pprint(ergebnisse)