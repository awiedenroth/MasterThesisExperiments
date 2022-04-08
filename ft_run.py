from copy import deepcopy
from typing import Union, Dict, Any
import time

from networkx.drawing.tests.test_pylab import plt
from tqdm import tqdm
from postsplit_datengenerierung import PS_Datengenerierer
from presplit_datengenerierung import Datengenerierer
from zusatzdatengenerierung import Zusatzdatengenerierer
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
from caching import mem
from average_calculater import calculate_average
from average_calculater import calculate_conf_average
from average_calculater import calculate_average_report

if not sys.warnoptions:
    warnings.simplefilter("ignore")

DEFAULT_CONFIG = {
    "fasttext_zusatzdaten": True,
    "meta_zusatzdaten" : False,
    "selbstständige" : "ohne",
    "oesch" : "oesch8",
    "lowercase" : False,
    "remove_stopwords": False,
    "remove_num_punc": False,
    "keyboard_aug" : True,
    "random_seed": 22,
    "path_welle1": "./Daten/welle1_und_3.csv",
    "path_welle2": "./Daten/wic_beruf-w2_data.csv",
    "path_welle3": "./Daten/wic_beruf-w4_data.csv",
    "path_wb": "./Wörterbücher/Wörterbuch_binär.csv",
    "path_pretrained_fasttext_model": "cc.de.300.bin",
    "k_fold_splits": 10,
    "ft_model": "nn",
    "meta_model": "xgboost",
    "combi_model": "nn" # "xgboost" oder "nn" oder "linear"
}

# caching funktion zur Datensatzerstellung
#@mem.cache
def instantiate_dataset(configuration: Dict[str, Union[bool,str]]) -> Any:
    # ich erzeuge für fasttext und meta jeweils die grunddaten
    fasttext_df, X_meta, y_meta = Datengenerierer.make_dataset(configuration)
    # ich erzeuge die Zusatzdaten für fasttext und meta
    fasttext_wb_df, X_meta_z, y_meta_z = Zusatzdatengenerierer.make_dataset(configuration)

    return fasttext_df, X_meta, y_meta, fasttext_wb_df, X_meta_z, y_meta_z

def main():
    run = wandb.init(project="ft_oesch8_nur", entity="awiedenroth")
    print("wandb says: ", wandb.config)
    #configuration = DEFAULT_CONFIG
    configuration = {k: v for k, v in wandb.config.items()}
    print("actual conf:", configuration)

    fasttext_df, X_meta, y_meta, fasttext_wb_df, X_meta_z, y_meta_z = instantiate_dataset(configuration)

    fasttext_df = clean_data(fasttext_df, configuration)
    if configuration["fasttext_zusatzdaten"] == True and configuration["selbstständige"] == "ohne":
        fasttext_wb_df = clean_data(fasttext_wb_df, configuration)
    #dict zum abspeichern der ergebnisse
    ergebnisse = []
    #meta_ergebnisse = []
    ft_ergebnisse = []
    #combi_ergebnisse = []
    #meta_ergebnisse_train = []
    ft_ergebnisse_train = []
    #combi_ergebnisse_train = []
    #meta_ergebnisse_val = []
    ft_ergebnisse_val = []
    #combi_ergebnisse_val = []
    #combi_conf_ergebnisse = []
    # index um zu tracken bei welchem durchgang man ist
    i = 0
    kf = KFold(n_splits=configuration["k_fold_splits"], shuffle = True, random_state = configuration["random_seed"])

    # die for schleife geht k mal durch
    for train_index, test_index in kf.split(fasttext_df):
        start_time = time.time()
        print("Durchgang ", i)

        X_train_fasttext, y_train_fasttext, X_test_fasttext, y_test_fasttext = \
            PS_Datengenerierer.make_ps_data_ft(fasttext_df, fasttext_wb_df, configuration, train_index, test_index)

        #X_train_meta, y_train_meta, X_test_meta, y_test_meta = \
         #   PS_Datengenerierer.make_ps_data_meta(X_meta, y_meta, X_meta_z, y_meta_z, configuration, train_index, test_index)

        print("Gesamtanzahl Datenpunkte Grunddaten vor k-split = ", len(fasttext_df))
        print("Anzahl Trainingsdaten ohne Zusatzdaten", len(fasttext_df)-len(X_test_fasttext))
        print("Anzahl Fasttext Trainingsdaten inklusive Zusatzdaten", len(X_train_fasttext))
        print("Anzahl fasttext Validierungsdaten = ", len(X_test_fasttext))
        #print("Anzahl meta Trainingsdaten inklusive Zusatzdaten = ", len(X_train_meta))
        #print("Anzahl meta Validierungsdaten = ", len(X_test_meta))

        #print("trainiere meta Modell")

        #meta_model = train_meta(X_train_meta, y_train_meta, configuration)
        #evaluation_meta, evaluation_meta_train, evaluation_meta_val = Evaluierer.make_evaluation(meta_model, X_train_meta, y_train_meta,
        #                                                 X_test_meta, y_test_meta, configuration, modelname = "meta", run = i)
        #meta_ergebnisse.append(evaluation_meta)
        #meta_ergebnisse_train.append(evaluation_meta_train)
        #meta_ergebnisse_val.append(evaluation_meta_val)

        print("trainiere fasttext Modell")
        fasttext_model = train_ft(X_train_fasttext, y_train_fasttext, configuration)
        evaluation_fasttext, evaluation_fasttext_train, evaluation_fasttext_val = Evaluierer.make_evaluation(fasttext_model, X_train_fasttext, y_train_fasttext,
                                   X_test_fasttext, y_test_fasttext, configuration, modelname = "fasttext", run = i)
        ft_ergebnisse.append(evaluation_fasttext)
        ft_ergebnisse_train.append(evaluation_fasttext_train)
        ft_ergebnisse_val.append(evaluation_fasttext_val)

        #X_train_combi, y_train_combi, X_test_combi, y_test_combi = \
          #  PS_Datengenerierer.make_ps_data_combi(fasttext_df, X_meta, fasttext_model, meta_model, train_index, test_index, configuration)

        #print("trainiere Combi Modell")

        #combi_model = train_combi(X_train_combi, y_train_combi, configuration)
        #evaluation_combi, evaluation_combi_train, evaluation_combi_val = Evaluierer.make_evaluation(combi_model, X_train_combi, y_train_combi,
        #                                                X_test_combi, y_test_combi, configuration, modelname = "combi", run = i)
        #combi_ergebnisse.append(evaluation_combi)
        #combi_ergebnisse_train.append(evaluation_combi_train)
        #combi_ergebnisse_val.append(evaluation_combi_val)

        #evaluation_combi_confidence = []
        #for confidence in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.992,0.994,0.995, 0.996, 0.997, 0.998]:
        #    evaluation_combi_confidence.append(Evaluierer.make_evaluation_confidence(combi_model, X_train_combi, y_train_combi, X_test_combi, y_test_combi, confidence, run=i))
        #combi_conf_ergebnisse.append(evaluation_combi_confidence)

        zeit = round(time.time() - start_time) / 60
        wandb.log({f"Dauer Durchlauf {i}min: ": zeit})
        i = i+1

    # hier berechne ich die averages von den einzelnen modell-evaluationen und speicher diese weg, diese sind das wichtigste für meine Auswertung
    #meta_average = calculate_average(meta_ergebnisse)
    ft_average = calculate_average(ft_ergebnisse)
    #combi_average = calculate_average(combi_ergebnisse)

    #meta_train_average = calculate_average_report(meta_ergebnisse_train)
    ft_train_average = calculate_average_report(ft_ergebnisse_train)
    #combi_train_average = calculate_average_report(combi_ergebnisse_train)

    #meta_val_average = calculate_average_report(meta_ergebnisse_val)
    ft_val_average = calculate_average_report(ft_ergebnisse_val)
    #combi_val_average = calculate_average_report(combi_ergebnisse_val)

    #combi_conf_average = calculate_conf_average(combi_conf_ergebnisse)
    #wandb.log({"meta model average performance": meta_average})
    wandb.log({"average": ft_average})
    #wandb.log({"combi model average performance": combi_average})

    #wandb.log({"meta model train average performance": meta_train_average})
    wandb.log({"train report average": ft_train_average})
    #wandb.log({"combi model train average performance": combi_train_average})

    #wandb.log({"meta model validation average performance": meta_val_average})
    wandb.log({"val report average": ft_val_average})
    #wandb.log({"combi model validation average performance": combi_val_average})

   # wandb.log({"combi confidence model average performance": combi_conf_average})
    wandb.log({"Anzahl Datenpunkte Grunddaten vor k-split = ": len(fasttext_df),
               "Anzahl Trainingsdaten ohne Zusatzdaten": len(fasttext_df) - len(X_test_fasttext),
               "Anzahl Trainingsdaten inklusive Zusatzdaten": len(X_train_fasttext),
               "Anzahl Validierungsdaten = ": len(X_test_fasttext),
               #"Anzahl meta Trainingsdaten inklusive Zusatzdaten = ": len(X_train_meta),
               #"Anzahl meta Validierungsdaten = ": len(X_test_meta)
               })

    #pprint(meta_average)
    pprint(ft_average)
    #pprint(combi_average)
    #pprint(combi_conf_average)
    plt.clf()
    plt.cla()
    run.finish()

if __name__ == '__main__':
    main()