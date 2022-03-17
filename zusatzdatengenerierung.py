import pandas as pd
import numpy as np
from pathlib import Path

def missing_val(x):
    if x in [".a", ".b", ".c", ".d"]:
        x = -1
    return int(x)

# oesch: kann "oesch8" oder "oesch16" sein, welche Daten als True Werte genommen werden
# selbstständige: kann "mit", "ohne", und "nur" sein, nimmt Datensatz inklusive Selbstständigen, Datensatz ohne
# Selbstständige und Datensatz nur mit Selbstständigen
class Zusatzdatengenerierer:

    @staticmethod
    def make_dataset(config):

        # Datensatz Generierung für Meta-Daten Modell ohne Selbstständige bestehend aus den bereits gelabelten Daten von Welle 2 und Welle 3
        Welle_2 = pd.read_csv(config["path_welle2"], sep=";")
        if config["selbstständige"] == "ohne":
            if config["oesch"] == "oesch16":
                Daten_2_clean = Welle_2[
                    Welle_2.oesch16.isin(["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])]
            elif config["oesch"] == "oesch8":
                Daten_2_clean = Welle_2[
                    Welle_2.oesch8.isin(["3", "4", "5", "6", "7", "8"])]

        elif config["selbstständige"] == "nur":
            if config["oesch"] == "oesch16":
                Daten_2_clean = Welle_2[
                    Welle_2.oesch16.isin(["1", "2", "3", "4"])]
            elif config["oesch"] == "oesch8":
                Daten_2_clean = Welle_2[
                    Welle_2.oesch8.isin(["1", "2"])]

        if config["selbstständige"] == "ohne":
            # ich verwende nicht die spalten erw_stat und selbst_gr da diese nur für selbstständige relevant sind
            X_meta_2 = Daten_2_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']]

        elif config["selbstständige"] == "nur":
            # hier verwende ich auch die Spalten erw_stat und selbst_gr
            X_meta_2 = Daten_2_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']]

        y_meta_2 = Daten_2_clean[config["oesch"]].astype(int)

        Welle_3 = pd.read_csv(config["path_welle3"], sep=";")

        if config["selbstständige"] == "ohne":
            if config["oesch"] == "oesch16":
                Daten_3_clean = Welle_3[
                    Welle_3.oesch16.isin(["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])]
            elif config["oesch"] == "oesch8":
                Daten_3_clean = Welle_3[
                    Welle_3.oesch8.isin(["3", "4", "5", "6", "7", "8"])]

        elif config["selbstständige"] == "nur":
            if config["oesch"] == "oesch16":
                Daten_3_clean = Welle_3[
                    Welle_3.oesch16.isin(["1", "2", "3", "4"])]
            elif config["oesch"] == "oesch8":
                Daten_3_clean = Welle_3[
                    Welle_3.oesch8.isin(["1", "2"])]

        if config["selbstständige"] == "ohne":
            # ich verwende nicht die spalten erw_stat und selbst_gr da diese nur für selbstständige relevant sind
            X_meta_3 = Daten_3_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']]
        elif config["selbstständige"] == "nur":
            # hier verwende ich auch erw_stat und selbst_gr
            X_meta_3 = Daten_3_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']]
        y_meta_3 = Daten_3_clean[config["oesch"]].astype(int)

        # ich caste alle Daten zu integern ersetze in den Daten die fehlenden Werte durch -1
        # außerdem caste ich Datensatz zu numpy array
        X_meta_2 = X_meta_2.applymap(lambda x: missing_val(x)).to_numpy()
        y_meta_2 = y_meta_2.to_numpy()
        X_meta_3 = X_meta_3.applymap(lambda x: missing_val(x)).to_numpy()
        y_meta_3 = y_meta_3.to_numpy()

        X_meta = np.concatenate((X_meta_2, X_meta_3), axis=0)
        y_meta = np.concatenate((y_meta_2, y_meta_3), axis=0)

        if config["selbstständige"] == "ohne":
            # Wörterbuch_de als Trainingsdatensatz für Fasttext, ohne Selbstständige

            Wörterbuch_de = pd.read_csv(config["path_wb"], sep=";")
            Wörterbuch_de = Wörterbuch_de[~Wörterbuch_de.oesch16.isin([-9])]

            # generiere Trainingsdatensatz mit taetigk string und oesch label
            fasttext_wb_df = Wörterbuch_de[["berufsbezeichnung", config["oesch"]]].copy()
            fasttext_wb_df = fasttext_wb_df.rename(columns={'berufsbezeichnung': 'taetigk'})

        else:
             fasttext_wb_df = None


        return fasttext_wb_df, X_meta, y_meta