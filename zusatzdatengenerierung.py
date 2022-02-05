import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
import fastText as fasttext
import fasttext.util

ft = fasttext.load_model('cc.de.300.bin')


def missing_val(x):
    if x in [".a", ".b", ".c", ".d"]:
        x = -1
    return int(x)


# ratio ist prozentsatz von trainingsdaten, also 0.2 bedeutet 20% trainings daten, 80% testdaten
# augmentation: wenn True, werden Daten aus dem Wörterbuch durch groß/kleinschreibung verdoppelt
# oesch: kann "oesch8" oder "oesch16" sein, welche Daten als True Werte genommen werden
# selbstständige: kann "mit", "ohne", und "nur" sein, nimmt Datensatz inklusive Selbstständigen, Datensatz ohne
# Selbstständige und Datensatz nur mit Selbstständigen
class Zusatzdatengenerierer:

    def __init__(self, oesch: str, selbstständige: str):
        self.oesch = oesch
        self.selbstständige = selbstständige


    def make_dataset(self):

        # Datensatz Generierung für Meta-Daten Modell ohne Selbstständige bestehend aus den bereits gelabelten Daten von Welle 2 und Welle 3
        Welle_2 = pd.read_csv("./Daten/wic_beruf-w2_data.csv", sep=";")
        if self.selbstständige == "ohne":
            if self.oesch == "oesch16":
                Daten_2_clean = Welle_2[
                    Welle_2.oesch16.isin(["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])]
            elif self.oesch == "oesch8":
                Daten_2_clean = Welle_2[
                    Welle_2.oesch8.isin(["3", "4", "5", "6", "7", "8"])]

        elif self.selbstständige == "nur":
            if self.oesch == "oesch16":
                Daten_2_clean = Welle_2[
                    Welle_2.oesch16.isin(["1", "2", "3", "4"])]
            elif self.oesch == "oesch8":
                Daten_2_clean = Welle_2[
                    Welle_2.oesch8.isin(["1", "2"])]

        if self.selbstständige == "ohne":
            # ich verwende nicht die spalten erw_stat und selbst_gr da diese nur für selbstständige relevant sind
            X_meta_2 = Daten_2_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']]

        elif self.selbstständige == "nur":
            # hier verwende ich auch die Spalten erw_stat und selbst_gr
            X_meta_2 = Daten_2_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']]

        y_meta_2 = Daten_2_clean[self.oesch].astype(int)

        Welle_3 = pd.read_csv("./Daten/wic_beruf-w3_data.csv", sep=";")

        if self.selbstständige == "ohne":
            if self.oesch == "oesch16":
                Daten_3_clean = Welle_3[
                    Welle_3.oesch16.isin(["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])]
            elif self.oesch == "oesch8":
                Daten_3_clean = Welle_3[
                    Welle_3.oesch8.isin(["3", "4", "5", "6", "7", "8"])]

        elif self.selbstständige == "nur":
            if self.oesch == "oesch16":
                Daten_3_clean = Welle_3[
                    Welle_3.oesch16.isin(["1", "2", "3", "4"])]
            elif self.oesch == "oesch8":
                Daten_3_clean = Welle_3[
                    Welle_3.oesch8.isin(["1", "2"])]

        if self.selbstständige == "ohne":
            # ich verwende nicht die spalten erw_stat und selbst_gr da diese nur für selbstständige relevant sind
            X_meta_3 = Daten_3_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']]
        elif self.selbstständige == "nur":
            # hier verwende ich auch erw_stat und selbst_gr
            X_meta_3 = Daten_3_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']]
        y_meta_3 = Daten_3_clean[self.oesch].astype(int)

        # ich caste alle Daten zu integern ersetze in den Daten die fehlenden Werte durch -1
        # außerdem caste ich Datensatz zu numpy array
        X_meta_2 = X_meta_2.applymap(lambda x: missing_val(x)).to_numpy()
        y_meta_2 = y_meta_2.to_numpy()
        X_meta_3 = X_meta_3.applymap(lambda x: missing_val(x)).to_numpy()
        y_meta_3 = y_meta_3.to_numpy()

        X_meta = np.concatenate((X_meta_2, X_meta_3), axis=0)
        y_meta = np.concatenate((y_meta_2, y_meta_3), axis=0)


        if self.selbstständige == "ohne":
            ft = fasttext.load_model('cc.de.300.bin')
            # Wörterbuch_de als Trainingsdatensatz für Fasttext, ohne Selbstständige

            path = Path("./Wörterbücher/wic_wörterbuch_aufbereitet_oesch.csv")
            Wörterbuch_de = pd.read_csv(path, sep=";")
            Wörterbuch_de = Wörterbuch_de[~Wörterbuch_de.oesch16.isin([-9])]

            # hier könnte man data-augmentation machen indem man lowercase hinzufügt

            # füge Spalte mit Embeddings hinzu
            Wörterbuch_de["embeddings"] = Wörterbuch_de["berufsbezeichnung"].apply(ft.get_word_vector)

            # generiere Trainingsdatensatz mit embeddings und oesch16

            fasttext_wb_df = Wörterbuch_de[["berufsbezeichnung", "embeddings", self.oesch]].copy()
            fasttext_wb_df = fasttext_wb_df.rename(columns={'berufsbezeichung': 'taetigk'})

            #X_wb = Wörterbuch_de["embeddings"].values
            #y_wb = Wörterbuch_de[self.oesch].values.astype(int)

            #X_wb = np.vstack((X_wb[i] for i in range(len(X_wb))))

            # Datensatz X_wb, y_wb besteht aus einer Matrix X_wb mit der Anzahl der Wörterbucheinträge als zeilen, und 300 spalten
            # die zeilen stehen für Datenpunkte, die spalten enthalten den 300 dimensionalen Fasttext embedding vektor
        else:
             fasttext_wb_df = None


        return fasttext_wb_df, X_meta, y_meta


