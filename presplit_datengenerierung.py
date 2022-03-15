import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
import fastText as fasttext
import fasttext.util

#ft = fasttext.load_model('cc.de.300.bin')


def missing_val(x):
    if x in [".a", ".b", ".c", ".d"]:
        x = -1
    return int(x)


# ratio ist prozentsatz von trainingsdaten, also 0.2 bedeutet 20% trainings daten, 80% testdaten
# augmentation: wenn True, werden Daten aus dem Wörterbuch durch groß/kleinschreibung verdoppelt
# oesch: kann "oesch8" oder "oesch16" sein, welche Daten als True Werte genommen werden
# selbstständige: kann "mit", "ohne", und "nur" sein, nimmt Datensatz inklusive Selbstständigen, Datensatz ohne
# Selbstständige und Datensatz nur mit Selbstständigen
class Datengenerierer:

    @staticmethod
    def _read_in_correct_data(config: dict, path: str = "./Daten/wic_beruf-w1_data.csv") -> pd.DataFrame:
        Welle_1 = pd.read_csv(path, sep=";")
        if config["selbstständige"] == "ohne":
            if config["oesch"] == "oesch16":
                Welle_1_clean = Welle_1[
                    Welle_1.oesch16.isin(["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])]
            elif config["oesch"] == "oesch8":
                Welle_1_clean = Welle_1[
                    Welle_1.oesch8.isin(["3", "4", "5", "6", "7", "8"])]
        elif config["selbstständige"] == "nur":
            if config["oesch"] == "oesch16":
                Welle_1_clean = Welle_1[
                    Welle_1.oesch16.isin(["1", "2", "3", "4"])]
            elif config["oesch"] == "oesch8":
                Welle_1_clean = Welle_1[
                    Welle_1.oesch8.isin(["1", "2"])]
        return Welle_1_clean

    @staticmethod
    def _cast_to_int_and_replace_missing(config: dict, df: pd.DataFrame) -> pd.DataFrame:
        # ich caste alle Daten für das meta-Modell zu integern ersetze in den Daten die fehlenden Werte durch -1
        if config["selbstständige"] == "ohne":
            df[['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']] = df[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].applymap(lambda x: missing_val(x))

        elif config["selbstständige"] == "nur":
            # nutze hier auch die spalten erw_stat und selbst_gr
            df[['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']] = df[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']].applymap(lambda x: missing_val(x))
        return df

    def __init__(self, config):

        self.config = config

        Welle_1_clean = self._read_in_correct_data(self.config)
        self.w1_training = self._cast_to_int_and_replace_missing(self.config, Welle_1_clean)


    # Hier erstelle ich den Grund-Datensatz für das Fasttext Modell, indem ich die embeddings und die Oesch Werte extrahiere
    def make_dataset(self):

        #ft = fasttext.load_model('cc.de.300.bin')

        # füge bei Fasttext Trainingsdatensatz embeddings hinzu
        w1_training_ft = deepcopy(self.w1_training)
        #w1_training_ft["embeddings"] = w1_training_ft["taetigk"].apply(ft.get_word_vector)
        # shuffle trainingsdatensatz
        #w1_training_ft = w1_training_ft.sample(frac=1, random_state=self.random_state)
        # generiere X und y
        # todo: hier sollte ich einfach w1_training_ft komplett zurück geben, bzw die Spalten taetigk, embeddings, oesch
        trainingsdaten_ft = w1_training_ft[["taetigk", self.oesch]].copy()
        #X_w1_ft = w1_training_ft["embeddings"].values
        #y_w1_ft = w1_training_ft[self.oesch].astype(int)
        # mache matrix aus den Trainingsdaten
        #X_w1_ft = np.vstack((X_w1_ft[i] for i in range(len(X_w1_ft))))

        if self.selbstständige == "ohne":
            # erstelle trainingsdatemsatz für meta Modell als np array
            X_w1_meta = self.w1_training[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].to_numpy()

        elif self.selbstständige == "nur":
            X_w1_meta = self.w1_training[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']].to_numpy()
        y_w1_meta = self.w1_training[self.oesch].astype(int).to_numpy()

        # ich gebe zurück: fasttext trainingsdaten mit "taetigk", "embeddings", self.oesch, meta daten, meta_labels
        return  trainingsdaten_ft, X_w1_meta, y_w1_meta
