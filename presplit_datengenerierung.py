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
class Datengenerierer:

    def __init__(self, ratio: float, augmentation: bool, oesch: str, selbstständige: str, random_state: int):
        self.ratio = ratio
        self.augmentation = augmentation
        self.oesch = oesch
        self.selbstständige = selbstständige
        self.random_state = random_state

        # Datensatzgenerierung aus Welle 1: 60% Trainingsdaten und 40% Validierungsdaten
        Welle_1 = pd.read_csv("./Daten/wic_beruf-w1_data.csv", sep=";")
        if self.selbstständige == "ohne":
            if self.oesch == "oesch16":
                Welle_1_clean = Welle_1[
                    Welle_1.oesch16.isin(["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])]
            elif self.oesch == "oesch8":
                Welle_1_clean = Welle_1[
                    Welle_1.oesch8.isin(["3", "4", "5", "6", "7", "8"])]
        elif self.selbstständige == "nur":
            if self.oesch == "oesch16":
                Welle_1_clean = Welle_1[
                    Welle_1.oesch16.isin(["1", "2", "3", "4"])]
            elif self.oesch == "oesch8":
                Welle_1_clean = Welle_1[
                    Welle_1.oesch8.isin(["1", "2"])]

        # ich caste alle Daten für das meta-Modell zu integern ersetze in den Daten die fehlenden Werte durch -1
        if self.selbstständige == "ohne":
            Welle_1_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']] = Welle_1_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].applymap(lambda x: missing_val(x))

        elif self.selbstständige == "nur":
            # nutze hier auch die spalten erw_stat und selbst_gr
            Welle_1_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']] = Welle_1_clean[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']].applymap(lambda x: missing_val(x))

        # teile datensatz in 60% trainings und 40% validierungsdatensatz
        # TODO diese aufteilung mache ich nicht mehr
        #self.w1_training = Welle_1_clean.sample(frac=self.ratio, random_state=self.random_state)
        #self.w1_validation = Welle_1_clean.drop(self.w1_training.index)

    # Hier erstelle ich den Grund-Datensatz für das Fasttext Modell, indem ich die embeddings und die Oesch Werte extrahiere
    def make_fasttext_dataset(self):

        ft = fasttext.load_model('cc.de.300.bin')

        # füge bei Fasttext Trainingsdatensatz embeddings hinzu
        w1_training_ft = deepcopy(self.w1_training)
        w1_training_ft["embeddings"] = w1_training_ft["taetigk"].apply(ft.get_word_vector)
        # shuffle trainingsdatensatz
        #w1_training_ft = w1_training_ft.sample(frac=1, random_state=self.random_state)
        # generiere X und y
        X_w1_ft = w1_training_ft["embeddings"].values
        y_w1_ft = w1_training_ft[self.oesch].astype(int)
        # mache matrix aus den Trainingsdaten
        X_w1_ft = np.vstack((X_w1_ft[i] for i in range(len(X_w1_ft))))


        # Todo: hier muss ich drauf achten, dass die daten das richtige Format haben zur weiteren Verarbeitung
        trainingsdaten = [X_w1_ft, y_w1_ft]
        return trainingsdaten