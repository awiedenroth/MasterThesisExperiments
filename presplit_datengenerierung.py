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

    def __init__(self, oesch: str, selbstständige: str):

        self.oesch = oesch
        self.selbstständige = selbstständige


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
        self.w1_training = Welle_1_clean
        #self.w1_validation = Welle_1_clean.drop(self.w1_training.index)

    # Hier erstelle ich den Grund-Datensatz für das Fasttext Modell, indem ich die embeddings und die Oesch Werte extrahiere
    def make_dataset(self):

        ft = fasttext.load_model('cc.de.300.bin')

        # füge bei Fasttext Trainingsdatensatz embeddings hinzu
        w1_training_ft = deepcopy(self.w1_training)
        w1_training_ft["embeddings"] = w1_training_ft["taetigk"].apply(ft.get_word_vector)
        # shuffle trainingsdatensatz
        #w1_training_ft = w1_training_ft.sample(frac=1, random_state=self.random_state)
        # generiere X und y
        # todo: hier sollte ich einfach w1_training_ft komplett zurück geben, bzw die Spalten taetigk, embeddings, oesch
        trainingsdaten_ft = w1_training_ft[["taetigk","embeddings", self.oesch]].copy()
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

# Todo: das sollte vermutlich eine eigene Klasse sein
    def make_combi_dataset(self, ft_model, meta_model):

        # generiere Fasttext daten
        X_ft = self.w1_training["taetigk"].apply(ft.get_word_vector).values
        X_ft = np.vstack((X_ft[i] for i in range(len(X_ft))))
        fasttext_proba = ft_model.predict_proba(X_ft)

        # generiere Meta daten
        if self.selbstständige == "ohne":
            X_meta = self.w1_training[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].to_numpy()
        elif self.selbstständige == "nur":
            X_meta = self.w1_training[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']].to_numpy()
        meta_proba = meta_model.predict_proba(X_meta)

        # Füge Daten  zusammen
        X_train = np.concatenate((fasttext_proba, meta_proba), axis=1)
        y_train = self.w1_training[self.oesch].astype(int).to_numpy()

        # generiere Fasttext daten für validation
        X_ft_val = self.w1_validation["taetigk"].apply(ft.get_word_vector).values
        X_ft_val = np.vstack((X_ft_val[i] for i in range(len(X_ft_val))))
        fasttext_proba_val = ft_model.predict_proba(X_ft_val)

        # generiere Meta daten für validation
        if self.selbstständige == "ohne":
            X_meta_val = self.w1_validation[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].to_numpy()
        elif self.selbstständige == "nur":
            X_meta_val = self.w1_validation[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']].to_numpy()
        meta_proba_val = meta_model.predict_proba(X_meta_val)

        # Füge Daten  zusammen
        X_val = np.concatenate((fasttext_proba_val, meta_proba_val), axis=1)
        y_val = self.w1_validation[self.oesch].astype(int).to_numpy()

        trainingsdaten = [X_train, y_train]
        validierungsdaten = [X_val, y_val]
        return trainingsdaten, validierungsdaten
