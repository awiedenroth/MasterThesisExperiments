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


    def __init__(self, ratio: float, augmentation:bool, oesch:str, selbstständige: str, random_state:int):
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
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat','selbst_gr']].applymap(lambda x: missing_val(x))

        # teile datensatz in 60% trainings und 40% validierungsdatensatz
        self.w1_training = Welle_1_clean.sample(frac=self.ratio, random_state=self.random_state)
        self.w1_validation = Welle_1_clean.drop(self.w1_training.index)



    def make_dataset(self):

        ft = fasttext.load_model('cc.de.300.bin')

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
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat','selbst_gr']]

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
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat','selbst_gr']]
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
            # Wörterbuch_de als Trainingsdatensatz für Fasttext, ohne Selbstständige

            path = Path("./Wörterbücher/wic_wörterbuch_aufbereitet_oesch.csv")
            Wörterbuch_de = pd.read_csv(path, sep=";")
            Wörterbuch_de = Wörterbuch_de[~Wörterbuch_de.oesch16.isin([-9])]

            # hier könnte man data-augmentation machen indem man lowercase hinzufügt

            # füge Spalte mit Embeddings hinzu
            Wörterbuch_de["embeddings"] = Wörterbuch_de["berufsbezeichnung"].apply(ft.get_word_vector)

            # generiere Trainingsdatensatz mit embeddings und oesch16

            X_wb = Wörterbuch_de["embeddings"].values
            y_wb = Wörterbuch_de[self.oesch].values.astype(int)

            X_wb = np.vstack((X_wb[i] for i in range(len(X_wb))))

        # Datensatz X_wb, y_wb besteht aus einer Matrix X_wb mit der Anzahl der Wörterbucheinträge als zeilen, und 300 spalten
        # die zeilen stehen für Datenpunkte, die spalten enthalten den 300 dimensionalen Fasttext embedding vektor

        # erstelle trainingsdatensatz für Fasttext Modell, indem alle Zeilen verdoppelt werden einmal mit .lower()
        w1_training_lower = deepcopy(self.w1_training)
        w1_training_lower["taetigk"] = w1_training_lower["taetigk"].apply(lambda x: x.lower())
        w1_training_ft = pd.concat([w1_training_lower, deepcopy(self.w1_training)])
        # füge bei Fasttext Trainingsdatensatz embeddings hinzu
        w1_training_ft["embeddings"] = w1_training_ft["taetigk"].apply(ft.get_word_vector)
        # shuffle trainingsdatensatz
        w1_training_ft = w1_training_ft.sample(frac=1, random_state=self.random_state)
        # generiere X und y
        X_w1_ft = w1_training_ft["embeddings"].values
        y_w1_ft = w1_training_ft[self.oesch].astype(int)
        # mache matrix aus den Trainingsdaten
        X_w1_ft = np.vstack((X_w1_ft[i] for i in range(len(X_w1_ft))))

        if self.selbstständige == "ohne":
            # erstelle trainingsdatemsatz für meta Modell als np array
            X_w1_meta = self.w1_training[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].to_numpy()

        elif self.selbstständige == "nur":
            X_w1_meta = self.w1_training[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat','selbst_gr']].to_numpy()
        y_w1_meta = self.w1_training[self.oesch].astype(int).to_numpy()

        # füge bei Validierungsdatensatz Spalte mit Embeddings hinzu
        self.w1_validation["embeddings"] = self.w1_validation["taetigk"].apply(ft.get_word_vector)
        # erstelle Fasttext validation datensatz
        # generiere X und y
        X_w1v_ft = self.w1_validation["embeddings"].values
        y_w1v_ft = self.w1_validation[self.oesch].astype(int)
        # mache matrix aus den Trainingsdaten
        X_w1v_ft = np.vstack((X_w1v_ft[i] for i in range(len(X_w1v_ft))))

        # erstelle meta Model validation datensatz
        if self.selbstständige == "ohne":
            X_w1v_meta = self.w1_validation[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit']].to_numpy()
        elif self.selbstständige == "nur":
            X_w1v_meta = self.w1_validation[
                ['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3', 'taetigk_m4', 'taetigk_m5',
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat','selbst_gr']].to_numpy()
        y_w1v_meta = self.w1_validation[self.oesch].astype(int).to_numpy()

        # verbinde Datensätze wb und w1 fasttext für das Training des Fasttext Modells
        if self.selbstständige == "ohne":
            X_ft_train = np.concatenate((X_wb, X_w1_ft))
            y_ft_train = np.concatenate((y_wb, y_w1_ft))
        elif self.selbstständige == "nur":
            X_ft_train = X_w1_ft
            y_ft_train = y_w1_ft

        from sklearn.utils import shuffle
        X_ft_train, y_ft_train = shuffle(X_ft_train, y_ft_train, random_state=self.random_state)

        # verbinde Datensätze meta und w1 meta für das Training des Fasttext Modells
        X_meta_train = np.concatenate((X_meta, X_w1_meta))
        y_meta_train = np.concatenate((y_meta, y_w1_meta))

        from sklearn.utils import shuffle
        X_meta_train, y_meta_train = shuffle(X_meta_train, y_meta_train, random_state=self.random_state)

        trainingsdaten = [X_ft_train, y_ft_train, X_meta_train, y_meta_train]
        validierungsdaten = [X_w1v_ft, y_w1v_ft, X_w1v_meta, y_w1v_meta]
        return trainingsdaten, validierungsdaten

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
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat','selbst_gr']].to_numpy()
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
                 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat','selbst_gr']].to_numpy()
        meta_proba_val = meta_model.predict_proba(X_meta_val)

        # Füge Daten  zusammen
        X_val = np.concatenate((fasttext_proba_val, meta_proba_val), axis=1)
        y_val = self.w1_validation[self.oesch].astype(int).to_numpy()

        trainingsdaten = [X_train, y_train]
        validierungsdaten = [X_val, y_val]
        return trainingsdaten, validierungsdaten

