import pandas as pd
import numpy as np


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
        """
        Liest die korrekten Daten ein, nimmt nur die zeilen mit bzw ohne selbstständigen

        :param config: configuration, in particular if selbstständige are included and which oesch schema
        :param path: path where data is in
        :return: dataframe which contains only those rows which fit to selbstständige selection
        """
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
        """
        castet die einträge in den spalten für das meta modell von string nach int, nimmt mehr spalten wenn es nur
        selbstständige sind, da bei denen 2 weitere Attribute abgefragt wurden, ersetzt fehlende werte mit -1
        :param config: config, ob mit oder ohne selbstständige
        :param df: datensatz
        :return: datensatz mit int64 als werten in allen spalten für meta modell
        """
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

    @staticmethod
    def _select_columns_for_ft(config: dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        macht kopie des datensatzes und nimmt davon nur die für das fasttext modell relevanten spalten, nämlich taetigk
        und oesch8 bzw oesch16
        :param config: config ob oesch8 oder oesch16
        :param df: vollständigen datensatz
        :return: teil von kopie des datensatzes, der für fasttext modell relevant ist
        """
        trainingsdaten_ft = df[["taetigk", config["oesch"]]].copy()
        return trainingsdaten_ft

    @staticmethod
    def _make_meta_numpy(config: dict, df: pd.DataFrame):
        """
        erstellt datensatz für meta, indem die richtigen spalten gewählt werden (je nachdem ob selbstständig oder nicht)
        und die werte alle in ein numpy array gecastet werden, die labels auch als numpy array
        :param config: config ob ohne oder nur selbstständige
        :param df: vollständiger datensatz
        :return: numpy arrays mit meta daten und labels"""
        if config["selbstständige"] == "ohne":
            # erstelle trainingsdatemsatz für meta Modell als np array
            X_w1_meta = df[['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3',
                                          'taetigk_m4', 'taetigk_m5','beab', 'einkommen', 'besch_arbzeit']].to_numpy()

        elif config["selbstständige"] == "nur":
            X_w1_meta = df[['branche2', 'taetigk_hierar', 'taetigk_m1', 'taetigk_m2', 'taetigk_m3',
                                          'taetigk_m4', 'taetigk_m5', 'beab', 'einkommen', 'besch_arbzeit', 'erw_stat', 'selbst_gr']].to_numpy()

        y_w1_meta = df[config["oesch"]].astype(int).to_numpy()

        # ich gebe zurück meta daten, meta_labels als numpy arrays?
        return X_w1_meta, y_w1_meta

    @staticmethod
    def make_dataset(config: dict):
        """
        fügt andere Methoden zusammen und führt sie in richtiger Reihenfolge aus
        :param config: config
        :return: trainingsdaten als df, numpy arrays mit meta daten und meta labels
        """
        Welle_1_clean = Datengenerierer._read_in_correct_data(config)
        w1_training = Datengenerierer._cast_to_int_and_replace_missing(config, Welle_1_clean)
        trainingsdaten_ft = Datengenerierer._select_columns_for_ft(config, w1_training)
        X_w1_meta, y_w1_meta = Datengenerierer._make_meta_numpy(config, w1_training)

        return trainingsdaten_ft, X_w1_meta, y_w1_meta




