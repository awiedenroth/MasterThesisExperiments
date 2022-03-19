from datenaugementierung import augment_data
from finalize_dataset import finalize_data
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

class PS_Datengenerierer:

    @staticmethod
    def make_ps_data_meta(X_meta, y_meta, X_meta_z, y_meta_z, configuration, train_index, test_index):

        # erstelle die meta modell trainings und test daten
        X_train_meta, X_test_meta = X_meta[train_index], X_meta[test_index]
        y_train_meta, y_test_meta = y_meta[train_index], y_meta[test_index]

        if configuration["meta_zusatzdaten"] == True:
            X_train_meta = np.concatenate((X_train_meta, X_meta_z))
            y_train_meta = np.concatenate((y_train_meta, y_meta_z))

        # Meta daten werden geshufflet
        X_train_meta, y_train_meta = shuffle(X_train_meta, y_train_meta, random_state=configuration["random_seed"])
        X_test_meta, y_test_meta = shuffle(X_test_meta, y_test_meta, random_state=configuration["random_seed"])

        return X_train_meta, y_train_meta, X_test_meta, y_test_meta

    @staticmethod
    def make_ps_data_ft(fasttext_df, fasttext_wb_df, configuration, train_index, test_index):

        X_train_fasttext, X_test_fasttext = fasttext_df.iloc[train_index], fasttext_df.iloc[test_index]

        # füge zu den trainingsdatensätzen die zusatzdaten hinzu falls gewünscht
        if configuration["fasttext_zusatzdaten"] == True:
            if configuration["selbstständige"] == "ohne":
                X_train_fasttext = pd.concat([X_train_fasttext, fasttext_wb_df])

        X_train_fasttext = augment_data(X_train_fasttext, configuration)

        # ich erzeuge aus training_df und test_df die embeddings bei den fasttext dingen und shuffle
        X_train_fasttext, y_train_fasttext = finalize_data(X_train_fasttext, configuration, shuffle=True)
        X_test_fasttext, y_test_fasttext = finalize_data(X_test_fasttext, configuration, shuffle=True)

        return X_train_fasttext, y_train_fasttext, X_test_fasttext, y_test_fasttext


    @staticmethod
    def make_ps_data_combi(fasttext_df, X_meta, fasttext_model, meta_model, train_index, test_index, configuration):
        """
        erzeuge Daten für Combi model: stelle nochmal Ursprungsdatensätze her (also ohne augmentierung oder zusatzdaten)
        lasse diese durch trainierte Modelle laufen so dass diese für jede der 8 bzw 16 klassen eine wahrscheinlichkeit
        ausgeben, diese werden dann in ein 2 dimensionales np array gepackt, wo jede zeile ein datenpunkt ist und eine
        spalte die wahrscheinlichkeitsverteilung von fasttext, die andere die wahrscheinlichkeitsverteilung von meta ist
        abschließend shuffle ich die daten noch
        :param fasttext_df:
        :param X_meta:
        :param fasttext_model:
        :param meta_model:
        :param train_index:
        :param test_index:
        :param configuration:
        :return: 
        """
        fasttext_raw_train, y_train_combi = finalize_data(fasttext_df.iloc[train_index], configuration, shuffle=False)
        fasttext_proba = fasttext_model.predict_proba(fasttext_raw_train)
        meta_proba = meta_model.predict_proba(X_meta[train_index])
        X_train_combi = np.concatenate((fasttext_proba, meta_proba), axis=1)
        X_train_combi, y_train_combi = shuffle(X_train_combi, y_train_combi, random_state=configuration["random_seed"])

        # erzeuge validierungsdaten für combi model
        fasttext_raw_test, y_test_combi = finalize_data(fasttext_df.iloc[test_index], configuration, shuffle=False)
        fasttext_proba_test = fasttext_model.predict_proba(fasttext_raw_test)
        meta_proba_test = meta_model.predict_proba(X_meta[test_index])
        X_test_combi = np.concatenate((fasttext_proba_test, meta_proba_test), axis=1)
        X_test_combi, y_test_combi = shuffle(X_test_combi, y_test_combi, random_state=configuration["random_seed"])

        return X_train_combi, y_train_combi, X_test_combi, y_test_combi
