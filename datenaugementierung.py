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

# wie übergebe ich die Daten? Ich brauche ja X und y, wie gehe ich damit um? Ich muss sicherstellen, dass x und y immernoch
# zusammen gehören ganz am Ende, wie kann ich das tun???
class Augmentierer:

    def __init__(self, fasttext_df, configuration):

        self.fasttext_df = fasttext_df
        self.config = configuration


    def augment_data(self):
        # fasttext_df ist ein df mit 3 Spalten: "taetigk","embeddings", self.oesch
        # hier nehmen wir nun die spalte taetigk, modifizieren sie, leiten daraus embeddings ab und fügen dann mit dem
        # identischen label den neuen Datenpunkt hinzu
        # Erinnerung: wir iterieren nicht durch df, sondern wenden funktion an auf alle zeilen

        #def modify_string(self):



        if self.lowercase:
            # erstelle trainingsdatensatz für Fasttext Modell, indem alle Zeilen verdoppelt werden einmal mit .lower()
            ft_training_lower = deepcopy(self.fasttext_df)
            ft_training_lower["taetigk"] = ft_training_lower["taetigk"].apply(lambda x: x.lower())
            ft_data_2 = pd.concat([ft_training_lower, deepcopy(self.fasttext_df)])

        return self.X_ft, self.y_ft

def lowercase(string:str) -> str:
    return string.lower()

def remove_stopwords(string:str) -> str:
    return [w for w in string if w not in stopwordList]

def remove_numbers(string:str) -> str:
    ...

def remove_punctuation(string:str) -> str:
    ...

def keyboard_augmentation(string:str) -> str:
    ...

def synonym_augmentation(string:str) -> str:
    ...