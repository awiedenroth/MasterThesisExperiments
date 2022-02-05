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

    def __init__(self, X_ft, y_ft, X_meta, y_meta, lowercase:bool):

        self.X_ft = X_ft
        self.y_ft = y_ft
        self.X_meta = X_meta
        self.y_meta = y_meta
        self.lowercase = lowercase


    def augment_data(self):

        if self.lowercase:
            # erstelle trainingsdatensatz für Fasttext Modell, indem alle Zeilen verdoppelt werden einmal mit .lower()
            ft_training_lower = deepcopy(self.X_ft)
            ft_training_lower["taetigk"] = ft_training_lower["taetigk"].apply(lambda x: x.lower())
            ft_data_2 = pd.concat([ft_training_lower, deepcopy(self.X_ft)])

        return self.X_ft, self.y_ft, self.X_meta, self.y_meta