import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
import fastText as fasttext
import fasttext.util
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas



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





    def keyboard_augmentation(string:str) -> str:
        aug = nac.KeyboardAug(aug_char_max=1, lang= "de")
        augmented_text = aug.augment(string)
        #print("Original:")
        #print(string)
        #print("Augmented Text:")
        #print(augmented_text)
        return augmented_text

    #def synonym_augmentation(string:str) -> str:
      #  ...