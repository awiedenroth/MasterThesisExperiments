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
from caching import mem

@mem.cache
def augment_data(df, config):
    if config["keyboard_aug"] == True:
        augmented_df = df.copy()
        augmented_df["taetigk"] = augmented_df["taetigk"].apply(lambda x: keyboard_augmentation(x))
        #augmented_df["embeddings"] = augmented_df["taetigk"].apply(ft.get_word_vector)
        df = pd.concat([df, augmented_df])

    return df

    # fasttext_df ist ein df mit 3 Spalten: "taetigk","embeddings", self.oesch
    # hier nehmen wir nun die spalte taetigk, modifizieren sie, leiten daraus embeddings ab und fügen dann mit dem
    # identischen label den neuen Datenpunkt hinzu
    # Erinnerung: wir iterieren nicht durch df, sondern wenden funktion an auf alle zeilen

    #def modify_string(self):

def keyboard_augmentation(x, string:str) -> str:
    aug = nac.KeyboardAug(aug_char_max=1, lang= "de")
    augmented_text = aug.augment(string)
    #print("Original:")
    #print(string)
    #print("Augmented Text:")
    #print(augmented_text)
    return augmented_text

#def synonym_augmentation(string:str) -> str:
  #  ...