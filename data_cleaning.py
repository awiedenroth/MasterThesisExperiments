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

nltk.download('stopwords')

# alle daten werden cleaned auch die validierungsdaten, das muss man dann immer machen
class Datacleaner:

    def __init__(self, fasttext_df, configuration):
        ...


    def lowercase(string: str) -> str:
        return string.lower()


    def remove_stopwords(string: str) -> str:
        stopWords = set(stopwords.words('german'))
        tokenized = word_tokenize(string)
        clean = [w for w in tokenized if w not in stopWords]
        return TreebankWordDetokenizer().detokenize([w for w in tokenized if w not in stopWords])

    def remove_numbers(string:str) -> str:
        ...

    def remove_punctuation(string:str) -> str:
        ...

