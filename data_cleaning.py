import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
import fastText as fasttext
import fasttext.util
import string

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from caching import mem

nltk.download('stopwords')
string.punctuation
# alle daten werden cleaned auch die validierungsdaten, das muss man dann immer machen

@mem.cache
def clean_data(fasttext_df, config):

    if config["lowercase"] == True:
        fasttext_df["taetigk"] = fasttext_df["taetigk"].apply(lambda x: lowercase(x))

    if config["remove_num_punc"] == True:
        fasttext_df["taetigk"] = fasttext_df["taetigk"].apply(lambda x: remove_punctuation(x))
        fasttext_df["taetigk"] = fasttext_df["taetigk"].apply(lambda x: remove_numbers(x))

    if config["remove_stopwords"] == True:
        fasttext_df["taetigk"] = fasttext_df["taetigk"].apply(lambda x: remove_stopwords(x))

    return fasttext_df

def lowercase(string: str) -> str:
    return string.lower()

def remove_stopwords(string: str) -> str:
    stopWords = set(stopwords.words('german'))
    tokenized = word_tokenize(string)
    return TreebankWordDetokenizer().detokenize([w for w in tokenized if w not in stopWords])

def remove_numbers(text:str) -> str:
    no_num = [words for words in text if not words.isdigit()]
    words_wo_num = ''.join(no_num)
    return words_wo_num

def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation or words == "-"]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct



