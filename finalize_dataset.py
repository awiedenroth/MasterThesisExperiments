import fastText as fasttext
import fasttext.util
import numpy as np
ft = fasttext.load_model('cc.de.300.bin')
from caching import mem
# ich nehme das df das die spalten "taetigk" und config["oesch"] enthält
# ich füge embeddings hinzu
# dann nehme ich embeddings und labels, shuffle und erstelle datensatz daraus

@mem.cache
def finalize_data(df, config):
    df["embeddings"] = df["taetigk"].apply(ft.get_word_vector)
    # shuffle Datensatz
    df = df.sample(frac=1, random_state= config["random_seed"])
    # generiere X und y
    X = df["embeddings"].values
    y = df[config["oesch"]].astype(int)
    # mache matrix aus den Ddaten
    X = np.vstack((X[i] for i in range(len(X))))
    return X,y