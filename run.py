from datensatzgenerierung import Datengenerierer
from modelltraining import Modelltrainer
from evaluation import Evaluierer
import pickle
import json
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Name des modells und ergebnisse: modell, oesch16/oesch8, 80%/90%/... Trainingsdaten, o/n "ohne Selbst
# ständige, nur selbstständige, anderer random seed (zb 42)

if __name__ == "__main__":
    datengenerierer = Datengenerierer(0.80,True,"oesch8","nur",0)
    trainingsdaten, validierungsdaten = datengenerierer.make_dataset()

    modelltrainer = Modelltrainer(0, "merror")
    evaluierer = Evaluierer()

    print("trainiere meta Modell")
    meta_model = modelltrainer.train_meta(trainingsdaten[2], trainingsdaten[3], hyperparameter=None)
    evaluation_meta = evaluierer.make_evaluation(meta_model, trainingsdaten[2], trainingsdaten[3],
                                                     validierungsdaten[2], validierungsdaten[3])
    print("Meta Modell: ", evaluation_meta)
    with open("Trained_Models/meta_8_80_n_0.pkl", "wb") as f:
        pickle.dump(meta_model, f)
    json.dump(evaluation_meta, open("Ergebnisse/meta_8_80_o_0.json", 'w'))


    print("trainiere fasttext Modell")
    fasttext_model = modelltrainer.train_ft(trainingsdaten[0], trainingsdaten[1], hyperparameter=None)
    evaluation_fasttext = evaluierer.make_evaluation(fasttext_model, trainingsdaten[0], trainingsdaten[1],
                               validierungsdaten[0], validierungsdaten[1])
    print("Fasttext Modell: ", evaluation_fasttext)
    with open("Trained_Models/fasttext_8_80_n_0.pkl", "wb") as f:
        pickle.dump(fasttext_model, f)
    json.dump(evaluation_fasttext, open("Ergebnisse/fasttext_8_80_n_0.json", 'w'))

    print("trainiere Combi Modell")
    train, val = datengenerierer.make_combi_dataset(fasttext_model, meta_model)
    combi_model = modelltrainer.train_combi(train[0], train[1], hyperparameter=None)
    evaluation_combi = evaluierer.make_evaluation(combi_model, train[0], train[1],
                                                     val[0], val[1])
    print("Combi Modell evaluation: ", evaluation_combi)
    with open("Trained_Models/combi_8_80_n_0.pkl", "wb") as f:
        pickle.dump(combi_model, f)
    json.dump(evaluation_combi, open("Ergebnisse/combi_8_80_n_0.json", 'w'))


    for confidence in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.992,0.994,0.995, 0.996, 0.997]:
        path = "Ergebnisse/combi_8_80_n_0_"
        evaluation_combi_confidence = evaluierer.make_evaluation_confidence(combi_model, train[0], train[1],
                                                      val[0], val[1], confidence)
        print("Combi Modell evaluation mit confidence", confidence, evaluation_combi_confidence)
        path += str(confidence) + ".json"
        json.dump(evaluation_combi_confidence, open(path, 'w'))