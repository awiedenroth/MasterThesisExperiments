from presplit_datengenerierung import Datengenerierer
from modelltraining import Modelltrainer
from evaluation import Evaluierer
from sklearn.model_selection import KFold
import pickle
import json
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")



if __name__ == "__main__":
    # hier muss ich angeben ob ich Oesch8 oder Oesch16 möchte und ob ich "nur" Selbstständige oder "ohne" Selbstständige haben möchte
    #Todo: hier könnte ich statt strings für oesch und selbstständige etwas eleganter Oesch=8 bzw Oesch=16 und Selbstständige = true/false machen
    datengenerierer = Datengenerierer("oesch8","nur")

    X_fasttext, y_fasttext, X_meta, y_meta = datengenerierer.make_dataset()
    #dict zum abspeichern der ergebnisse
    ergebnisse = {}
    # index um zu tracken bei welchem durchgang man ist
    i = 0
    kf = KFold(n_splits=8)
    # die for schleife geht k mal durch
    for train_index, test_index in kf.split(X_fasttext):
        print("Durchgang ", i)
        # erstelle die fasttext trainings und test daten
        X_train_fasttext, X_test_fasttext = X_fasttext[train_index], X_fasttext[test_index]
        y_train_fasttext, y_test_fasttext = y_fasttext[train_index], y_fasttext[test_index]
        # erstelle die meta modell trainings und test daten
        X_train_meta, X_test_meta = X_meta[train_index], X_meta[test_index]
        y_train_meta, y_test_meta = y_meta[train_index], y_meta[test_index]

        #Todo: Datenaugmentierung

        # hier füge ich die anderen Metriken hinzu
        modelltrainer = Modelltrainer(0, "merror")
        evaluierer = Evaluierer()

        print("trainiere meta Modell")
        meta_model = modelltrainer.train_meta(X_train_meta, y_train_meta, hyperparameter=None)
        evaluation_meta = evaluierer.make_evaluation(meta_model, X_train_meta, y_train_meta,
                                                         X_test_meta, y_test_meta)
        print("Meta Modell: ", evaluation_meta)
        #with open("Trained_Models/meta_8_80_n_0.pkl", "wb") as f:
        #    pickle.dump(meta_model, f)
        #json.dump(evaluation_meta, open("Ergebnisse/meta_8_80_o_0.json", 'w'))


        print("trainiere fasttext Modell")
        fasttext_model = modelltrainer.train_ft(X_train_fasttext, y_train_fasttext, hyperparameter=None)
        evaluation_fasttext = evaluierer.make_evaluation(fasttext_model, X_train_fasttext, y_train_fasttext,
                                   X_test_fasttext, y_test_fasttext)
        print("Fasttext Modell: ", evaluation_fasttext)
        #with open("Trained_Models/fasttext_8_80_n_0.pkl", "wb") as f:
            #pickle.dump(fasttext_model, f)
        #json.dump(evaluation_fasttext, open("Ergebnisse/fasttext_8_80_n_0.json", 'w'))

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
        ergebnisse[i] = {"meta":evaluation_meta, "fasttext": evaluation_fasttext, "combi": evaluation_combi}
        i = i+1

    json.dump(ergebnisse, open("Ergebnisse/kfold_8_nur", 'w'))