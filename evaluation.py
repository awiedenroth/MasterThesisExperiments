from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import hamming_loss


class Evaluierer:
    @staticmethod
    def make_evaluation(model, X_train, y_train, X_val, y_val):

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        result = {}
        result["train accuracy"] = model.score(X_train, y_train)
        result["train balanced acc"] = balanced_accuracy_score(y_train, y_train_pred)
        result["train balanced adjusted accuracy"] = balanced_accuracy_score(y_train, y_train_pred, adjusted=True)

        result["validation accuracy"] = model.score(X_val, y_val)
        result["validation balanced acc"] = balanced_accuracy_score(y_val, y_val_pred)
        result["validation balanced adjusted accuracy"] = balanced_accuracy_score(y_val, y_val_pred, adjusted=True)

        result["micro-f1 score"] = f1_score(y_val, y_val_pred, average='micro')
        result["macro-f1 score"] = f1_score(y_val, y_val_pred, average='macro')
        result["precision score"] = precision_score(y_val, y_val_pred, average='weighted')
        result["recall score"] = recall_score(y_val, y_val_pred, average='weighted')
        result["hamming loss"] = hamming_loss(y_val, y_val_pred)

        return result

    # todo: muss getestet werden
    @staticmethod
    def make_evaluation_confidence(model, X_train, y_train, X_val, y_val, confidence):
        y_train_pred_proba = model.predict_proba(X_train)
        y_val_pred_proba = model.predict_proba(X_val)

        total = len(X_train)
        deleted = 0
        X_train_clean = []
        y_train_clean = []
        for X_train, y_train, y_train_pred_proba in zip(X_train,y_train,y_train_pred_proba):

            if max(y_train_pred_proba) > confidence:
                X_train_clean.append(X_train)
                y_train_clean.append(y_train)
            else:
                deleted += 1

        y_train_pred = model.predict(np.asarray(X_train_clean))
        y_train = np.asarray(y_train_clean)

        total_val = len(X_val)
        deleted_val = 0
        X_val_clean = []
        y_val_clean = []
        for X_val, y_val, y_val_pred_proba in zip(X_val,y_val,y_val_pred_proba):

            if max(y_val_pred_proba) > confidence:
                X_val_clean.append(X_val)
                y_val_clean.append(y_val)

            else:
                deleted_val += 1

        y_val_pred = model.predict(np.asarray(X_val_clean))
        y_val = np.asarray(y_val_clean)


        result = {}
        result["confidence"] = confidence
        result["deleted datapoints:"] = deleted
        result["percentage deleted train:"] = deleted/total
        result["train accuracy"] = accuracy_score(y_train, y_train_pred)
        result["train balanced acc"] = balanced_accuracy_score(y_train, y_train_pred)
        result["train balanced adjusted accuracy"] = balanced_accuracy_score(y_train, y_train_pred, adjusted=True)
        result["deleted datapoints val:"] = deleted_val
        result["percentage deleted val:"] = deleted_val / total_val
        result["validation accuracy"] = accuracy_score(y_val, y_val_pred)
        result["validation balanced acc"] = balanced_accuracy_score(y_val, y_val_pred)
        result["validation balanced adjusted accuracy"] = balanced_accuracy_score(y_val, y_val_pred, adjusted=True)

        return result