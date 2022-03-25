import os

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import plot_confusion_matrix
import wandb
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import shutil




class Evaluierer:
    @staticmethod
    def make_evaluation(model, X_train, y_train, X_val, y_val, config, modelname, run):

        y_train_pred = model.predict(X_train)
        y_train_pred_conf = np.max(model.predict_proba(X_train), axis=1)
        y_val_pred = model.predict(X_val)
        y_val_pred_conf = np.max(model.predict_proba(X_val), axis=1)

        pd.DataFrame.from_dict({"y_pred": y_train_pred, "y_true": y_train, "confidence": y_train_pred_conf}).to_csv(f"{modelname}_run{run}_train.csv", sep= ";")
        pd.DataFrame.from_dict({"y_pred": y_val_pred, "y_true": y_val, "confidence": y_val_pred_conf}).to_csv(f"{modelname}_run{run}_val.csv", sep=";")

        shutil.copy(f"{modelname}_run{run}_train.csv", os.path.join(wandb.run.dir, f"{modelname}_run{run}_train.csv"))
        shutil.copy(f"{modelname}_run{run}_val.csv", os.path.join(wandb.run.dir, f"{modelname}_run{run}_val.csv"))

        wandb.save(f"{modelname}_run{run}_train.csv")
        wandb.save(f"{modelname}_run{run}_val.csv")

        result = {}
        result[f"{modelname} train accuracy"] = model.score(X_train, y_train)
        result[f"{modelname} train balanced acc"] = balanced_accuracy_score(y_train, y_train_pred)
        result[f"{modelname} train balanced adjusted accuracy"] = balanced_accuracy_score(y_train, y_train_pred, adjusted=True)

        result[f"{modelname} validation accuracy"] = model.score(X_val, y_val)
        result[f"{modelname} validation balanced acc"] = balanced_accuracy_score(y_val, y_val_pred)
        result[f"{modelname} validation balanced adjusted accuracy"] = balanced_accuracy_score(y_val, y_val_pred, adjusted=True)

        result[f"{modelname} micro-f1 score"] = f1_score(y_val, y_val_pred, average='micro')
        result[f"{modelname} macro-f1 score"] = f1_score(y_val, y_val_pred, average='macro')
        result[f"{modelname} precision score"] = precision_score(y_val, y_val_pred, average='weighted')
        result[f"{modelname} recall score"] = recall_score(y_val, y_val_pred, average='weighted')
        result[f"{modelname} hamming loss"] = hamming_loss(y_val, y_val_pred)

        #wandb.log({f"{modelname} run {run}": result})

        if config["oesch"] == "oesch8":
            if config["selbstständige"] == "ohne":
                labels= [3,4,5,6,7,8]
            elif config["selbstständige"] == "nur":
                labels= [1,2]
        if config["oesch"] == "oesch16":
            if config["selbstständige"] == "ohne":
                labels= [5,6,7,8,9,10,11,12,13,14,15,16]
            elif config["selbstständige"] == "nur":
                labels= [1,2,3,4]


        result_train = classification_report(y_train, y_train_pred, labels=labels, output_dict=True, zero_division=0)
        result_val = classification_report(y_val, y_val_pred, labels=labels, output_dict=True, zero_division=0)

        result_train.pop("accuracy")
        result_val.pop("accuracy")

        #report= wandb.Table(columns=result_train.keys(), data=result_train.values())

        #wandb.run.summary[f"{modelname} train run {run}"] = result_train
        #wandb.run.summary[f"{modelname} validation run {run}"] = result_val
        wandb.log({f"{modelname} training report run {run}": result_train})
        wandb.log({f"{modelname} validation report run {run}": result_val})



        fig0 = plt.figure(figsize=(10, 10))
        ax0=plt.gca()
        plot_confusion_matrix(model, X_val, y_val, normalize=None, ax=ax0)
        fig0.tight_layout()
        #wandb.log({f"{modelname} run{run}": plt})


        fig1= plt.figure(figsize=(10, 10))
        ax1= plt.gca()
        plot_confusion_matrix(model, X_val, y_val, normalize="true", ax=ax1)
        fig1.tight_layout()
        #wandb.log({f"{modelname} normalized run{run}": plt})
        wandb.log({f"{modelname} run {run}": result,
                   f"{modelname} run {run} confusion matrix": fig0, f"{modelname} run {run} confusion matrix normalized": fig1})


        return result, result_train, result_val
        #return result_train, result_val

    # todo: muss getestet werden
    @staticmethod
    def make_evaluation_confidence(model, X_train, y_train, X_val, y_val, confidence, run):
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

        if len(y_val_clean) > 1:
            y_val_pred = model.predict(np.asarray(X_val_clean))
            y_val = np.asarray(y_val_clean)


            result = {}
            result[f"confidence: "] = confidence
            result[f"deleted datapoints @{confidence}: "] = deleted
            result[f"percentage deleted train @{confidence}: "] = deleted/total
            result[f"train accuracy @{confidence}: "] = accuracy_score(y_train, y_train_pred)
            result[f"train balanced acc @{confidence}: "] = balanced_accuracy_score(y_train, y_train_pred)
            result[f"train balanced adjusted accuracy @{confidence}: "] = balanced_accuracy_score(y_train, y_train_pred, adjusted=True)
            result[f"deleted datapoints val @{confidence}: "] = deleted_val
            result[f"percentage deleted val @{confidence}: "] = deleted_val / total_val
            result[f"validation accuracy @{confidence}: "] = accuracy_score(y_val, y_val_pred)
            result[f"validation balanced acc @{confidence}: "] = balanced_accuracy_score(y_val, y_val_pred)
            result[f"validation balanced adjusted accuracy @{confidence}: "] = balanced_accuracy_score(y_val, y_val_pred, adjusted=True)
        else:
            result = {"confidence: " : confidence, f"percentage deleted val @{confidence}: ": "100%"}

        wandb.log({f"combi model run {run} at confidence {confidence}": result})

        return result