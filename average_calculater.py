import numpy as np

def calculate_average(ergebnisse):
    train_accuracy= []
    train_balanced_acc =  []
    result[f"{modelname} train accuracy"] = model.score(X_train, y_train)
    result[f"{modelname} train balanced acc"] = balanced_accuracy_score(y_train, y_train_pred)
    result[f"{modelname} train balanced adjusted accuracy"] = balanced_accuracy_score(y_train, y_train_pred,
                                                                                      adjusted=True)

    result[f"{modelname} validation accuracy"] = model.score(X_val, y_val)
    result[f"{modelname} validation balanced acc"] = balanced_accuracy_score(y_val, y_val_pred)
    result[f"{modelname} validation balanced adjusted accuracy"] = balanced_accuracy_score(y_val, y_val_pred,
                                                                                           adjusted=True)

    result[f"{modelname} micro-f1 score"] = f1_score(y_val, y_val_pred, average='micro')
    result[f"{modelname} macro-f1 score"] = f1_score(y_val, y_val_pred, average='macro')
    result[f"{modelname} precision score"] = precision_score(y_val, y_val_pred, average='weighted')
    result[f"{modelname} recall score"] = recall_score(y_val, y_val_pred, average='weighted')
    result[f"{modelname} hamming loss"] = hamming_loss(y_val, y_val_pred)

    for ergebnis in ergebnisse:


    # ich iteriere durch die dicts für jeden k durchgang und summiere jeweils die werte an den richtigen positionen
    # dann teile ich durch k
    # damit bekomme ich den average metrik für die configuration