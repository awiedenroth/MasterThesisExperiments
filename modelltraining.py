import wandb
import xgboost as xgb
from wandb.xgboost import wandb_callback
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def train_ft(X, y, config):
    if config["ft_model"] == "xgboost":
        ft_model = xgb.XGBClassifier(n_jobs=-1, random_state= config["random_seed"])
        ft_model.fit(X, y, verbose=True,  callbacks=[wandb_callback()])

    if config["ft_model"] == "linear":
        ft_model = LogisticRegression(n_jobs=-1, random_state= config["random_seed"])
        ft_model.fit(X, y)

    if config["ft_model"] == "nn":
        ft_model = MLPClassifier(hidden_layer_sizes= (100,), random_state= config["random_seed"], max_iter=300)
        ft_model.fit(X, y)

    return ft_model

def train_meta(X, y, config):
    if config["meta_model"] == "xgboost":
        meta_model = xgb.XGBClassifier(n_jobs=-1, random_state= config["random_seed"])
        meta_model.fit(X, y, verbose=True,  callbacks=[wandb_callback()])

    if config["meta_model"] == "linear":
        meta_model = LogisticRegression(n_jobs=-1, random_state= config["random_seed"])
        meta_model.fit(X, y)

    if config["meta_model"] == "nn":
        meta_model = MLPClassifier(hidden_layer_sizes= (100,), random_state= config["random_seed"], max_iter=300)
        meta_model.fit(X, y)

    return meta_model

def train_combi(X, y, config):

    if config["combi_model"] == "xgboost":
        combi_model = xgb.XGBClassifier(n_jobs=-1, random_state= config["random_seed"])
        combi_model.fit(X, y, verbose=True,  callbacks=[wandb_callback()])

    if config["combi_model"] == "linear":
        combi_model = LogisticRegression(n_jobs=-1, random_state= config["random_seed"])
        combi_model.fit(X, y)

    if config["combi_model"] == "nn":
        combi_model = MLPClassifier(hidden_layer_sizes= (100,), random_state= config["random_seed"], max_iter=300)
        combi_model.fit(X, y)

    return combi_model