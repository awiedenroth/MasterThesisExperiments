import wandb
import xgboost as xgb
from wandb.xgboost import wandb_callback
from caching import mem


@mem.cache
def train_ft(X, y, config):
    ft_xgb = xgb.XGBClassifier(n_jobs=-1, random_state=config["random_seed"])
    ft_xgb.fit(X, y, verbose=True,  callbacks=[wandb_callback()])

    return ft_xgb

@mem.cache
def train_meta(X, y, config):
    meta_xgb = xgb.XGBClassifier(n_jobs=-1, random_state=config["random_seed"])
    meta_xgb.fit(X, y, verbose=True,  callbacks=[wandb_callback()])
    return meta_xgb

@mem.cache
def train_combi(X, y, config):
    combi_xgb = xgb.XGBClassifier(n_jobs=-1, random_state=config["random_seed"])
    combi_xgb.fit(X, y, verbose=True,  callbacks=[wandb_callback()])

    return combi_xgb