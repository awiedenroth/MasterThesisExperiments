import xgboost as xgb

class Modelltrainer:
    def __init__(self, randomstate:int, evalmetric: str):
        self.randomstate = randomstate
        self.evalmetric = evalmetric


    def train_ft(self, X, y, hyperparameter):
        ft_xgb = xgb.XGBClassifier(n_jobs=-1, random_state=self.randomstate)
        ft_xgb.fit(X, y, eval_metric= self.evalmetric, verbose=True)

        return ft_xgb

    def train_meta(self, X, y, hyperparameter):
        meta_xgb = xgb.XGBClassifier(n_jobs=-1, random_state=self.randomstate)
        meta_xgb.fit(X, y, eval_metric=self.evalmetric,verbose=True)
        return meta_xgb

    def train_combi(self, X, y, hyperparameter):
        combi_xgb = xgb.XGBClassifier(n_jobs=-1, random_state=self.randomstate)
        combi_xgb.fit(X, y, eval_metric=self.evalmetric,verbose=True)

        return combi_xgb