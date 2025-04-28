import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tqdm.auto import tqdm

class ModelSelector:
    def __init__(self, tol=1e-5):
        self.tol = tol

    def forward_stepwise_selection(self, X, y):
        remaining = list(X.columns)
        selected = []
        best_adj_r2 = 0.0
        n = len(y)
        final_model = None

        while remaining:
            best_candidate = None
            best_score = best_adj_r2
            best_model = None

            for cand in tqdm(remaining, desc="Forward stepwise candidates"):
                feat_list = selected + [cand]
                model = LinearRegression().fit(X[feat_list], y)
                r2 = r2_score(y, model.predict(X[feat_list]))
                p = len(feat_list)
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                if adj_r2 > best_score + self.tol:
                    best_score = adj_r2
                    best_candidate = cand
                    best_model = model

            if best_candidate is None:
                break

            remaining.remove(best_candidate)
            selected.append(best_candidate)
            best_adj_r2 = best_score
            final_model = best_model

        return selected, final_model, best_adj_r2

    def get_base_models(self, X, y):
        base_models = {
            "LIN": LinearRegression(),
            "SVR": SVR(C=2.0, gamma="scale"),
            "RF": RandomForestRegressor(n_estimators=300, min_samples_leaf=25, n_jobs=-1),
            "ADA": AdaBoostRegressor(n_estimators=400, learning_rate=0.05),
            "MLP": MLPRegressor(hidden_layer_sizes=(64,32), alpha=1e-3, max_iter=400),
            "XGB": xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=500),
            "LGBM": lgb.LGBMRegressor(max_depth=5, learning_rate=0.1, n_estimators=500),
            "CAT": cb.CatBoostRegressor(depth=5, learning_rate=0.1, iterations=500, verbose=False)
        }
        return base_models

    def get_ensemble(self, X, y):
        base_models = self.get_base_models(X, y)
        stack_estimators = [(name, clone(mdl)) for name, mdl in base_models.items()]
        stack = StackingRegressor(estimators=stack_estimators, final_estimator=LinearRegression())
        return stack.fit(X, y), base_models 