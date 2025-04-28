import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .strategies import BettingStrategies
from .model import ModelSelector
from .visualizer import Visualizer

class TennisProfitAnalyzer:
    def __init__(self):
        warnings.filterwarnings("ignore")
        np.random.seed(42)
        
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.strategies = BettingStrategies()
        self.model_selector = ModelSelector()
        self.visualizer = Visualizer()

    def preprocess_data(self, df):
        X = self.feature_engineer.build_feature_matrix(df)
        y_fav = self.strategies.profit_always_fav(df)
        y_ud = self.strategies.profit_always_underdog(df)
        fav_odds = X["fav_odds"]

        num_cols = X.select_dtypes(float).columns
        X[num_cols] = StandardScaler().fit_transform(
            SimpleImputer(strategy="median").fit_transform(X[num_cols])
        )

        split = int(len(df) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y_fav.iloc[:split], y_fav.iloc[split:]
        y_ud_te = y_ud.iloc[split:]
        fav_odds_te = fav_odds.iloc[split:]

        return X_tr, X_te, y_tr, y_te, y_ud_te, fav_odds_te

    def run_analysis(self):
        df = self.data_loader.load_dataframe()
        df = df.dropna(subset=["B365W", "B365L"]).reset_index(drop=True)
        
        X_tr, X_te, y_tr, y_te, y_ud_te, fav_odds_te = self.preprocess_data(df)
        
        feats, model, adj_r2 = self.model_selector.forward_stepwise_selection(X_tr, y_tr)
        print(f"Selected features ({len(feats)} total, adj-R² = {adj_r2:.4f}):")
        for f in feats:
            print(f"  - {f}")

        y_pred_te = pd.Series(model.predict(X_te[feats]), index=X_te.index)
        
        strategies = self.generate_strategies(y_pred_te, y_te, y_ud_te, fav_odds_te)
        
        self.visualizer.plot_cumulative_profits(strategies)
        self.visualizer.plot_baseline_comparison(strategies)
        self.visualizer.plot_random_control(strategies["model_top_0.10"].iloc[-1], y_te, y_pred_te)
        self.visualizer.plot_prediction_distribution(y_pred_te)
        self.visualizer.plot_expected_profit(1 / fav_odds_te, y_te)

    def generate_strategies(self, y_pred_te, y_te, y_ud_te, fav_odds_te):
        strategies = {}
        p_grid = np.linspace(0.1, 1.0, 10)
        
        for p in p_grid:
            cutoff = y_pred_te.quantile(1 - p)
            mask = y_pred_te >= cutoff
            profit = pd.Series(np.where(mask, y_te, 0.0), index=y_te.index).cumsum()
            strategies[f"model_top_{p:.2f}"] = profit

        for p in p_grid:
            cutoff = fav_odds_te.quantile(p)
            mask = fav_odds_te <= cutoff
            profit = pd.Series(np.where(mask, y_te, 0.0), index=y_te.index).cumsum()
            strategies[f"heavyfav_top_{p:.2f}"] = profit

        strategies["always_fav"] = y_te.cumsum()
        strategies["always_ud"] = y_ud_te.cumsum()

        rnd = np.random.RandomState(0).randint(0, 2, size=len(y_te))
        rnd_profit = pd.Series(np.where(rnd == 1, y_te.values, y_ud_te.values), index=y_te.index).cumsum()
        strategies["random_side"] = rnd_profit

        return strategies

def main():
    analyzer = TennisProfitAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 