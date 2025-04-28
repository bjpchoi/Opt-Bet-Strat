import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self):
        sns.set_style("whitegrid")

    def plot_cumulative_profits(self, strategies):
        plt.figure(figsize=(14, 7))
        for k, v in strategies.items():
            if k.startswith("model_top"):
                plt.plot(v.values, label=k, lw=2, alpha=0.9)
        plt.title("Model cumulative profit walks for varying top-p thresholds")
        plt.xlabel("Test match index")
        plt.ylabel("Cumulative profit (units)")
        plt.legend(ncol=3, fontsize="x-small")
        plt.tight_layout()
        plt.show()

    def plot_baseline_comparison(self, strategies):
        plt.figure(figsize=(14, 7))
        highlight = [
            "model_top_0.10",
            "model_top_0.20",
            "model_top_0.50",
            "heavyfav_top_0.10",
            "heavyfav_top_0.20",
            "heavyfav_top_0.50",
            "always_fav",
            "always_ud",
            "random_side",
        ]
        for k in highlight:
            plt.plot(strategies[k].values, label=k, lw=2 if "0.10" in k else 1.2)
        plt.title("Cumulative profit comparison (key strategies)")
        plt.xlabel("Test match index")
        plt.ylabel("Cumulative profit (units)")
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        plt.show()

    def plot_random_control(self, model_final, y_te, y_pred_te):
        n_bets = (y_pred_te >= y_pred_te.quantile(0.9)).sum()
        controls = []
        for _ in range(100):
            ix = np.random.choice(len(y_te), n_bets, replace=False)
            profit = pd.Series(
                np.where(np.isin(np.arange(len(y_te)), ix), y_te, 0.0), index=y_te.index
            ).cumsum()
            controls.append(profit.iloc[-1])
        p_val = (np.sum(np.array(controls) >= model_final) + 1) / (len(controls) + 1)

        plt.figure(figsize=(8, 5))
        sns.histplot(controls, kde=True, bins=20)
        plt.axvline(model_final, color="red", ls="--", lw=2, label=f"Model p=0.10\nprofit={model_final:.1f}")
        plt.title(f"Random control distribution (p-value={p_val:.3f})")
        plt.xlabel("Final cumulative profit")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_prediction_distribution(self, y_pred_te):
        plt.figure(figsize=(6, 4))
        sns.histplot(y_pred_te, kde=True, bins=40, color="purple")
        plt.title("Distribution of model-predicted profit (test set)")
        plt.xlabel("Predicted profit per bet")
        plt.tight_layout()
        plt.show()

    def plot_expected_profit(self, implied_prob_te, y_te):
        bins = pd.qcut(implied_prob_te, q=20, duplicates="drop")
        exp_profit = y_te.groupby(bins).mean()
        midpoints = implied_prob_te.groupby(bins).mean()

        plt.figure(figsize=(7, 5))
        plt.plot(midpoints.values, exp_profit.values, marker="o")
        plt.title("Empirical expected profit vs favourite implied win-probability")
        plt.xlabel("Implied win probability (1 / odds)")
        plt.ylabel("Mean profit per bet (units)")
        plt.tight_layout()
        plt.show() 