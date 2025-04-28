import numpy as np
import pandas as pd

class NaiveBettingStrategies:
    def __init__(self):
        pass

    def profit_always_fav(self, df):
        ow, ol = df["B365W"].values, df["B365L"].values
        fav = ow < ol
        fo = np.where(fav, ow, ol)
        return pd.Series(np.where(fav, fo - 1.0, -1.0), index=df.index)

    def profit_always_underdog(self, df):
        ow, ol = df["B365W"].values, df["B365L"].values
        fav = ow < ol
        ud = np.where(fav, ol, ow)
        return pd.Series(np.where(~fav, ud - 1.0, -1.0), index=df.index)

    def profit_random(self, df):
        """Random betting strategy that bets on either favorite or underdog with equal probability"""
        ow, ol = df["B365W"].values, df["B365L"].values
        fav = ow < ol
        # Generate random choices for each match
        random_choices = np.random.choice([True, False], size=len(df))
        # Select odds based on random choice
        selected_odds = np.where(random_choices, np.where(fav, ow, ol), np.where(fav, ol, ow))
        # Calculate profit: if random choice matches actual outcome, profit is odds-1, else -1
        return pd.Series(np.where(random_choices == fav, selected_odds - 1.0, -1.0), index=df.index) 
