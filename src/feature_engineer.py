import numpy as np
import pandas as pd
from .encoders import Encoders

class FeatureEngineer:
    def __init__(self, rng_seed=42):
        self.rng = np.random.RandomState(rng_seed)
        self.encoders = Encoders()

    def build_feature_matrix(self, df):
        mask = self.rng.rand(len(df)) < 0.5
        feats = pd.DataFrame(index=df.index)

        for col in ["rank", "age", "rank_points", "ht"]:
            w, l = df[f"winner_{col}"], df[f"loser_{col}"]
            a = np.where(mask, w, l)
            b = np.where(mask, l, w)
            b = np.where(b == 0, 1, b)
            feats[f"delta_{col}"] = a - b
            feats[f"ratio_{col}"] = a / b

        hand_map = {"R": 1, "L": -1, "U": 0}
        feats["hand_adv"] = np.where(
            mask,
            df["winner_hand"].fillna("U").map(hand_map) - df["loser_hand"].fillna("U").map(hand_map),
            df["loser_hand"].fillna("U").map(hand_map) - df["winner_hand"].fillna("U").map(hand_map),
        )

        feats["tourney_level_tier"] = self.encoders.encode_tourney_level(df["tourney_level"])
        feats["round_ord"] = self.encoders.encode_round(df["round"])
        feats["surface_speed"] = self.encoders.encode_surface(df["surface"])

        feats = pd.get_dummies(feats.join(df[["tour"]]), drop_first=True)
        feats["fav_odds"] = np.where(df["B365W"] < df["B365L"], df["B365W"], df["B365L"])

        return feats 