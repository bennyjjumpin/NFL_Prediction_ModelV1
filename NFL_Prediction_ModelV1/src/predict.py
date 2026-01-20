from __future__ import annotations

import json
import joblib
import pandas as pd

from config import MODEL_PATH, FEATURES_JSON_PATH, MODEL_TABLE_PATH


def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    feature_cols = json.loads(FEATURES_JSON_PATH.read_text())
    return model, feature_cols


def predict_game(season: int, week: int, home_team: str, away_team: str) -> dict:
    """
    V1: predict from the precomputed model_table row for that game.
    """
    model, feature_cols = load_model_and_features()
    df = pd.read_parquet(MODEL_TABLE_PATH)

    row = df[
        (df["season"] == season)
        & (df["week"] == week)
        & (df["home_team"] == home_team)
        & (df["away_team"] == away_team)
    ]

    if row.empty:
        raise ValueError("Game not found in model table (check season/week/teams).")

    X = row.iloc[[0]][feature_cols]
    p_home_win = float(model.predict_proba(X)[:, 1][0])

    return {
        "season": int(season),
        "week": int(week),
        "home_team": home_team,
        "away_team": away_team,
        "p_home_win": p_home_win,
        "features": row.iloc[0][feature_cols].to_dict(),
    }


if __name__ == "__main__":
    print(predict_game(2024, 1, "KC", "BAL"))
