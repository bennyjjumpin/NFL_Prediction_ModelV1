from __future__ import annotations

import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from config import (
    MODEL_TABLE_PATH, MODEL_PATH, FEATURES_JSON_PATH, METRICS_JSON_PATH, MODELS_DIR
)


FEATURE_COLS = [
    "home_win_pct_lastN",
    "away_win_pct_lastN",
    "home_ptdiff_lastN",
    "away_ptdiff_lastN",
    "win_pct_diff",
    "ptdiff_diff",
]


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    V1 split: train on all but last 2 seasons; test on last 2 seasons.
    """
    seasons = sorted(df["season"].unique().tolist())
    if len(seasons) < 3:
        raise ValueError("Need at least 3 seasons for a reasonable time split.")
    test_seasons = seasons[-2:]
    train = df[~df["season"].isin(test_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()
    return train, test


def train_and_save() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(MODEL_TABLE_PATH)
    train_df, test_df = time_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["home_win"].astype(int)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["home_win"].astype(int)

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_log_loss": float(log_loss(y_test, proba)),
        "test_brier": float(brier_score_loss(y_test, proba)),
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "test_seasons": sorted(test_df["season"].unique().tolist()),
        "feature_cols": FEATURE_COLS,
    }

    joblib.dump(model, MODEL_PATH)
    FEATURES_JSON_PATH.write_text(json.dumps(FEATURE_COLS, indent=2))
    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    m = train_and_save()
    print(json.dumps(m, indent=2))
    print(f"Saved model to {MODEL_PATH}")
