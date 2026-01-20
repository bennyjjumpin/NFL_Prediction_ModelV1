from __future__ import annotations

import pandas as pd
import nflreadpy as nfl

from config import RAW_DIR, PROCESSED_DIR, GAMES_PATH, SEASONS, GAME_TYPE


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    """
    Loads schedule/results data and returns a pandas DataFrame.
    nflreadpy provides load_schedules() for game schedules/results. :contentReference[oaicite:1]{index=1}
    """
    schedules_pl = nfl.load_schedules(seasons)
    df = schedules_pl.to_pandas()  # nflreadpy returns Polars by default. :contentReference[oaicite:2]{index=2}
    return df


def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce to the columns we need and standardize names.
    Column names can differ slightly across datasets/versions, so this function is defensive.
    """
    # Common nflreadr/nflverse schedule columns include:
    # season, week, game_type, gameday, home_team, away_team, home_score, away_score, game_id, ...
    col_map = {
        "gameday": "game_date",
        "game_type": "game_type",
        "season": "season",
        "week": "week",
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "game_id": "game_id",
    }

    missing = [c for c in col_map.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in schedules data: {missing}")

    out = df[list(col_map.keys())].rename(columns=col_map).copy()

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")

    # Filter to regular season for V1
    out = out[out["game_type"] == GAME_TYPE].copy()

    # Keep only completed games for training table; app can still show future games if you want later.
    out = out.dropna(subset=["home_score", "away_score"])

    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)
    out["ptdiff"] = out["home_score"] - out["away_score"]

    # Sort for time-based features
    out = out.sort_values(["season", "week", "game_date", "game_id"]).reset_index(drop=True)
    return out


def build_games_parquet() -> pd.DataFrame:
    ensure_dirs()
    sched = load_schedules(SEASONS)
    games = clean_games(sched)
    games.to_parquet(GAMES_PATH, index=False)
    return games


if __name__ == "__main__":
    games = build_games_parquet()
    print(f"Wrote {len(games):,} games to {GAMES_PATH}")
