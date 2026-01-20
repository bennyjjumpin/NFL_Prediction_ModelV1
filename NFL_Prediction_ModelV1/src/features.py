from __future__ import annotations

import numpy as np
import pandas as pd

from config import GAMES_PATH, MODEL_TABLE_PATH, PROCESSED_DIR, ROLLING_WINDOW


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each game into two team-rows (one for home team, one for away team)
    so we can compute rolling stats per team.
    """
    g = games.copy()

    home = g[["game_id", "season", "week", "game_date", "home_team", "home_score", "away_score"]].copy()
    home.rename(columns={"home_team": "team"}, inplace=True)
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]
    home["win"] = (home["points_for"] > home["points_against"]).astype(int)

    away = g[["game_id", "season", "week", "game_date", "away_team", "away_score", "home_score"]].copy()
    away.rename(columns={"away_team": "team"}, inplace=True)
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]
    away["win"] = (away["points_for"] > away["points_against"]).astype(int)

    long = pd.concat([home, away], ignore_index=True)
    long["ptdiff"] = long["points_for"] - long["points_against"]

    long = long.sort_values(["team", "season", "week", "game_date", "game_id"]).reset_index(drop=True)
    return long


def _rolling_features(team_games: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling stats that use only PRIOR games (shift by 1).
    """
    tg = team_games.copy()

    grp = tg.groupby("team", sort=False)

    tg["win_pct_lastN"] = (
        grp["win"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    tg["ptdiff_lastN"] = (
        grp["ptdiff"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    return tg[["game_id", "team", "win_pct_lastN", "ptdiff_lastN"]]


def build_model_table(games: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Returns one row per game with home/away rolling features and target label (home_win).
    """
    team_games = _team_game_rows(games)
    roll = _rolling_features(team_games, window)

    # Merge features for home and away teams onto each game row
    out = games.merge(
        roll.rename(columns={"team": "home_team", "win_pct_lastN": "home_win_pct_lastN", "ptdiff_lastN": "home_ptdiff_lastN"}),
        on=["game_id", "home_team"],
        how="left",
    ).merge(
        roll.rename(columns={"team": "away_team", "win_pct_lastN": "away_win_pct_lastN", "ptdiff_lastN": "away_ptdiff_lastN"}),
        on=["game_id", "away_team"],
        how="left",
    )

    # Simple matchup diffs
    out["win_pct_diff"] = out["home_win_pct_lastN"] - out["away_win_pct_lastN"]
    out["ptdiff_diff"] = out["home_ptdiff_lastN"] - out["away_ptdiff_lastN"]

    # Fill any early-season nulls (teams with no prior games)
    for c in ["home_win_pct_lastN", "away_win_pct_lastN"]:
        out[c] = out[c].fillna(0.5)
    for c in ["home_ptdiff_lastN", "away_ptdiff_lastN", "win_pct_diff", "ptdiff_diff"]:
        out[c] = out[c].fillna(0.0)

    return out


def run_feature_pipeline() -> pd.DataFrame:
    ensure_dirs()
    games = pd.read_parquet(GAMES_PATH)
    model_table = build_model_table(games, window=ROLLING_WINDOW)
    model_table.to_parquet(MODEL_TABLE_PATH, index=False)
    return model_table


if __name__ == "__main__":
    df = run_feature_pipeline()
    print(f"Wrote {len(df):,} rows to {MODEL_TABLE_PATH}")
