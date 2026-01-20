from __future__ import annotations

import pandas as pd
import math

def compute_elo_table(games_df):
    elo_df = pd.DataFrame(
        columns=["game_id", "season", "week", "game_date", "team", "opponent", "is_home", "elo_pre", "elo_post"]
    )
    same_cols = ["game_id", "season", "week", "game_date"]
    elo_df[same_cols] = games_df[same_cols]

    base_elo = 1500.0

    teams = pd.unique(pd.concat([games_df["home_team"], games_df["away_team"]]))
    elo = {team: base_elo for team in teams}

    games_df = games_df.sort_values(["season", "week", "game_date", "game_id"]).reset_index(drop=True)

    for g in games_df.itertuples(index=False):
        elo_sway = 0.5
        home = g.home_team
        away = g.away_team
        p_home = (1.0 / (1.0 + math.pow(10, ((elo.get(away) - elo.get(home)) / 400))))
        p_away = (1.0 / (1.0 + math.pow(10, ((elo.get(home) - elo.get(away)) / 400))))

        elo.update({home: elo.get(home) + elo_sway*(g.home_win - p_home)})
        elo.update({away: elo.get(away) + elo_sway*((1 - g.home_win) - p_away)})

    return elo_df.to_parquet("data/processed/elo.parquet", index=False)

    #TODO: Load parquet into file system and implement into features.py and implement elo balancing in between seasons

