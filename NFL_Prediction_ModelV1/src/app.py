import json
import joblib
import pandas as pd
import streamlit as st

from config import MODEL_PATH, FEATURES_JSON_PATH, MODEL_TABLE_PATH, METRICS_JSON_PATH


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_tables():
    df = pd.read_parquet(MODEL_TABLE_PATH)
    features = json.loads(FEATURES_JSON_PATH.read_text())
    metrics = json.loads(METRICS_JSON_PATH.read_text())
    return df, features, metrics


st.title("NFL V1 Predictor (Home Win Probability)")

df, feature_cols, metrics = load_tables()
model = load_model()

st.caption(f"Test seasons: {metrics['test_seasons']} | Rolling window: {len([c for c in feature_cols if 'lastN' in c])} (features include diffs)")

season = st.selectbox("Season", sorted(df["season"].unique()), index=len(sorted(df["season"].unique())) - 1)
weeks = sorted(df.loc[df["season"] == season, "week"].unique())
week = st.selectbox("Week", weeks)

games = df[(df["season"] == season) & (df["week"] == week)][["home_team", "away_team"]].copy()
games["matchup"] = games["away_team"] + " @ " + games["home_team"]
matchup = st.selectbox("Matchup", games["matchup"].tolist())

away_team, home_team = matchup.split(" @ ")

row = df[
    (df["season"] == season)
    & (df["week"] == week)
    & (df["home_team"] == home_team)
    & (df["away_team"] == away_team)
].iloc[[0]]

X = row[feature_cols]
p_home = float(model.predict_proba(X)[:, 1][0])

st.metric(label=f"Percentage({home_team} wins)", value=f"{p_home:.1%}")

with st.expander("Features used (V1)"):
    st.dataframe(row[feature_cols].T.rename(columns={row.index[0]: "value"}))
