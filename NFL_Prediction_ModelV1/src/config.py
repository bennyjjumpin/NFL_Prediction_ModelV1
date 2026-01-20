from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Modeling scope
SEASONS = list(range(2016, 2025 + 1))  # adjust
ROLLING_WINDOW = 5                     # last N games
GAME_TYPE = "REG"                      # regular season only (recommended for V1)

# Artifacts
GAMES_PATH = PROCESSED_DIR / "games.parquet"
MODEL_TABLE_PATH = PROCESSED_DIR / "model_table.parquet"
MODEL_PATH = MODELS_DIR / "logreg_homewin.joblib"
FEATURES_JSON_PATH = MODELS_DIR / "features.json"
METRICS_JSON_PATH = MODELS_DIR / "metrics.json"
