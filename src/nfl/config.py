from __future__ import annotations

"""
Global configuration for the NFL Showdown correlation pipeline.

All filesystem paths are relative to the project root by default but can be
overridden as needed by editing this module.
"""

from pathlib import Path
from typing import Final, Dict, List

from ..shared import config_base


# -----------------------------------------------------------------------------
# Base paths
# -----------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = config_base.PROJECT_ROOT

# Treat this module as the NFL-specific configuration. Data and outputs are
# routed through sport-aware subdirectories:
#   - data/nfl/...
#   - outputs/nfl/...
DATA_DIR: Final[Path] = config_base.get_data_dir_for_sport("nfl")
NFL_RAW_DIR: Final[Path] = DATA_DIR / "nfl_raw"
NFL_PROCESSED_DIR: Final[Path] = DATA_DIR / "nfl_processed"
SABERSIM_DIR: Final[Path] = DATA_DIR / "sabersim"

MODELS_DIR: Final[Path] = config_base.MODELS_ROOT
OUTPUTS_DIR: Final[Path] = config_base.get_outputs_dir_for_sport("nfl")
CORR_OUTPUTS_DIR: Final[Path] = OUTPUTS_DIR / "correlations"
DIAGNOSTICS_DIR: Final[Path] = config_base.get_diagnostics_dir_for_sport("nfl")


# -----------------------------------------------------------------------------
# Data file locations
# -----------------------------------------------------------------------------

NFL_PLAYER_GAMES_PARQUET: Final[str] = str(NFL_RAW_DIR / "player_stats.parquet")
NFL_GAMES_PARQUET: Final[str] = str(NFL_RAW_DIR / "games.parquet")

# Processed datasets
PROCESSED_PLAYER_GAMES_PARQUET: Final[str] = str(
    NFL_PROCESSED_DIR / "player_games_with_z.parquet"
)

SABERSIM_CSV: Final[str] = str(
    SABERSIM_DIR / "NFL_2025-11-24-815pm_DK_SHOWDOWN_CAR-@-SF.csv"
)

OUTPUT_CORR_EXCEL: Final[str] = str(
    CORR_OUTPUTS_DIR / "showdown_corr_matrix.xlsx"
)

CORR_MODEL_PATH: Final[str] = str(MODELS_DIR / "corr_model.pkl")


# -----------------------------------------------------------------------------
# Modeling / filtering constants
# -----------------------------------------------------------------------------

MIN_PLAYER_GAMES: Final[int] = 8

# Enable or disable writing diagnostics snapshots.
ENABLE_DIAGNOSTICS: Final[bool] = True

# Seasons for model splits (inclusive ranges).
# Your current dataset (per diagnostics) only contains 2023–2024, so we
# allocate 2023 for training and 2024 for validation/test. Adjust as you
# add more historical seasons.
TRAIN_SEASONS: Final[range] = range(2023, 2024)  # 2023
VAL_SEASONS: Final[range] = range(2024, 2025)    # 2024
TEST_SEASONS: Final[range] = range(2024, 2025)   # 2024 (reuse for test)

# Offensive positions to include downstream (plus optional K)
OFFENSIVE_POSITIONS: Final[List[str]] = ["QB", "RB", "WR", "TE", "K", "DST"]


# -----------------------------------------------------------------------------
# Column name mappings
# -----------------------------------------------------------------------------

# Canonical internal names used throughout the pipeline
COL_GAME_ID: Final[str] = "game_id"
COL_PLAYER_ID: Final[str] = "player_id"
COL_SEASON: Final[str] = "season"
COL_WEEK: Final[str] = "week"
COL_TEAM: Final[str] = "team"
COL_OPPONENT: Final[str] = "opponent"
COL_POSITION: Final[str] = "position"
COL_HOME_TEAM: Final[str] = "home_team"
COL_AWAY_TEAM: Final[str] = "away_team"
COL_HOME_SCORE: Final[str] = "home_score"
COL_AWAY_SCORE: Final[str] = "away_score"
COL_SEASON_TYPE: Final[str] = "season_type"

# Offensive box score stats (canonical)
COL_PASS_YARDS: Final[str] = "pass_yards"
COL_PASS_TDS: Final[str] = "pass_tds"
COL_INTERCEPTIONS: Final[str] = "interceptions"
COL_RUSH_YARDS: Final[str] = "rush_yards"
COL_RUSH_TDS: Final[str] = "rush_tds"
COL_REC_YARDS: Final[str] = "rec_yards"
COL_REC_TDS: Final[str] = "rec_tds"
COL_RECEPTIONS: Final[str] = "receptions"

# Canonical fantasy points columns
COL_DK_POINTS: Final[str] = "dk_points"
COL_MU_PLAYER: Final[str] = "mu_player"
COL_SIGMA_PLAYER: Final[str] = "sigma_player"
COL_Z_SCORE: Final[str] = "z"


# Mappings from expected nflverse-style raw column names to canonical names.
# Adjust these as needed to match the user's local Parquet schema.
RAW_TO_CANONICAL_STATS: Final[Dict[str, str]] = {
    # Passing
    "passing_yards": COL_PASS_YARDS,
    "pass_yards": COL_PASS_YARDS,
    "passing_tds": COL_PASS_TDS,
    "pass_tds": COL_PASS_TDS,
    "interceptions": COL_INTERCEPTIONS,
    "int": COL_INTERCEPTIONS,
    # Rushing
    "rushing_yards": COL_RUSH_YARDS,
    "rush_yards": COL_RUSH_YARDS,
    "rushing_tds": COL_RUSH_TDS,
    "rush_tds": COL_RUSH_TDS,
    # Receiving
    "receiving_yards": COL_REC_YARDS,
    "rec_yards": COL_REC_YARDS,
    "receiving_tds": COL_REC_TDS,
    "rec_tds": COL_REC_TDS,
    "receptions": COL_RECEPTIONS,
    "targets": COL_RECEPTIONS,  # fallback if receptions not provided
}


"""
Simulation configuration
------------------------

These settings control the Monte Carlo simulator used to build a correlation
matrix directly from Sabersim projections.
"""

# Default method for building the correlation matrix from Sabersim projections.
# Options are:
#   - "simulation": Monte Carlo simulator (default)
#   - "ml": historical ML regression model (z-score product)
DEFAULT_CORR_METHOD: Final[str] = "simulation"

# Number of Monte Carlo simulations (games) to run per Showdown slate.
SIM_N_GAMES: Final[int] = 5000

# Random seed for the simulator (set to None to use nondeterministic seed).
SIM_RANDOM_SEED: Final[int | None] = 42

# Dirichlet concentration parameters that control how tightly player stat
# shares cluster around their projected shares. Higher values => lower
# variance around projections.
SIM_DIRICHLET_K_YARDS: Final[float] = 50.0
SIM_DIRICHLET_K_RECEPTIONS: Final[float] = 50.0
SIM_DIRICHLET_K_TDS: Final[float] = 20.0

# Small epsilon used when guarding against division by zero in share
# computations.
SIM_EPS: Final[float] = 1e-9


def ensure_directories() -> None:
    """
    Ensure that key directories used by the pipeline exist.

    This does not create raw-data directories (user is expected to place
    Parquet files there) but will create outputs and diagnostics dirs.
    """

    CORR_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)



