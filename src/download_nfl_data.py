from __future__ import annotations

"""
Download historical NFL data using nfl_data_py and write Parquet files
compatible with the correlation pipeline.

This script:
  - Downloads weekly player-game stats for regular seasons (2005+).
  - Downloads games/schedule data for the same seasons.
  - Normalizes key columns to align with the canonical schema defined in
    src.config and expected by src.data_loading.
  - Writes Parquet files to config.NFL_PLAYER_GAMES_PARQUET and
    config.NFL_GAMES_PARQUET.

Usage (from project root):

    python -m src.download_nfl_data --start-season 2005 --end-season 2024
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import nfl_data_py as nfl
import pandas as pd

from . import config


def _validate_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns {missing} in {context}. "
            "Check nfl_data_py version or update the downloader's mappings."
        )


def download_games(seasons: List[int]) -> pd.DataFrame:
    """
    Download games/schedule data using nfl_data_py for the given seasons.

    Returns a dataframe with one row per game, filtered to regular season only
    and normalized to the canonical columns expected by the pipeline.
    """
    schedules = nfl.import_schedules(seasons)

    # Filter to regular season only
    if "season_type" in schedules.columns:
        schedules = schedules[schedules["season_type"] == "REG"]

    required_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "season_type",
    ]
    _validate_columns(schedules, required_cols, context="schedules")

    games = schedules[required_cols].copy()

    # Align column names exactly with config expectations (already matching
    # in most nfl_data_py versions, but kept explicit for clarity).
    rename_map = {
        "game_id": config.COL_GAME_ID,
        "season": config.COL_SEASON,
        "week": config.COL_WEEK,
        "home_team": config.COL_HOME_TEAM,
        "away_team": config.COL_AWAY_TEAM,
        "home_score": config.COL_HOME_SCORE,
        "away_score": config.COL_AWAY_SCORE,
        "season_type": config.COL_SEASON_TYPE,
    }
    games = games.rename(columns=rename_map)

    return games


def download_player_game_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Download weekly player-game statistics using nfl_data_py for the given seasons.

    Returns a dataframe with one row per player per game, filtered to regular
    season only and with columns normalized to align with the pipeline's
    expectations.
    """
    weekly = nfl.import_weekly_data(seasons, downcast=True)

    # Join with schedules to attach season_type and filter to regular season
    schedules = nfl.import_schedules(seasons)
    if "season_type" in schedules.columns:
        schedules = schedules[schedules["season_type"] == "REG"]

    # Only keep join columns we need
    sched_cols = [
        "game_id",
        "season",
        "week",
        "season_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    schedules = schedules[sched_cols].copy()

    weekly = weekly.merge(schedules, on=["game_id", "season", "week"], how="inner")

    # Determine team column name used by nfl_data_py (recent_team vs team)
    team_col = "recent_team" if "recent_team" in weekly.columns else "team"

    required_weekly_cols = [
        "game_id",
        "player_id",
        "season",
        "week",
        team_col,
        "opponent_team",
        "position",
    ]
    _validate_columns(weekly, required_weekly_cols, context="weekly player data")

    # Core identifiers
    weekly[config.COL_GAME_ID] = weekly["game_id"]
    weekly[config.COL_PLAYER_ID] = weekly["player_id"]
    weekly[config.COL_SEASON] = weekly["season"]
    weekly[config.COL_WEEK] = weekly["week"]
    weekly[config.COL_TEAM] = weekly[team_col]
    weekly[config.COL_OPPONENT] = weekly["opponent_team"]
    weekly[config.COL_POSITION] = weekly["position"]

    # Attach season_type and game-level info
    weekly[config.COL_SEASON_TYPE] = weekly["season_type"]
    weekly[config.COL_HOME_TEAM] = weekly["home_team"]
    weekly[config.COL_AWAY_TEAM] = weekly["away_team"]
    weekly[config.COL_HOME_SCORE] = weekly["home_score"]
    weekly[config.COL_AWAY_SCORE] = weekly["away_score"]

    # Keep key identifier and stat columns (leave stat names as provided by
    # nfl_data_py; they will be mapped to canonical names later via
    # config.RAW_TO_CANONICAL_STATS in data_loading._rename_stats_to_canonical).
    base_cols = [
        config.COL_GAME_ID,
        config.COL_PLAYER_ID,
        config.COL_SEASON,
        config.COL_WEEK,
        config.COL_TEAM,
        config.COL_OPPONENT,
        config.COL_POSITION,
        config.COL_SEASON_TYPE,
        config.COL_HOME_TEAM,
        config.COL_AWAY_TEAM,
        config.COL_HOME_SCORE,
        config.COL_AWAY_SCORE,
    ]

    # Include all potential raw stat columns that data_loading knows how to map
    stat_cols = [
        col
        for col in config.RAW_TO_CANONICAL_STATS.keys()
        if col in weekly.columns
    ]

    cols_to_keep = base_cols + stat_cols
    player_games = weekly[cols_to_keep].copy()

    return player_games


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NFL weekly player stats and games using nfl_data_py."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2005,
        help="First season to include (default: 2005).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=datetime.now().year,
        help="Last season to include, inclusive (default: current year).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Parquet files if they exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    seasons = list(range(args.start_season, args.end_season + 1))
    print(f"Downloading data for seasons: {seasons}")

    raw_dir = Path(config.NFL_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    player_path = Path(config.NFL_PLAYER_GAMES_PARQUET)
    games_path = Path(config.NFL_GAMES_PARQUET)

    if not args.overwrite:
        if player_path.exists():
            print(f"Player stats file already exists at {player_path} (use --overwrite to replace).")
        if games_path.exists():
            print(f"Games file already exists at {games_path} (use --overwrite to replace).")
        if player_path.exists() and games_path.exists():
            return

    print("Downloading games/schedule data...")
    games_df = download_games(seasons)
    print(f"Writing games Parquet to {games_path}...")
    games_df.to_parquet(games_path, index=False)

    print("Downloading weekly player-game stats...")
    player_games_df = download_player_game_stats(seasons)
    print(f"Writing player-game stats Parquet to {player_path}...")
    player_games_df.to_parquet(player_path, index=False)

    print("Download complete.")


if __name__ == "__main__":
    main()


