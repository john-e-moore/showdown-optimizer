from __future__ import annotations

"""
Download historical NFL data using nfl_data_py and write Parquet files
compatible with the NFL correlation pipeline.

This script:
  - Downloads weekly player-game stats for regular seasons (2005+).
  - Downloads games/schedule data for the same seasons.
  - Normalizes key columns to align with the canonical schema defined in
    src.nfl.config and expected by src.nfl.data_loading.
  - Writes Parquet files to config.NFL_PLAYER_GAMES_PARQUET and
    config.NFL_GAMES_PARQUET.

Usage (from project root):

    python -m src.nfl.download_nfl_data --start-season 2005 --end-season 2024
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

    # nfl_data_py versions differ on the name of the season-type column.
    # Prefer "season_type" if present, otherwise fall back to "game_type".
    if "season_type" in schedules.columns:
        season_type_col = "season_type"
    elif "game_type" in schedules.columns:
        season_type_col = "game_type"
    else:
        season_type_col = None

    # Filter to regular season only when possible
    if season_type_col is not None:
        schedules = schedules[schedules[season_type_col] == "REG"]

    required_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    _validate_columns(schedules, required_cols, context="schedules")

    games = schedules[required_cols].copy()

    # Add/normalize season_type for downstream code
    if season_type_col is not None:
        games[config.COL_SEASON_TYPE] = schedules[season_type_col].values
    else:
        # If no explicit season-type column is present, assume regular season
        games[config.COL_SEASON_TYPE] = "REG"

    # Align column names exactly with config expectations.
    rename_map = {
        "game_id": config.COL_GAME_ID,
        "season": config.COL_SEASON,
        "week": config.COL_WEEK,
        "home_team": config.COL_HOME_TEAM,
        "away_team": config.COL_AWAY_TEAM,
        "home_score": config.COL_HOME_SCORE,
        "away_score": config.COL_AWAY_SCORE,
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

    # Determine team column name used by nfl_data_py (recent_team vs team)
    team_col = "recent_team" if "recent_team" in weekly.columns else "team"
    if team_col not in weekly.columns:
        raise KeyError(
            "Weekly data missing team column 'recent_team' or 'team'. "
            "Check nfl_data_py version or update download_player_game_stats."
        )

    # Import schedules to get game_ids and filter to regular season
    schedules = nfl.import_schedules(seasons)
    if "season_type" in schedules.columns:
        season_type_col = "season_type"
    elif "game_type" in schedules.columns:
        season_type_col = "game_type"
    else:
        season_type_col = None

    if season_type_col is not None:
        schedules = schedules[schedules[season_type_col] == "REG"]

    sched_required = ["game_id", "season", "week", "home_team", "away_team"]
    _validate_columns(schedules, sched_required, context="schedules for weekly data")

    # Expand schedules to team-level rows (one per team per game)
    home_map = schedules[["game_id", "season", "week", "home_team"]].rename(
        columns={"home_team": "team"}
    )
    away_map = schedules[["game_id", "season", "week", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    team_games = pd.concat([home_map, away_map], ignore_index=True)

    # Attach game_id to weekly data via season/week/team
    weekly = weekly.merge(
        team_games,
        left_on=["season", "week", team_col],
        right_on=["season", "week", "team"],
        how="inner",
    )

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

    # Build canonical player-game dataframe
    player_games = pd.DataFrame()
    player_games[config.COL_GAME_ID] = weekly["game_id"]
    player_games[config.COL_PLAYER_ID] = weekly["player_id"]
    player_games[config.COL_SEASON] = weekly["season"]
    player_games[config.COL_WEEK] = weekly["week"]
    player_games[config.COL_TEAM] = weekly[team_col]
    player_games[config.COL_OPPONENT] = weekly["opponent_team"]
    player_games[config.COL_POSITION] = weekly["position"]
    player_games[config.COL_SEASON_TYPE] = "REG"

    # Include all potential raw stat columns that data_loading knows how to map
    for raw_col in config.RAW_TO_CANONICAL_STATS.keys():
        if raw_col in weekly.columns:
            player_games[raw_col] = weekly[raw_col]

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
            print(
                f"Player stats file already exists at {player_path} "
                "(use --overwrite to replace)."
            )
        if games_path.exists():
            print(
                f"Games file already exists at {games_path} "
                "(use --overwrite to replace)."
            )
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



