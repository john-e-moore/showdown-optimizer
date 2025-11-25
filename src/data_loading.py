from __future__ import annotations

"""
Data loading utilities for nflverse-style Parquet inputs.

This module provides thin wrappers around pandas to:
- Load player-game statistics
- Load games/schedule data
- Apply basic filtering (regular season only, 2005+)
- Normalize a few key column names using config-level conventions
"""

from typing import Iterable

import pandas as pd

from . import config


def _rename_stats_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw nflverse stat columns to canonical names defined in config.

    Only columns present in the dataframe are renamed.
    """
    to_rename = {
        raw: canonical
        for raw, canonical in config.RAW_TO_CANONICAL_STATS.items()
        if raw in df.columns
    }
    if to_rename:
        df = df.rename(columns=to_rename)
    return df


def _filter_regular_season_and_years(
    df: pd.DataFrame, *, season_col: str, season_type_col: str | None = None
) -> pd.DataFrame:
    """
    Filter to regular-season games and seasons >= 2005.

    If a season_type column exists, it is assumed to contain a flag like
    \"REG\"/\"POST\"; only regular-season rows are kept. If not present, no
    season-type filtering is applied.
    """
    if season_type_col and season_type_col in df.columns:
        df = df[df[season_type_col] == "REG"]

    if season_col in df.columns:
        df = df[df[season_col] >= 2005]

    return df


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """
    Raise a helpful error if key columns are missing.

    This keeps failures early and debuggable if local Parquet schemas differ.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns {missing} in dataframe. "
            "Please adjust your Parquet schema or update config.py mappings."
        )


def load_player_game_stats(path: str) -> pd.DataFrame:
    """
    Load player-game statistics from a Parquet file.

    The returned dataframe is filtered to:
      - Regular season games (if season_type column exists)
      - Seasons >= 2005

    It also:
      - Renames raw stat columns to canonical names using config.RAW_TO_CANONICAL_STATS
      - Ensures presence of core identifier columns such as game_id, player_id,
        season, week, team, position (using config column constants)
    """
    df = pd.read_parquet(path)

    # Normalize stat columns to canonical names
    df = _rename_stats_to_canonical(df)

    # Filter by season type and year
    df = _filter_regular_season_and_years(
        df,
        season_col=config.COL_SEASON,
        season_type_col=config.COL_SEASON_TYPE,
    )

    # Ensure minimal required identifier columns are present
    required_cols = [
        config.COL_GAME_ID,
        config.COL_PLAYER_ID,
        config.COL_SEASON,
        config.COL_WEEK,
        config.COL_TEAM,
        config.COL_POSITION,
    ]
    _ensure_columns(df, required_cols)

    return df


def load_games(path: str) -> pd.DataFrame:
    """
    Load games/schedule data from a Parquet file.

    The returned dataframe is filtered to:
      - Regular season games (if season_type column exists)
      - Seasons >= 2005

    It also ensures presence of key identifiers:
      - game_id, season, week, home/away teams, and scores
    """
    df = pd.read_parquet(path)

    df = _filter_regular_season_and_years(
        df,
        season_col=config.COL_SEASON,
        season_type_col=config.COL_SEASON_TYPE,
    )

    required_cols = [
        config.COL_GAME_ID,
        config.COL_SEASON,
        config.COL_WEEK,
        config.COL_HOME_TEAM,
        config.COL_AWAY_TEAM,
        config.COL_HOME_SCORE,
        config.COL_AWAY_SCORE,
    ]
    _ensure_columns(df, required_cols)

    return df


