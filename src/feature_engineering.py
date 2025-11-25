from __future__ import annotations

"""
Feature engineering for player-game level data.

This module computes per-player fantasy point statistics and standardized
z-scores that are later used to build the pairwise training dataset.
"""

from typing import List

import numpy as np
import pandas as pd

from . import config


def _filter_offensive_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only offensive positions (and optional K) defined in config.
    """
    if config.COL_POSITION not in df.columns:
        return df
    return df[df[config.COL_POSITION].isin(config.OFFENSIVE_POSITIONS)]


def add_player_dk_stats(df: pd.DataFrame, min_games: int) -> pd.DataFrame:
    """
    Add per-player DK fantasy statistics and standardized z-scores.

    Expected input columns (canonical):
      - player_id, dk_points, season, week, game_id, team, opponent, position

    Steps:
      1. Filter to offensive positions.
      2. Compute per-player mean (mu_player) and std (sigma_player) of dk_points.
      3. Drop players with fewer than `min_games` appearances.
      4. Avoid sigma=0 by replacing with a small epsilon.
      5. Compute z = (dk_points - mu_player) / sigma_player.
    """
    required_cols: List[str] = [
        config.COL_PLAYER_ID,
        config.COL_DK_POINTS,
        config.COL_SEASON,
        config.COL_WEEK,
        config.COL_GAME_ID,
        config.COL_TEAM,
        config.COL_OPPONENT,
        config.COL_POSITION,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns {missing} in player-game dataframe. "
            "Ensure data_loading and fantasy_scoring have been applied."
        )

    df = _filter_offensive_players(df.copy())

    # Compute per-player aggregates
    grouped = df.groupby(config.COL_PLAYER_ID)[config.COL_DK_POINTS]
    mu = grouped.mean().rename(config.COL_MU_PLAYER)
    sigma = grouped.std(ddof=0).rename(config.COL_SIGMA_PLAYER)
    count = grouped.count().rename("games_played")

    stats = pd.concat([mu, sigma, count], axis=1).reset_index()

    # Filter by min_games
    stats = stats[stats["games_played"] >= min_games]

    # Avoid zero std; clamp to epsilon
    epsilon = 0.1
    stats[config.COL_SIGMA_PLAYER] = stats[config.COL_SIGMA_PLAYER].replace(
        0, epsilon
    ).fillna(epsilon)

    # Join back to main frame
    df = df.merge(
        stats[
            [
                config.COL_PLAYER_ID,
                config.COL_MU_PLAYER,
                config.COL_SIGMA_PLAYER,
            ]
        ],
        on=config.COL_PLAYER_ID,
        how="inner",
    )

    # Compute z-score
    df[config.COL_Z_SCORE] = (
        df[config.COL_DK_POINTS] - df[config.COL_MU_PLAYER]
    ) / df[config.COL_SIGMA_PLAYER]

    # Persist processed dataframe for reuse
    config.NFL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_PLAYER_GAMES_PARQUET, index=False)

    return df


