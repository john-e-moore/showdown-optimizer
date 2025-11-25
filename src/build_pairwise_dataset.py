from __future__ import annotations

"""
Construction of the pairwise training dataset for correlation modeling.

Each row in the output corresponds to an unordered pair of offensive players
who appeared in the same game. The target is y = z_A * z_B, where z_* are the
standardized fantasy point scores for that game.
"""

from typing import List

import numpy as np
import pandas as pd

from . import config


def _add_is_home_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean column indicating whether the player's team is at home.
    """
    if (
        config.COL_TEAM not in df.columns
        or config.COL_HOME_TEAM not in df.columns
        or config.COL_AWAY_TEAM not in df.columns
    ):
        return df

    is_home = np.where(
        df[config.COL_TEAM] == df[config.COL_HOME_TEAM],
        1,
        np.where(df[config.COL_TEAM] == df[config.COL_AWAY_TEAM], 0, np.nan),
    )
    df["is_home"] = is_home
    return df


def _add_game_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple game-level features: total points and point differential.
    """
    if (
        config.COL_HOME_SCORE in df.columns
        and config.COL_AWAY_SCORE in df.columns
    ):
        total_points = (
            df[config.COL_HOME_SCORE] + df[config.COL_AWAY_SCORE]
        )
        point_diff = (
            df[config.COL_HOME_SCORE] - df[config.COL_AWAY_SCORE]
        ).abs()
        df["total_points"] = total_points
        df["point_diff"] = point_diff
    return df


def _add_season_to_date_means(
    df: pd.DataFrame, stats: List[str]
) -> pd.DataFrame:
    """
    For each player, season, and stat column, compute the season-to-date mean
    using only prior games (no leakage).
    """
    df = df.sort_values(
        [config.COL_PLAYER_ID, config.COL_SEASON, config.COL_WEEK]
    ).copy()

    group_keys = [config.COL_PLAYER_ID, config.COL_SEASON]
    for stat in stats:
        if stat not in df.columns:
            continue
        grp = df.groupby(group_keys)[stat]
        cumsum = grp.cumsum().shift(1)
        counts = grp.cumcount()
        mean_prior = cumsum / counts.replace(0, np.nan)
        df[f"{stat}_mean_szn_to_date"] = mean_prior

    return df


def build_pairwise_dataset(
    player_games_df: pd.DataFrame, games_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the pairwise training dataset for the correlation regressor.

    The returned dataframe includes:
      - A_player_id, B_player_id
      - A_* player features, B_* player features
      - Game-level features (season, week, total_points, point_diff)
      - Target y = z_A * z_B

    Only offensive players (as defined in config.OFFENSIVE_POSITIONS) are used.
    """
    # Merge in game-level metadata
    games_cols = [
        config.COL_GAME_ID,
        config.COL_SEASON,
        config.COL_WEEK,
        config.COL_HOME_TEAM,
        config.COL_AWAY_TEAM,
        config.COL_HOME_SCORE,
        config.COL_AWAY_SCORE,
    ]
    games_meta = games_df[games_cols].copy()

    df = player_games_df.merge(
        games_meta,
        on=config.COL_GAME_ID,
        how="inner",
    )

    # Offensive players only
    if config.COL_POSITION in df.columns:
        df = df[df[config.COL_POSITION].isin(config.OFFENSIVE_POSITIONS)]

    # Add game-level features and home/away flag
    df = _add_game_level_features(df)
    df = _add_is_home_flag(df)

    # Season-to-date rolling features using prior games only
    rolling_stats = [
        config.COL_DK_POINTS,
        config.COL_PASS_YARDS,
        config.COL_RUSH_YARDS,
        config.COL_REC_YARDS,
        config.COL_RECEPTIONS,
    ]
    df = _add_season_to_date_means(df, rolling_stats)

    # Build unordered pairs within each game
    # Self-merge on game_id and then keep A_player_id < B_player_id
    pair_df = df.merge(
        df,
        on=config.COL_GAME_ID,
        suffixes=("_A", "_B"),
    )

    pair_df = pair_df[
        pair_df[f"{config.COL_PLAYER_ID}_A"]
        < pair_df[f"{config.COL_PLAYER_ID}_B"]
    ].copy()

    # Target: product of standardized scores
    z_A_col = f"{config.COL_Z_SCORE}_A"
    z_B_col = f"{config.COL_Z_SCORE}_B"
    if z_A_col not in pair_df.columns or z_B_col not in pair_df.columns:
        raise KeyError(
            "Expected z-score columns for both players in pairwise dataset. "
            "Did you run feature_engineering.add_player_dk_stats first?"
        )

    pair_df["y"] = pair_df[z_A_col] * pair_df[z_B_col]

    # Build a compact feature set
    output_cols: List[str] = [
        # identifiers
        f"{config.COL_PLAYER_ID}_A",
        f"{config.COL_PLAYER_ID}_B",
        config.COL_GAME_ID,
        f"{config.COL_TEAM}_A",
        f"{config.COL_TEAM}_B",
        f"{config.COL_POSITION}_A",
        f"{config.COL_POSITION}_B",
        "is_home_A",
        "is_home_B",
        # player DK stats
        f"{config.COL_MU_PLAYER}_A",
        f"{config.COL_MU_PLAYER}_B",
        f"{config.COL_SIGMA_PLAYER}_A",
        f"{config.COL_SIGMA_PLAYER}_B",
        f"{config.COL_DK_POINTS}_A",
        f"{config.COL_DK_POINTS}_B",
        # season-to-date means
        f"{config.COL_DK_POINTS}_mean_szn_to_date_A",
        f"{config.COL_DK_POINTS}_mean_szn_to_date_B",
        f"{config.COL_PASS_YARDS}_mean_szn_to_date_A",
        f"{config.COL_PASS_YARDS}_mean_szn_to_date_B",
        f"{config.COL_RUSH_YARDS}_mean_szn_to_date_A",
        f"{config.COL_RUSH_YARDS}_mean_szn_to_date_B",
        f"{config.COL_REC_YARDS}_mean_szn_to_date_A",
        f"{config.COL_REC_YARDS}_mean_szn_to_date_B",
        f"{config.COL_RECEPTIONS}_mean_szn_to_date_A",
        f"{config.COL_RECEPTIONS}_mean_szn_to_date_B",
        # game-level features (take from A side)
        f"{config.COL_SEASON}_A",
        f"{config.COL_WEEK}_A",
        "total_points_A",
        "point_diff_A",
        # target
        "y",
    ]

    # Map base column names to suffixed names for A/B players
    rename_map = {}
    for base_col in [
        config.COL_TEAM,
        config.COL_POSITION,
        config.COL_MU_PLAYER,
        config.COL_SIGMA_PLAYER,
        config.COL_DK_POINTS,
        config.COL_SEASON,
        config.COL_WEEK,
        "total_points",
        "point_diff",
        "is_home",
    ]:
        rename_map[f"{base_col}_A"] = (
            base_col if base_col in {config.COL_SEASON, config.COL_WEEK,
                                     "total_points", "point_diff"} else f"A_{base_col}"
        )
        rename_map[f"{base_col}_B"] = (
            f"B_{base_col}"
            if base_col not in {config.COL_SEASON, config.COL_WEEK,
                                "total_points", "point_diff"}
            else rename_map.get(f"{base_col}_B", None)  # ignored for game-level
        )

    # Season-to-date means
    for stat in rolling_stats:
        for side in ("A", "B"):
            col_name = f"{stat}_mean_szn_to_date_{side}"
            rename_map[col_name] = f"{side}_{stat}_mean_szn_to_date"

    # Apply renaming and select final columns
    pair_df = pair_df.rename(columns=rename_map)

    final_cols = [
        "A_" + config.COL_PLAYER_ID,
        "B_" + config.COL_PLAYER_ID,
        config.COL_GAME_ID,
        "A_" + config.COL_TEAM,
        "B_" + config.COL_TEAM,
        "A_" + config.COL_POSITION,
        "B_" + config.COL_POSITION,
        "A_is_home",
        "B_is_home",
        "A_" + config.COL_MU_PLAYER,
        "B_" + config.COL_MU_PLAYER,
        "A_" + config.COL_SIGMA_PLAYER,
        "B_" + config.COL_SIGMA_PLAYER,
        "A_" + config.COL_DK_POINTS,
        "B_" + config.COL_DK_POINTS,
        "A_" + config.COL_DK_POINTS + "_mean_szn_to_date",
        "B_" + config.COL_DK_POINTS + "_mean_szn_to_date",
        "A_" + config.COL_PASS_YARDS + "_mean_szn_to_date",
        "B_" + config.COL_PASS_YARDS + "_mean_szn_to_date",
        "A_" + config.COL_RUSH_YARDS + "_mean_szn_to_date",
        "B_" + config.COL_RUSH_YARDS + "_mean_szn_to_date",
        "A_" + config.COL_REC_YARDS + "_mean_szn_to_date",
        "B_" + config.COL_REC_YARDS + "_mean_szn_to_date",
        "A_" + config.COL_RECEPTIONS + "_mean_szn_to_date",
        "B_" + config.COL_RECEPTIONS + "_mean_szn_to_date",
        config.COL_SEASON,
        config.COL_WEEK,
        "total_points",
        "point_diff",
        "y",
    ]

    # Some stats may not exist; keep intersection
    final_cols = [c for c in final_cols if c in pair_df.columns]

    pairwise_df = pair_df[final_cols].copy()
    return pairwise_df


