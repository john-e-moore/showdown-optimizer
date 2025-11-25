from __future__ import annotations

"""
DraftKings-style offensive fantasy scoring.

This module implements a helper that takes a player-game dataframe and
computes DraftKings-like fantasy points for offensive positions.
"""

import pandas as pd

from . import config


def compute_dk_points_offense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DraftKings-style offensive fantasy points for each player-game row.

    The input dataframe is expected to already contain canonical stat columns
    as defined in config, e.g.:
        - pass_yards, pass_tds, interceptions
        - rush_yards, rush_tds
        - rec_yards, rec_tds, receptions

    A new column `dk_points` is added and the same dataframe is returned.
    Missing stat columns are treated as zeros.
    """
    # Ensure all stat columns exist so arithmetic is straightforward
    for col in [
        config.COL_PASS_YARDS,
        config.COL_PASS_TDS,
        config.COL_INTERCEPTIONS,
        config.COL_RUSH_YARDS,
        config.COL_RUSH_TDS,
        config.COL_REC_YARDS,
        config.COL_REC_TDS,
        config.COL_RECEPTIONS,
    ]:
        if col not in df.columns:
            df[col] = 0.0

    # Passing
    pass_yards = df[config.COL_PASS_YARDS].fillna(0.0)
    pass_tds = df[config.COL_PASS_TDS].fillna(0.0)
    interceptions = df[config.COL_INTERCEPTIONS].fillna(0.0)

    dk_pass = (
        0.04 * pass_yards
        + 4.0 * pass_tds
        - 1.0 * interceptions
        + 3.0 * (pass_yards >= 300).astype(float)
    )

    # Rushing
    rush_yards = df[config.COL_RUSH_YARDS].fillna(0.0)
    rush_tds = df[config.COL_RUSH_TDS].fillna(0.0)

    dk_rush = (
        0.1 * rush_yards
        + 6.0 * rush_tds
        + 3.0 * (rush_yards >= 100).astype(float)
    )

    # Receiving
    rec_yards = df[config.COL_REC_YARDS].fillna(0.0)
    rec_tds = df[config.COL_REC_TDS].fillna(0.0)
    receptions = df[config.COL_RECEPTIONS].fillna(0.0)

    dk_rec = (
        receptions
        + 0.1 * rec_yards
        + 6.0 * rec_tds
        + 3.0 * (rec_yards >= 100).astype(float)
    )

    df[config.COL_DK_POINTS] = dk_pass + dk_rush + dk_rec
    return df


