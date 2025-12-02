from __future__ import annotations

"""
Sabersim projections loading utilities for NBA Showdown.

This module mirrors `src.build_corr_matrix_from_projections` but:
  - Lives under the NBA namespace.
  - Does not apply NFL-specific position filters.
"""

from pathlib import Path

import pandas as pd

from . import config


SABERSIM_NAME_COL = "Name"
SABERSIM_TEAM_COL = "Team"
SABERSIM_POS_COL = "Pos"
SABERSIM_SALARY_COL = "Salary"
SABERSIM_DK_PROJ_COL = "My Proj"


def load_sabersim_projections(path: str | Path) -> pd.DataFrame:
    """
    Load Sabersim NBA Showdown projections and drop Captain (CPT) rows,
    keeping FLEX-equivalent rows only.

    Heuristic:
      - For each (Name, Team) pair, keep the row with the LOWER salary, which
        corresponds to FLEX in DraftKings Showdown.
      - Optionally filter to offensive positions using config.OFFENSIVE_POSITIONS
        when that set is non-empty.
    """
    df = pd.read_csv(path)

    # Basic sanity checks
    for col in [SABERSIM_NAME_COL, SABERSIM_TEAM_COL, SABERSIM_SALARY_COL]:
        if col not in df.columns:
            raise KeyError(
                f"Sabersim CSV missing required column '{col}'. "
                "Please check the file schema."
            )

    # Keep lowest-salary row per player-team as FLEX
    df = (
        df.sort_values(by=[SABERSIM_NAME_COL, SABERSIM_TEAM_COL, SABERSIM_SALARY_COL])
        .groupby([SABERSIM_NAME_COL, SABERSIM_TEAM_COL], as_index=False)
        .first()
    )

    # Optional filter to offensive positions
    if SABERSIM_POS_COL in df.columns and config.OFFENSIVE_POSITIONS:
        df = df[df[SABERSIM_POS_COL].isin(config.OFFENSIVE_POSITIONS)]

    return df



