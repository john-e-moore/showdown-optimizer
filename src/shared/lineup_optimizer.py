from __future__ import annotations

"""
Sabersim CSV loader and thin adaptor around the shared Showdown optimizer core.

This module is sport-agnostic and can be used by both NFL and NBA:

  - Parses Sabersim-style Showdown projections CSVs.
  - Constructs a `PlayerPool` of shared `Player` objects.
  - Re-exports shared optimizer types/functions for convenience.

All MILP model construction and optimization logic lives in
`src/shared/optimizer_core.py`.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .optimizer_core import (
    Player,
    PlayerPool,
    Lineup,
    ConstraintBuilder,
    optimize_showdown_lineups,
)


# -----------------------------------------------------------------------------
# Sabersim CSV loading
# -----------------------------------------------------------------------------


SABERSIM_NAME_COL = "Name"
SABERSIM_TEAM_COL = "Team"
SABERSIM_POS_COL = "Pos"
SABERSIM_SALARY_COL = "Salary"
SABERSIM_DK_PROJ_COL = "My Proj"
SABERSIM_DK_STD_COL = "dk_std"
SABERSIM_IS_CPT_ELIGIBLE_COL = "is_cpt_eligible"
SABERSIM_IS_FLEX_ELIGIBLE_COL = "is_flex_eligible"


def _load_raw_sabersim_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a Sabersim Showdown CSV and reduce to one flex-style row per player.

    Heuristic:
      - For each (Name, Team) pair, keep the row with the LOWER salary, which
        corresponds to FLEX/UTIL in DraftKings Showdown.
      - Do NOT filter by position; optimizer may want DST, K, etc.
    """
    df = pd.read_csv(path)

    required = [SABERSIM_NAME_COL, SABERSIM_TEAM_COL, SABERSIM_SALARY_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Sabersim CSV missing required columns {missing}. "
            "Please check the file schema."
        )

    df = (
        df.sort_values(by=[SABERSIM_NAME_COL, SABERSIM_TEAM_COL, SABERSIM_SALARY_COL])
        .groupby([SABERSIM_NAME_COL, SABERSIM_TEAM_COL], as_index=False)
        .first()
    )
    return df


def load_players_from_sabersim(path: str | Path) -> PlayerPool:
    """
    Load Sabersim projections from CSV and build a PlayerPool.

    Expected (or inferred) columns:
      - Name, Team, Pos, Salary, My Proj
      - Optional: dk_std, is_cpt_eligible, is_flex_eligible

    Player IDs are synthetic but stable within a single run.
    """
    df = _load_raw_sabersim_csv(path)

    if SABERSIM_DK_PROJ_COL not in df.columns:
        raise KeyError(
            f"Sabersim CSV missing DK projection column '{SABERSIM_DK_PROJ_COL}'."
        )

    # Drop players with zero (or negative) projection; they cannot contribute
    # positively to an optimal lineup and only slow down the solver.
    df = df[df[SABERSIM_DK_PROJ_COL] > 0]

    # Ensure basic columns exist even if missing in source.
    if SABERSIM_POS_COL not in df.columns:
        df[SABERSIM_POS_COL] = ""

    players: List[Player] = []
    for _, row in df.iterrows():
        name = str(row[SABERSIM_NAME_COL])
        team = str(row[SABERSIM_TEAM_COL])
        position = str(row[SABERSIM_POS_COL])

        # Synthetic, human-readable ID based on name + team.
        player_id = f"{name}|{team}"

        dk_salary = int(row[SABERSIM_SALARY_COL])
        dk_proj = float(row[SABERSIM_DK_PROJ_COL])

        if SABERSIM_DK_STD_COL in df.columns:
            val = row[SABERSIM_DK_STD_COL]
            dk_std: Optional[float] = float(val) if pd.notna(val) else None
        else:
            dk_std = None

        if SABERSIM_IS_CPT_ELIGIBLE_COL in df.columns:
            is_cpt_eligible = bool(row[SABERSIM_IS_CPT_ELIGIBLE_COL])
        else:
            is_cpt_eligible = True

        if SABERSIM_IS_FLEX_ELIGIBLE_COL in df.columns:
            is_flex_eligible = bool(row[SABERSIM_IS_FLEX_ELIGIBLE_COL])
        else:
            is_flex_eligible = True

        players.append(
            Player(
                player_id=player_id,
                name=name,
                team=team,
                position=position,
                dk_salary=dk_salary,
                dk_proj=dk_proj,
                dk_std=dk_std,
                is_cpt_eligible=is_cpt_eligible,
                is_flex_eligible=is_flex_eligible,
            )
        )

    return PlayerPool(players)


__all__ = [
    "Player",
    "PlayerPool",
    "Lineup",
    "ConstraintBuilder",
    "load_players_from_sabersim",
    "optimize_showdown_lineups",
]


