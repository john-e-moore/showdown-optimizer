from __future__ import annotations

"""
NBA-specific hard constraints for the Showdown optimizer.

These are applied only by the NBA optimizer entrypoint.
"""

from typing import List

import pulp

from ..shared.optimizer_core import (
    CPT_SLOT,
    FLEX_SLOTS,
    SLOTS,
    ConstraintBuilder,
    Player,
    PlayerPool,
)


def cpt_min_minutes(min_minutes: float = 24.0) -> ConstraintBuilder:
    """
    Enforce: CPT projected minutes >= min_minutes.

    Implementation:
      - For each player with proj_min present and < min_minutes, force them to be
        ineligible for CPT (x[player, CPT] == 0).
    """

    def builder(prob: pulp.LpProblem, x, player_pool: PlayerPool) -> None:
        for p in player_pool.players:
            if p.proj_min is None:
                continue
            if float(p.proj_min) < float(min_minutes):
                prob += x[(p.player_id, CPT_SLOT)] == 0, f"nba_cpt_min_{p.player_id}"

    return builder


def max_low_proj_utils(max_count: int = 1, threshold: float = 8.0) -> ConstraintBuilder:
    """
    Enforce: At most `max_count` UTIL/FLEX players with base projection <= threshold.

    Note: This applies only to FLEX_SLOTS (UTIL), not CPT, per user preference.
    """

    def builder(prob: pulp.LpProblem, x, player_pool: PlayerPool) -> None:
        low_proj_players: List[Player] = [
            p for p in player_pool.players if float(p.dk_proj) <= float(threshold)
        ]
        if not low_proj_players:
            return

        low_util_count = pulp.lpSum(
            x[(p.player_id, slot)] for p in low_proj_players for slot in FLEX_SLOTS
        )
        prob += low_util_count <= int(max_count), "nba_max_low_proj_utils"

    return builder


def min_lineup_salary(min_salary: int = 48_500) -> ConstraintBuilder:
    """
    Enforce: total lineup salary >= min_salary (with CPT salary at 1.5x rounded).
    """

    def builder(prob: pulp.LpProblem, x, player_pool: PlayerPool) -> None:
        salary_terms = []
        for p in player_pool.players:
            salary_terms.append(p.cpt_salary * x[(p.player_id, CPT_SLOT)])
            for slot in FLEX_SLOTS:
                salary_terms.append(p.dk_salary * x[(p.player_id, slot)])

        prob += pulp.lpSum(salary_terms) >= int(min_salary), "nba_min_lineup_salary"

    return builder


def build_nba_constraints() -> List[ConstraintBuilder]:
    """
    Hard-enforced NBA Showdown constraints.
    """

    return [
        cpt_min_minutes(24.0),
        max_low_proj_utils(max_count=1, threshold=8.0),
        min_lineup_salary(48_500),
    ]


__all__ = [
    "cpt_min_minutes",
    "max_low_proj_utils",
    "min_lineup_salary",
    "build_nba_constraints",
]


