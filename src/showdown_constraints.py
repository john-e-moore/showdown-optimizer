from __future__ import annotations

"""
Custom DFS constraint configuration for the Showdown optimizer.

Edit this file to enable/disable or tweak rules without touching the core
optimizer logic.

All functions here return `ConstraintBuilder` callables that can be passed into
`optimize_showdown_lineups(..., constraint_builders=[...])`.
"""

from typing import Dict, List, Tuple

import pulp

from .lineup_optimizer import (
    CPT_SLOT,
    SLOTS,
    ConstraintBuilder,
    Player,
    PlayerPool,
    min_players_from_team,
)


# ---------------------------------------------------------------------------
# Simple configuration knobs you can edit for each slate
# ---------------------------------------------------------------------------

# Example: "must use at least N players from team TEAM_A".
# Set TEAM_A to something like "KC" or "SF", and MIN_PLAYERS_FROM_TEAM_A to
# a positive integer. Set MIN_PLAYERS_FROM_TEAM_A to None to disable.
TEAM_A: str | None = None
MIN_PLAYERS_FROM_TEAM_A: int | None = None

# If True, enforce:
#   If a QB is CPT, must use at least two WR/TE from the same team.
ENABLE_QB_CPT_TWO_PASS_CATCHERS: bool = True

# If True, enforce:
#   If we have an RB as CPT, forbid using the opposing DST.
ENABLE_RB_CPT_NO_OPPOSING_DST: bool = True


# ---------------------------------------------------------------------------
# Constraint builder implementations
# ---------------------------------------------------------------------------


def qb_cpt_requires_two_pass_catchers_same_team() -> ConstraintBuilder:
    """
    If a QB is CPT, require at least two WR/TE from the same team as that QB.

    For each team T:
        wrte_T >= 2 * qb_cpt_T
    where:
      - qb_cpt_T is the sum of CPT variables for QBs on team T (0 or 1)
      - wrte_T is the total count of WR/TE from team T across all slots
    """

    def builder(prob: pulp.LpProblem, x, player_pool: PlayerPool) -> None:
        teams = sorted({p.team for p in player_pool.players})
        for team in teams:
            qb_players: List[Player] = [
                p for p in player_pool.players
                if p.team == team and p.position == "QB"
            ]
            pass_catchers: List[Player] = [
                p for p in player_pool.players
                if p.team == team and p.position in {"WR", "TE"}
            ]
            if not qb_players or not pass_catchers:
                continue

            qb_cpt = pulp.lpSum(x[(p.player_id, CPT_SLOT)] for p in qb_players)
            wrte_any = pulp.lpSum(
                x[(p.player_id, slot)] for p in pass_catchers for slot in SLOTS
            )
            # If qb_cpt = 1, wrte_any >= 2; if qb_cpt = 0, no restriction.
            prob += wrte_any >= 2 * qb_cpt, f"qb_cpt_two_pass_catchers_{team}"

    return builder


def rb_cpt_forbid_opposing_dst() -> ConstraintBuilder:
    """
    If we have an RB as CPT, forbid using the opposing DST.

    For each ordered pair of distinct teams (team_rb, team_dst):
        rb_cpt_team_rb + dst_any_team_dst <= 1
    """

    def builder(prob: pulp.LpProblem, x, player_pool: PlayerPool) -> None:
        teams = sorted({p.team for p in player_pool.players})
        for team_rb in teams:
            for team_dst in teams:
                if team_rb == team_dst:
                    continue

                rb_players: List[Player] = [
                    p for p in player_pool.players
                    if p.team == team_rb and p.position == "RB"
                ]
                dst_players: List[Player] = [
                    p for p in player_pool.players
                    if p.team == team_dst and p.position == "DST"
                ]
                if not rb_players or not dst_players:
                    continue

                rb_cpt = pulp.lpSum(x[(p.player_id, CPT_SLOT)] for p in rb_players)
                dst_any = pulp.lpSum(
                    x[(p.player_id, slot)] for p in dst_players for slot in SLOTS
                )
                prob += (
                    rb_cpt + dst_any <= 1,
                    f"rb_cpt_no_opp_dst_{team_rb}_vs_{team_dst}",
                )

    return builder


def build_custom_constraints() -> List[ConstraintBuilder]:
    """
    Return the list of custom constraints to apply for this slate.

    Edit the module-level config variables above (TEAM_A, MIN_PLAYERS_FROM_TEAM_A,
    ENABLE_QB_CPT_TWO_PASS_CATCHERS, ENABLE_RB_CPT_NO_OPPOSING_DST) to adjust.
    """
    constraints: List[ConstraintBuilder] = []

    if TEAM_A and MIN_PLAYERS_FROM_TEAM_A is not None and MIN_PLAYERS_FROM_TEAM_A > 0:
        constraints.append(min_players_from_team(TEAM_A, MIN_PLAYERS_FROM_TEAM_A))

    if ENABLE_QB_CPT_TWO_PASS_CATCHERS:
        constraints.append(qb_cpt_requires_two_pass_catchers_same_team())

    if ENABLE_RB_CPT_NO_OPPOSING_DST:
        constraints.append(rb_cpt_forbid_opposing_dst())

    return constraints



