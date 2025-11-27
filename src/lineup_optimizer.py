from __future__ import annotations

"""
MILP-based DraftKings NFL Showdown (Captain Mode) lineup optimizer.

This module defines:
  - Core domain models: Player, PlayerPool, Lineup
  - CSV loader for Sabersim-style Showdown projections -> PlayerPool
  - Helpers to build a single-lineup MILP model with PuLP
  - Constraint hooks for custom DFS rules
  - Functions to solve for one lineup or the top N lineups

The optimizer currently maximizes mean projected DK points under standard
DraftKings Showdown roster rules:
  - 1 CPT (1.5x salary, 1.5x points)
  - 5 FLEX (normal salary, normal points)
  - 6 distinct players
  - Salary cap (configurable)
"""

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import time

import pandas as pd
import pulp


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Player:
    player_id: str
    name: str
    team: str
    position: str
    dk_salary: int
    dk_proj: float
    dk_std: Optional[float] = None
    is_cpt_eligible: bool = True
    is_flex_eligible: bool = True

    @property
    def cpt_salary(self) -> int:
        """
        DraftKings CPT slot salary: 1.5x FLEX salary, rounded to nearest int.
        """
        return int(round(1.5 * self.dk_salary))


class PlayerPool:
    """
    Thin wrapper around a list of Player objects with convenience helpers.
    """

    def __init__(self, players: Sequence[Player]) -> None:
        self._players: List[Player] = list(players)
        # Index by player_id for O(1) lookup.
        self._by_id: Dict[str, Player] = {p.player_id: p for p in self._players}

    @property
    def players(self) -> List[Player]:
        return list(self._players)

    def by_team(self, team: str) -> List[Player]:
        return [p for p in self._players if p.team == team]

    def by_position(self, position: str) -> List[Player]:
        return [p for p in self._players if p.position == position]

    def get(self, player_id: str) -> Player:
        return self._by_id[player_id]


@dataclass
class Lineup:
    cpt: Player
    flex: List[Player]

    def __post_init__(self) -> None:
        if len(self.flex) != 5:
            raise ValueError(f"Lineup must have exactly 5 FLEX players, got {len(self.flex)}")
        # Enforce 6 distinct players.
        all_ids = [self.cpt.player_id] + [p.player_id for p in self.flex]
        if len(set(all_ids)) != 6:
            raise ValueError("Lineup must contain 6 distinct players.")

    def salary(self) -> int:
        """
        Total DK salary for the lineup using CPT 1.5x salary (rounded).
        """
        cpt_salary = self.cpt.cpt_salary
        flex_salary = sum(p.dk_salary for p in self.flex)
        return cpt_salary + flex_salary

    def projection(self) -> float:
        """
        Total projected DK fantasy points for the lineup using CPT 1.5x points.
        """
        cpt_points = 1.5 * self.cpt.dk_proj
        flex_points = sum(p.dk_proj for p in self.flex)
        return cpt_points + flex_points

    def as_tuple_ids(self) -> Tuple[str, ...]:
        """
        Player IDs in a stable order: CPT first, then FLEX in list order.
        """
        return (self.cpt.player_id,) + tuple(p.player_id for p in self.flex)


# ---------------------------------------------------------------------------
# Sabersim CSV loading
# ---------------------------------------------------------------------------


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
    Load a Sabersim Showdown CSV and reduce to one FLEX-equivalent row per player.

    Heuristic:
      - For each (Name, Team) pair, keep the row with the LOWER salary, which
        corresponds to FLEX in DraftKings Showdown.
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
    for idx, row in df.iterrows():
        name = str(row[SABERSIM_NAME_COL])
        team = str(row[SABERSIM_TEAM_COL])
        position = str(row[SABERSIM_POS_COL])

        # Synthetic, human-readable ID based on name + team.
        player_id = f"{name}|{team}"

        dk_salary = int(row[SABERSIM_SALARY_COL])
        dk_proj = float(row[SABERSIM_DK_PROJ_COL])

        dk_std: Optional[float]
        if SABERSIM_DK_STD_COL in df.columns:
            val = row[SABERSIM_DK_STD_COL]
            dk_std = float(val) if pd.notna(val) else None
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


# ---------------------------------------------------------------------------
# MILP model construction
# ---------------------------------------------------------------------------


SLOTS: Tuple[str, ...] = ("CPT", "F1", "F2", "F3", "F4", "F5")
CPT_SLOT: str = "CPT"
FLEX_SLOTS: Tuple[str, ...] = SLOTS[1:]

VarKey = Tuple[str, str]  # (player_id, slot)
VarDict = Dict[VarKey, pulp.LpVariable]


def build_showdown_model(
    player_pool: PlayerPool,
    salary_cap: int,
) -> Tuple[pulp.LpProblem, VarDict]:
    """
    Create an empty MILP model and binary decision variables for CPT/FLEX slots.

    Base constraints and objective are added by separate helper functions.
    """
    prob = pulp.LpProblem("dk_showdown_lineup", sense=pulp.LpMaximize)

    x: VarDict = {}
    for player in player_pool.players:
        for slot in SLOTS:
            var_name = f"x_{player.player_id}_{slot}"
            x[(player.player_id, slot)] = pulp.LpVariable(var_name, cat="Binary")

    # Salary cap is passed here for API symmetry but applied in a separate helper.
    _ = salary_cap
    return prob, x


def add_single_cpt_constraint(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
) -> None:
    prob += (
        pulp.lpSum(x[(p.player_id, CPT_SLOT)] for p in player_pool.players) == 1,
        "single_cpt",
    )


def add_flex_count_constraint(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
) -> None:
    prob += (
        pulp.lpSum(
            x[(p.player_id, slot)] for p in player_pool.players for slot in FLEX_SLOTS
        )
        == 5,
        "flex_count_5",
    )


def add_unique_player_constraint(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
) -> None:
    for player in player_pool.players:
        prob += (
            pulp.lpSum(x[(player.player_id, slot)] for slot in SLOTS) <= 1,
            f"unique_player_{player.player_id}",
        )


def add_salary_cap_constraint(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
    salary_cap: int,
) -> None:
    salary_expr = []
    for p in player_pool.players:
        # CPT salary uses 1.5x with rounding, matching Lineup.salary().
        cpt_salary = p.cpt_salary
        salary_expr.append(cpt_salary * x[(p.player_id, CPT_SLOT)])
        for slot in FLEX_SLOTS:
            salary_expr.append(p.dk_salary * x[(p.player_id, slot)])

    prob += (
        pulp.lpSum(salary_expr) <= salary_cap,
        "salary_cap",
    )


def add_eligibility_constraints(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
) -> None:
    for p in player_pool.players:
        if not p.is_cpt_eligible:
            prob += x[(p.player_id, CPT_SLOT)] == 0, f"no_cpt_{p.player_id}"
        if not p.is_flex_eligible:
            for slot in FLEX_SLOTS:
                prob += (
                    x[(p.player_id, slot)] == 0,
                    f"no_flex_{p.player_id}_{slot}",
                )


def add_min_one_per_team_constraint(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
) -> None:
    """
    Ensure each team represented in the player pool has at least one player
    in the lineup (either CPT or FLEX).
    """
    teams = sorted({p.team for p in player_pool.players})
    for team in teams:
        team_players = [p for p in player_pool.players if p.team == team]
        if not team_players:
            continue
        team_count = pulp.lpSum(
            x[(p.player_id, slot)] for p in team_players for slot in SLOTS
        )
        prob += team_count >= 1, f"min_one_from_team_{team}"


def set_mean_projection_objective(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
) -> None:
    """
    Maximize mean projected DK points with CPT weighted 1.5x.
    """
    terms = []
    for p in player_pool.players:
        terms.append(1.5 * p.dk_proj * x[(p.player_id, CPT_SLOT)])
        for slot in FLEX_SLOTS:
            terms.append(p.dk_proj * x[(p.player_id, slot)])

    prob += pulp.lpSum(terms), "max_mean_projection"


def add_base_constraints(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
    salary_cap: int,
) -> None:
    add_single_cpt_constraint(prob, x, player_pool)
    add_flex_count_constraint(prob, x, player_pool)
    add_unique_player_constraint(prob, x, player_pool)
    add_salary_cap_constraint(prob, x, player_pool, salary_cap)
    add_eligibility_constraints(prob, x, player_pool)
    add_min_one_per_team_constraint(prob, x, player_pool)


# ---------------------------------------------------------------------------
# Custom DFS constraint hooks
# ---------------------------------------------------------------------------


ConstraintBuilder = Callable[[pulp.LpProblem, VarDict, PlayerPool], None]


def min_players_from_team(team: str, k: int) -> ConstraintBuilder:
    """
    Require at least k players from a given team across all slots.
    """

    def builder(prob: pulp.LpProblem, x: VarDict, player_pool: PlayerPool) -> None:
        team_players = [p for p in player_pool.players if p.team == team]
        if not team_players:
            return
        team_count = pulp.lpSum(
            x[(p.player_id, slot)] for p in team_players for slot in SLOTS
        )
        prob += team_count >= k, f"min_players_{team}_{k}"

    return builder


def max_players_from_team(team: str, k: int) -> ConstraintBuilder:
    """
    Require at most k players from a given team across all slots.
    """

    def builder(prob: pulp.LpProblem, x: VarDict, player_pool: PlayerPool) -> None:
        team_players = [p for p in player_pool.players if p.team == team]
        if not team_players:
            return
        team_count = pulp.lpSum(
            x[(p.player_id, slot)] for p in team_players for slot in SLOTS
        )
        prob += team_count <= k, f"max_players_{team}_{k}"

    return builder


def mutually_exclusive_groups(
    group1_pred: Callable[[Player], bool],
    group2_pred: Callable[[Player], bool],
) -> ConstraintBuilder:
    """
    Prohibit any lineup that contains at least one player from group 1 AND
    at least one player from group 2.

    Example usage:
        mutually_exclusive_groups(
            lambda p: p.team == \"KC\" and p.position == \"RB\",
            lambda p: p.team == \"SF\" and p.position == \"DST\",
        )
    """

    def builder(prob: pulp.LpProblem, x: VarDict, player_pool: PlayerPool) -> None:
        group1_players = [p for p in player_pool.players if group1_pred(p)]
        group2_players = [p for p in player_pool.players if group2_pred(p)]
        if not group1_players or not group2_players:
            return

        g1 = pulp.lpSum(
            x[(p.player_id, slot)] for p in group1_players for slot in SLOTS
        )
        g2 = pulp.lpSum(
            x[(p.player_id, slot)] for p in group2_players for slot in SLOTS
        )
        prob += g1 + g2 <= 1, "mutually_exclusive_groups"

    return builder


def if_qb_cpt_then_no_dst(team_qb: str, team_dst: str) -> ConstraintBuilder:
    """
    If a QB from team_qb is CPT, prohibit any DST from team_dst in any slot.
    Implemented as:
        qb_cpt + dst_any <= 1
    """

    def builder(prob: pulp.LpProblem, x: VarDict, player_pool: PlayerPool) -> None:
        qb_players = [
            p for p in player_pool.players if p.team == team_qb and p.position == "QB"
        ]
        dst_players = [
            p for p in player_pool.players if p.team == team_dst and p.position == "DST"
        ]
        if not qb_players or not dst_players:
            return

        qb_cpt = pulp.lpSum(x[(p.player_id, CPT_SLOT)] for p in qb_players)
        dst_any = pulp.lpSum(
            x[(p.player_id, slot)] for p in dst_players for slot in SLOTS
        )
        prob += qb_cpt + dst_any <= 1, "if_qb_cpt_then_no_dst"

    return builder


# ---------------------------------------------------------------------------
# Solving helpers
# ---------------------------------------------------------------------------


def _extract_lineup_from_solution(
    player_pool: PlayerPool,
    x: VarDict,
) -> Lineup:
    """
    Build a Lineup object from solved decision variables.
    """
    cpt_player: Optional[Player] = None
    flex_players: List[Player] = []

    # Preserve player_pool order for stable FLEX ordering.
    players_by_id: Dict[str, Player] = {p.player_id: p for p in player_pool.players}

    for p in player_pool.players:
        # CPT slot
        val_cpt = x[(p.player_id, CPT_SLOT)].value()
        if val_cpt is not None and val_cpt >= 0.5:
            if cpt_player is not None and cpt_player.player_id != p.player_id:
                raise RuntimeError("Multiple CPT players selected in solution.")
            cpt_player = p

        # FLEX slots
        for slot in FLEX_SLOTS:
            val = x[(p.player_id, slot)].value()
            if val is not None and val >= 0.5:
                flex_players.append(players_by_id[p.player_id])

    if cpt_player is None:
        raise RuntimeError("No CPT player selected in solution.")
    if len(flex_players) != 5:
        raise RuntimeError(f"Expected 5 FLEX players, got {len(flex_players)}.")

    return Lineup(cpt=cpt_player, flex=flex_players)


def solve_single_lineup(
    player_pool: PlayerPool,
    salary_cap: int,
    constraint_builders: Optional[Sequence[ConstraintBuilder]] = None,
) -> Optional[Lineup]:
    """
    Build and solve a single-lineup MILP, returning the optimal Lineup.

    Returns None if the model is infeasible or no optimal solution is found.
    """
    prob, x = build_showdown_model(player_pool, salary_cap)
    add_base_constraints(prob, x, player_pool, salary_cap)

    if constraint_builders:
        for builder in constraint_builders:
            builder(prob, x, player_pool)

    set_mean_projection_objective(prob, x, player_pool)

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status_str = pulp.LpStatus.get(prob.status, "Unknown")
    if status_str != "Optimal":
        return None

    return _extract_lineup_from_solution(player_pool, x)


def _build_showdown_model_with_constraints(
    player_pool: PlayerPool,
    salary_cap: int,
    constraint_builders: Optional[Sequence[ConstraintBuilder]] = None,
) -> Tuple[pulp.LpProblem, VarDict]:
    """
    Helper to build a fresh MILP model with base + custom constraints applied.
    """
    prob, x = build_showdown_model(player_pool, salary_cap)
    add_base_constraints(prob, x, player_pool, salary_cap)

    if constraint_builders:
        for builder in constraint_builders:
            builder(prob, x, player_pool)

    return prob, x


def _add_projection_cap_constraint(
    prob: pulp.LpProblem,
    x: VarDict,
    player_pool: PlayerPool,
    max_projection: float,
) -> None:
    """
    Constrain total projected DK points (with CPT 1.5x) to be at most max_projection.

    The projection expression is aligned with Lineup.projection() and
    set_mean_projection_objective().
    """
    terms = []
    for p in player_pool.players:
        terms.append(1.5 * p.dk_proj * x[(p.player_id, CPT_SLOT)])
        for slot in FLEX_SLOTS:
            terms.append(p.dk_proj * x[(p.player_id, slot)])

    proj_expr = pulp.lpSum(terms)
    prob += proj_expr <= max_projection, "projection_cap"


def optimize_showdown_lineups(
    projections_path_pattern: str,
    num_lineups: int,
    salary_cap: int = 50_000,
    constraint_builders: Optional[Sequence[ConstraintBuilder]] = None,
    chunk_size: Optional[int] = None,
    projection_eps: float = 1e-4,
) -> List[Lineup]:
    """
    Generate up to num_lineups optimal lineups from Sabersim projections.

    The function:
      1) Resolves projections_path_pattern to a single CSV path.
      2) Builds a PlayerPool.
      3) Builds a MILP model with base + custom constraints.
      4) Iteratively solves and extracts lineups, adding a no-duplicate
         constraint after each solution so that at least one player changes.
    """
    matches = sorted(glob(projections_path_pattern))
    if not matches:
        raise FileNotFoundError(
            f"No Sabersim CSVs found matching pattern: {projections_path_pattern!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Expected exactly one Sabersim CSV for optimizer, "
            f"but found {len(matches)}: {matches}"
        )

    csv_path = matches[0]
    player_pool = load_players_from_sabersim(csv_path)

    # If chunking is disabled, fall back to legacy behavior: single model with
    # a growing set of no-duplicate constraints.
    if not chunk_size or chunk_size <= 0:
        prob, x = _build_showdown_model_with_constraints(
            player_pool, salary_cap, constraint_builders
        )
        set_mean_projection_objective(prob, x, player_pool)

        solver = pulp.PULP_CBC_CMD(msg=False)
        lineups: List[Lineup] = []

        for k in range(num_lineups):
            prob.solve(solver)
            status_str = pulp.LpStatus.get(prob.status, "Unknown")
            if status_str != "Optimal":
                break

            lineup = _extract_lineup_from_solution(player_pool, x)
            lineups.append(lineup)

            # Add a no-duplicate constraint so that the next solution differs
            # by at least one player (CPT or FLEX).
            used_player_ids = lineup.as_tuple_ids()
            no_dup_expr = pulp.lpSum(
                x[(pid, slot)] for pid in used_player_ids for slot in SLOTS
            )
            # 6 distinct players in a valid lineup; force at most 5 of them to
            # appear together next time.
            prob += no_dup_expr <= 5, f"no_duplicate_lineup_{k}"

        return lineups

    # Chunked behavior: repeatedly build fresh models with an upper bound on
    # allowed projection, and extract up to chunk_size unique lineups per model.
    solver = pulp.PULP_CBC_CMD(msg=False)
    all_lineups: List[Lineup] = []
    remaining = num_lineups
    prev_min_proj: Optional[float] = None
    chunk_index = 0

    while remaining > 0:
        current_chunk_size = min(remaining, chunk_size)
        chunk_start = time.perf_counter()
        before_count = len(all_lineups)

        prob, x = _build_showdown_model_with_constraints(
            player_pool, salary_cap, constraint_builders
        )

        if prev_min_proj is not None:
            _add_projection_cap_constraint(
                prob, x, player_pool, max_projection=prev_min_proj - projection_eps
            )

        set_mean_projection_objective(prob, x, player_pool)

        chunk_min_proj: Optional[float] = None

        for j in range(current_chunk_size):
            prob.solve(solver)
            status_str = pulp.LpStatus.get(prob.status, "Unknown")
            if status_str != "Optimal":
                # No more feasible/optimal solutions within this chunk.
                break

            lineup = _extract_lineup_from_solution(player_pool, x)
            lineup_proj = lineup.projection()
            all_lineups.append(lineup)
            remaining -= 1

            if chunk_min_proj is None or lineup_proj < chunk_min_proj:
                chunk_min_proj = lineup_proj

            # Add a no-duplicate constraint within this chunk so that the next
            # solution differs by at least one player (CPT or FLEX).
            used_player_ids = lineup.as_tuple_ids()
            no_dup_expr = pulp.lpSum(
                x[(pid, slot)] for pid in used_player_ids for slot in SLOTS
            )
            prob += (
                no_dup_expr <= 5,
                f"no_duplicate_lineup_chunk{chunk_index}_{j}",
            )

            if remaining <= 0:
                break

        if chunk_min_proj is None:
            # This chunk could not find any feasible lineups; stop overall.
            chunk_elapsed = time.perf_counter() - chunk_start
            print(
                f"Finished chunk {chunk_index + 1}: 0 lineups "
                f"(no feasible solutions), time={chunk_elapsed:.2f}s"
            )
            break

        prev_min_proj = chunk_min_proj
        chunk_elapsed = time.perf_counter() - chunk_start
        produced = len(all_lineups) - before_count
        print(
            f"Finished chunk {chunk_index + 1}: {produced} lineups, "
            f"time={chunk_elapsed:.2f}s"
        )
        chunk_index += 1

    return all_lineups



