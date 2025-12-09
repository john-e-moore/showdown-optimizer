from __future__ import annotations

"""Quota-balanced random field lineup builder.

This module implements a simple version of the quota-balanced field builder
sketched in the `prompts/quota-balanced-builder.md` document.

It is intentionally sport-agnostic and works with the same inputs that
`top1pct_core` already loads from the optimizer and correlation workbooks.

The public entrypoint is :func:`build_quota_balanced_field`, which returns a
DataFrame of CPT + 5 FLEX lineups representing a simulated contest field whose
player-level CPT/FLEX usage approximately matches the projected ownership.

The initial implementation uses a single "entrant type" with fixed
hyperparameters. The API and internals are structured so that additional
entrant types or tuning knobs can be added later without changing callers.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


LINEUP_COLS = ["cpt"] + [f"flex{i}" for i in range(1, 6)]


def _hamilton_rounding(proportions: np.ndarray, total_slots: int) -> np.ndarray:
    """Hamilton / largest-remainder rounding for integer quotas.

    Given non-negative proportions that (approximately) sum to 1, allocate
    ``total_slots`` integer slots whose sum is exactly ``total_slots``.
    """

    if total_slots <= 0 or proportions.size == 0:
        return np.zeros_like(proportions, dtype=int)

    # Ideal real-valued targets.
    target = proportions * float(total_slots)
    base = np.floor(target).astype(int)

    remaining = int(total_slots - int(base.sum()))
    if remaining < 0:
        # Numerical-quirk safeguard: never un-assign slots.
        remaining = 0

    if remaining > 0:
        remainders = target - base
        order = np.argsort(-remainders)
        for idx in order[:remaining]:
            base[idx] += 1

    return base


@dataclass
class FieldBuilderConfig:
    """Hyperparameters for the quota-balanced field builder.

    The defaults are intentionally conservative and can be tuned later based
    on comparisons to historical contests.
    """

    # How strongly to chase projection/value.
    alpha: float = 1.0
    # How strongly to follow remaining ownership quotas.
    beta: float = 1.0
    # How strongly to favor correlated stacks.
    gamma: float = 0.5
    # How strongly to prefer using most of the salary cap.
    delta: float = 0.5

    # DraftKings Showdown salary cap and minimum spend.
    salary_cap: int = 50_000
    min_salary: int = 46_000


def _compute_player_quantas(
    field_size: int,
    ownership_df: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Compute CPT and FLEX quotas per player from projected ownership.

    Args:
        field_size: Total number of lineups in the simulated field.
        ownership_df: DataFrame with at least columns
            ``["player", "cpt_ownership", "flex_ownership"]`` in percent.

    Returns:
        cpt_quota, flex_quota: dicts mapping player name -> integer quota.
    """

    required = {"player", "cpt_ownership", "flex_ownership"}
    missing = required.difference(ownership_df.columns)
    if missing:
        raise KeyError(
            "Ownership sheet missing required columns " f"{sorted(missing)}."
        )
    own = ownership_df.copy()
    own["player"] = own["player"].astype(str)
    # Aggregate by player in case of duplicates.
    own = (
        own.groupby("player")[["cpt_ownership", "flex_ownership"]]  # type: ignore
        .sum()
        .reset_index()
    )

    num_players = len(own)
    if num_players == 0:
        return {}, {}

    # Convert ownership percentages to fractions.
    p_cpt_raw = own["cpt_ownership"].to_numpy(dtype=float) / 100.0
    p_flex_raw = own["flex_ownership"].to_numpy(dtype=float) / 100.0

    total_cpt = float(p_cpt_raw.sum())
    total_flex = float(p_flex_raw.sum())

    if total_cpt > 0:
        p_cpt = p_cpt_raw / total_cpt
    else:
        # Degenerate case: fall back to uniform CPT fractions.
        p_cpt = np.full(num_players, 1.0 / num_players, dtype=float)

    if total_flex > 0:
        p_flex = p_flex_raw / total_flex
    else:
        # Degenerate case: fall back to uniform FLEX fractions.
        p_flex = np.full(num_players, 1.0 / num_players, dtype=float)

    total_cpt_slots = int(field_size)
    total_flex_slots = int(5 * field_size)

    base_cpt = _hamilton_rounding(p_cpt, total_cpt_slots)
    base_flex = _hamilton_rounding(p_flex, total_flex_slots)

    cpt_quota: Dict[str, int] = {}
    flex_quota: Dict[str, int] = {}

    for idx, row in own.iterrows():
        name = str(row["player"])
        cpt_quota[name] = int(base_cpt[idx]) if idx < len(base_cpt) else 0
        flex_quota[name] = int(base_flex[idx]) if idx < len(base_flex) else 0

    return cpt_quota, flex_quota


def _build_projection_universe(
    sabersim_proj_df: pd.DataFrame,
    lineups_proj_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, float], Dict[str, float], Dict[str, str], Dict[str, str]]:
    """Build a simple player universe with projection/salary/pos/team.

    This mirrors a subset of the logic in ``top1pct_core._build_player_universe``
    but stays minimal: it only needs to support the field builder.

    Returns:
        names: list of player names.
        proj: name -> mean DK points.
        salary: name -> DK salary (flex-style, non-CPT).
        pos: name -> position string.
        team: name -> team string.
    """

    sabersim = sabersim_proj_df.copy()
    if "Name" not in sabersim.columns or "My Proj" not in sabersim.columns:
        raise KeyError(
            "Correlation Sabersim projections sheet missing 'Name' or 'My Proj'."
        )
    sabersim["Name"] = sabersim["Name"].astype(str)

    # Handle duplicate Name entries (e.g. CPT vs non-CPT rows from a raw
    # Sabersim Showdown export). For the field builder we also want a single
    # flex-style row per player, so we keep the lower-projection row and drop
    # the higher-projection CPT rows.
    if sabersim["Name"].duplicated().any():
        before = len(sabersim)
        sabersim = (
            sabersim.sort_values(by=["Name", "My Proj"], ascending=[True, True])
            .groupby("Name", as_index=False)
            .first()
        )
        after = len(sabersim)
        dropped = before - after
        if dropped > 0:
            print(
                "Warning: Sabersim_Projections sheet contained duplicate player "
                "names for field builder; dropped "
                f"{dropped} higher-projection CPT rows, keeping one flex-style "
                "row per player."
            )

    sab_idx = sabersim.set_index("Name")

    lineups_proj = lineups_proj_df.copy()
    required_lp = ["Name", "Team", "Salary", "My Proj"]
    missing_lp = [c for c in required_lp if c not in lineups_proj.columns]
    if missing_lp:
        raise KeyError(
            "Lineups projections sheet missing required columns " f"{missing_lp}."
        )
    lineups_proj["Name"] = lineups_proj["Name"].astype(str)
    lineups_proj["Team"] = lineups_proj["Team"].astype(str)
    lineups_proj = (
        lineups_proj.sort_values(
            by=["Name", "Team", "Salary"], ascending=[True, True, True]
        )
        .groupby(["Name", "Team"], as_index=False)
        .first()
    )
    lp_idx = lineups_proj.set_index("Name")

    # Union of all players present in either projections source.
    universe: List[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            universe.append(name)

    for name in sab_idx.index:
        _add(str(name))
    for name in lp_idx.index:
        _add(str(name))

    proj: Dict[str, float] = {}
    salary: Dict[str, float] = {}
    pos: Dict[str, str] = {}
    team: Dict[str, str] = {}

    for name in universe:
        if name in sab_idx.index:
            proj[name] = float(sab_idx.at[name, "My Proj"])
        elif name in lp_idx.index:
            proj[name] = float(lp_idx.at[name, "My Proj"])
        else:
            proj[name] = 0.0

        if name in lp_idx.index:
            salary[name] = float(lp_idx.at[name, "Salary"])
            team[name] = str(lp_idx.at[name, "Team"])
        else:
            salary[name] = 0.0
            team[name] = ""

        if "Pos" in sab_idx.columns and name in sab_idx.index:
            pos[name] = str(sab_idx.at[name, "Pos"])
        elif "Pos" in lp_idx.columns and name in lp_idx.index:
            pos[name] = str(lp_idx.at[name, "Pos"])
        else:
            pos[name] = ""

    return universe, proj, salary, pos, team


def _build_corr_lookup(corr_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Create a simple (player_i, player_j) -> corr lookup from matrix df."""

    names = [str(x) for x in corr_df.index.tolist()]
    name_to_idx = {n: i for i, n in enumerate(names)}
    arr = corr_df.to_numpy(dtype=float)

    lookup: Dict[Tuple[str, str], float] = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            lookup[(ni, nj)] = float(arr[i, j])
    return lookup


def _estimate_value(proj: float, salary: float) -> float:
    if salary <= 0:
        return proj
    return proj / salary


def _soft_cpt_rule(pos: str) -> float:
    """Very simple CPT position heuristic.

    Boost QBs and primary skill positions slightly; penalize K/DST.
    """

    if pos in {"QB"}:
        return 1.2
    if pos in {"RB", "WR", "TE"}:
        return 1.1
    if pos in {"K", "DST"}:
        return 0.8
    return 1.0


def _rule_penalty(
    name: str,
    team: str,
    pos: str,
    lineup: List[str],
    team_by_name: Dict[str, str],
    pos_by_name: Dict[str, str],
) -> float:
    """Soft lineup rule penalty/bias.

    Currently a very light-touch implementation that mostly aims to avoid
    obviously unreasonable constructions. This can be made more sophisticated
    later as needed.
    """

    # Discourage extreme duplication of same-team skill players beyond 4.
    teams_in_lineup = [team_by_name.get(p, "") for p in lineup]
    same_team_count = sum(1 for t in teams_in_lineup if t == team)
    if same_team_count >= 4:
        return 0.7

    # Slight penalty for leaving huge salary amounts when almost done.
    return 1.0


def _sample_cpt(
    rng: np.random.Generator,
    universe: Iterable[str],
    remaining_cpt: Dict[str, int],
    proj: Dict[str, float],
    salary: Dict[str, float],
    pos: Dict[str, str],
    cfg: FieldBuilderConfig,
) -> str:
    epsilon = 1e-6
    names: List[str] = []
    weights: List[float] = []

    for name in universe:
        quota_left = remaining_cpt.get(name, 0)
        if quota_left <= 0:
            continue
        v = _estimate_value(proj.get(name, 0.0) * 1.5, salary.get(name, 0.0) * 1.5)
        value_term = max(v, 0.0) ** cfg.alpha
        quota_term = (quota_left + epsilon) ** cfg.beta
        rule_term = _soft_cpt_rule(pos.get(name, ""))
        w = value_term * quota_term * (rule_term ** cfg.gamma)
        if w <= 0:
            continue
        names.append(name)
        weights.append(w)

    if not names:
        raise RuntimeError("No eligible CPT candidates with remaining quota.")

    weights_arr = np.array(weights, dtype=float)
    probs = weights_arr / weights_arr.sum()
    idx = rng.choice(len(names), p=probs)
    return names[idx]


def _sample_flex_sequence(
    rng: np.random.Generator,
    universe: Iterable[str],
    remaining_flex: Dict[str, int],
    proj: Dict[str, float],
    salary: Dict[str, float],
    pos: Dict[str, str],
    team: Dict[str, str],
    corr_lookup: Dict[Tuple[str, str], float],
    cfg: FieldBuilderConfig,
    current_lineup: List[str],
) -> List[str]:
    """Fill the 5 FLEX slots sequentially for a single lineup.

    Returns the names of the 5 FLEX players.
    """

    flex_players: List[str] = []

    def corr_with_lineup(candidate: str) -> float:
        total = 0.0
        for p in current_lineup:
            total += corr_lookup.get((candidate, p), 0.0)
        for p in flex_players:
            total += corr_lookup.get((candidate, p), 0.0)
        return total

    for _slot in range(5):
        names: List[str] = []
        weights: List[float] = []

        for name in universe:
            if name in current_lineup or name in flex_players:
                continue
            quota_left = remaining_flex.get(name, 0)
            if quota_left <= 0:
                continue

            v = _estimate_value(proj.get(name, 0.0), salary.get(name, 0.0))
            value_term = max(v, 0.0) ** cfg.alpha
            quota_term = (quota_left + 1e-6) ** cfg.beta
            corr_term = np.exp(cfg.gamma * corr_with_lineup(name))
            rule_term = _rule_penalty(
                name,
                team.get(name, ""),
                pos.get(name, ""),
                current_lineup + flex_players,
                team,
                pos,
            )
            w = value_term * quota_term * corr_term * rule_term
            if w <= 0 or not np.isfinite(w):
                continue
            names.append(name)
            weights.append(w)

        if not names:
            # Fallback: ignore quotas and choose based purely on value and
            # correlation among legal players.
            for name in universe:
                if name in current_lineup or name in flex_players:
                    continue
                v = _estimate_value(proj.get(name, 0.0), salary.get(name, 0.0))
                value_term = max(v, 0.0)
                corr_term = np.exp(0.5 * corr_with_lineup(name))
                w = value_term * corr_term
                if w <= 0 or not np.isfinite(w):
                    continue
                names.append(name)
                weights.append(w)

        if not names:
            raise RuntimeError("Could not find any FLEX candidates for lineup.")

        weights_arr = np.array(weights, dtype=float)
        probs = weights_arr / weights_arr.sum()
        idx = rng.choice(len(names), p=probs)
        chosen = names[idx]
        flex_players.append(chosen)
        remaining_flex[chosen] = remaining_flex.get(chosen, 0) - 1

    return flex_players


def build_quota_balanced_field(
    *,
    field_size: int,
    ownership_df: pd.DataFrame,
    sabersim_proj_df: pd.DataFrame,
    lineups_proj_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    random_seed: int | None = None,
    config: FieldBuilderConfig | None = None,
) -> pd.DataFrame:
    """Generate a quota-balanced field of CPT+FLEX lineups.

    Args:
        field_size: Number of lineups (entrants) to generate.
        ownership_df: Optimizer ownership sheet.
        sabersim_proj_df: Correlation workbook Sabersim projections sheet.
        lineups_proj_df: Lineup optimizer projections sheet.
        corr_df: Player correlation matrix from the correlation workbook.
        random_seed: Optional RNG seed.
        config: Optional :class:`FieldBuilderConfig` to override defaults.

    Returns:
        DataFrame with columns ``["cpt", "flex1", ..., "flex5"]`` containing
        player names. Additional metadata columns can be added later if needed.
    """

    if field_size <= 0:
        raise ValueError("field_size must be positive.")

    cfg = config or FieldBuilderConfig()

    cpt_quota, flex_quota = _compute_player_quantas(field_size, ownership_df)
    universe, proj, salary, pos, team = _build_projection_universe(
        sabersim_proj_df=sabersim_proj_df,
        lineups_proj_df=lineups_proj_df,
    )
    corr_lookup = _build_corr_lookup(corr_df)

    remaining_cpt = dict(cpt_quota)
    remaining_flex = dict(flex_quota)

    rng = np.random.default_rng(random_seed)

    lineups: List[Dict[str, str]] = []

    for _ in range(field_size):
        # Sample CPT.
        cpt = _sample_cpt(
            rng=rng,
            universe=universe,
            remaining_cpt=remaining_cpt,
            proj=proj,
            salary=salary,
            pos=pos,
            cfg=cfg,
        )
        remaining_cpt[cpt] = remaining_cpt.get(cpt, 0) - 1

        # Sample FLEX sequence.
        flex_players = _sample_flex_sequence(
            rng=rng,
            universe=universe,
            remaining_flex=remaining_flex,
            proj=proj,
            salary=salary,
            pos=pos,
            team=team,
            corr_lookup=corr_lookup,
            cfg=cfg,
            current_lineup=[cpt],
        )

        lineup_record: Dict[str, str] = {"cpt": cpt}
        for i, name in enumerate(flex_players, start=1):
            lineup_record[f"flex{i}"] = name
        lineups.append(lineup_record)

    df = pd.DataFrame(lineups, columns=LINEUP_COLS)
    return df
