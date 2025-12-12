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


def _lineup_salary(
    *,
    cpt: str,
    flex_players: List[str],
    salary_by_name: Dict[str, float],
) -> float:
    """Compute DraftKings Showdown salary with CPT 1.5× weighting."""
    cpt_salary = 1.5 * float(salary_by_name.get(cpt, 0.0))
    flex_salary = sum(float(salary_by_name.get(p, 0.0)) for p in flex_players)
    return cpt_salary + flex_salary


def _build_sorted_salary_index(
    eligible_names: List[str],
    salary_by_name: Dict[str, float],
) -> tuple[List[tuple[float, str]], np.ndarray, Dict[str, int]]:
    """Build a stable sorted salary list + prefix sums for fast k-sum queries."""
    pairs = [(float(salary_by_name.get(n, 0.0)), str(n)) for n in eligible_names]
    pairs.sort(key=lambda x: (x[0], x[1]))
    sal_arr = np.array([p[0] for p in pairs], dtype=float)
    prefix = np.concatenate(([0.0], np.cumsum(sal_arr)))
    pos = {name: i for i, (_s, name) in enumerate(pairs)}
    return pairs, prefix, pos


def _sum_smallest_k_excluding(
    *,
    k: int,
    pairs: List[tuple[float, str]],
    prefix: np.ndarray,
    pos: Dict[str, int],
    exclude_name: str | None,
) -> float:
    if k <= 0:
        return 0.0
    n = len(pairs)
    if n < k:
        return float("inf")
    if exclude_name is None:
        return float(prefix[k])
    idx = pos.get(exclude_name)
    if idx is None:
        return float(prefix[k])
    # If excluded item falls outside the smallest-k window, smallest-k sum unchanged.
    if idx >= k:
        return float(prefix[k])
    # Otherwise, take k+1 and subtract the excluded salary.
    if n < k + 1:
        return float("inf")
    return float(prefix[k + 1] - pairs[idx][0])


def _sum_largest_k_excluding(
    *,
    k: int,
    pairs: List[tuple[float, str]],
    prefix: np.ndarray,
    pos: Dict[str, int],
    exclude_name: str | None,
) -> float:
    if k <= 0:
        return 0.0
    n = len(pairs)
    if n < k:
        return float("-inf")
    total = float(prefix[n])
    if exclude_name is None:
        return float(total - prefix[n - k])
    idx = pos.get(exclude_name)
    if idx is None:
        return float(total - prefix[n - k])
    start_last_k = n - k
    # If excluded item not in last-k window, last-k sum unchanged.
    if idx < start_last_k:
        return float(total - prefix[start_last_k])
    # Otherwise, take last k+1 and subtract excluded salary.
    if n < k + 1:
        return float("-inf")
    sum_last_k_plus_1 = float(total - prefix[n - (k + 1)])
    return float(sum_last_k_plus_1 - pairs[idx][0])


def _salary_bounds_feasible(
    *,
    used_salary: float,
    slots_left: int,
    cfg: "FieldBuilderConfig",
    pairs: List[tuple[float, str]],
    prefix: np.ndarray,
    pos: Dict[str, int],
    exclude_name: str | None,
) -> bool:
    min_rem = _sum_smallest_k_excluding(
        k=slots_left, pairs=pairs, prefix=prefix, pos=pos, exclude_name=exclude_name
    )
    if not np.isfinite(min_rem):
        return False
    if used_salary + min_rem > float(cfg.salary_cap):
        return False
    max_rem = _sum_largest_k_excluding(
        k=slots_left, pairs=pairs, prefix=prefix, pos=pos, exclude_name=exclude_name
    )
    if not np.isfinite(max_rem):
        return False
    if used_salary + max_rem < float(cfg.min_salary):
        return False
    return True


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

    # Retry behavior for lineup construction.
    max_attempts_per_lineup: int = 200
    relax_quotas_after_attempts: int = 20
    # When relaxing quotas, reduce effective beta by this factor but never violate salary.
    relaxed_beta_factor: float = 0.25


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
    *,
    relaxed_quotas: bool = False,
    flex_salary_index: tuple[List[tuple[float, str]], np.ndarray, Dict[str, int]] | None = None,
) -> str:
    epsilon = 1e-6
    names: List[str] = []
    weights: List[float] = []

    if flex_salary_index is None:
        # Default to all players in universe as flex-eligible (excluding CPT later).
        flex_names = [str(n) for n in universe]
        flex_salary_index = _build_sorted_salary_index(flex_names, salary)
    flex_pairs, flex_prefix, flex_pos = flex_salary_index

    for name in universe:
        quota_left = remaining_cpt.get(name, 0)
        if not relaxed_quotas and quota_left <= 0:
            continue
        # Salary feasibility with 5 flex slots remaining (exclude CPT name).
        used_salary = 1.5 * float(salary.get(name, 0.0))
        if not _salary_bounds_feasible(
            used_salary=used_salary,
            slots_left=5,
            cfg=cfg,
            pairs=flex_pairs,
            prefix=flex_prefix,
            pos=flex_pos,
            exclude_name=str(name),
        ):
            continue

        beta_eff = float(cfg.beta) * (float(cfg.relaxed_beta_factor) if relaxed_quotas else 1.0)
        quota_for_weight = max(float(quota_left), 0.0)
        v = _estimate_value(proj.get(name, 0.0) * 1.5, salary.get(name, 0.0) * 1.5)
        value_term = max(v, 0.0) ** cfg.alpha
        quota_term = (quota_for_weight + epsilon) ** beta_eff
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
    *,
    relaxed_quotas: bool = False,
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
        # Build eligible pool for this slot and precompute salary index for fast bounds.
        selected = set(current_lineup) | set(flex_players)
        eligible_names: List[str] = []
        for n in universe:
            name = str(n)
            if name in selected:
                continue
            if not relaxed_quotas and remaining_flex.get(name, 0) <= 0:
                continue
            eligible_names.append(name)
        pairs, prefix, pos_map = _build_sorted_salary_index(eligible_names, salary)

        names: List[str] = []
        weights: List[float] = []

        for name in universe:
            if name in current_lineup or name in flex_players:
                continue
            quota_left = remaining_flex.get(name, 0)
            if not relaxed_quotas and quota_left <= 0:
                continue

            used_salary = _lineup_salary(
                cpt=current_lineup[0], flex_players=flex_players, salary_by_name=salary
            )
            cand_salary = float(salary.get(name, 0.0))
            if used_salary + cand_salary > float(cfg.salary_cap):
                continue
            slots_left_after = 5 - (len(flex_players) + 1)
            if slots_left_after > 0:
                # Feasible bounds after selecting candidate (exclude candidate from remaining pool).
                if not _salary_bounds_feasible(
                    used_salary=used_salary + cand_salary,
                    slots_left=slots_left_after,
                    cfg=cfg,
                    pairs=pairs,
                    prefix=prefix,
                    pos=pos_map,
                    exclude_name=str(name),
                ):
                    continue
            else:
                # Final slot: ensure we also meet min salary.
                total_after = used_salary + cand_salary
                if total_after < float(cfg.min_salary):
                    continue

            beta_eff = float(cfg.beta) * (float(cfg.relaxed_beta_factor) if relaxed_quotas else 1.0)
            quota_for_weight = max(float(quota_left), 0.0)
            v = _estimate_value(proj.get(name, 0.0), salary.get(name, 0.0))
            value_term = max(v, 0.0) ** cfg.alpha
            quota_term = (quota_for_weight + 1e-6) ** beta_eff
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
                used_salary = _lineup_salary(
                    cpt=current_lineup[0], flex_players=flex_players, salary_by_name=salary
                )
                cand_salary = float(salary.get(name, 0.0))
                if used_salary + cand_salary > float(cfg.salary_cap):
                    continue
                slots_left_after = 5 - (len(flex_players) + 1)
                if slots_left_after > 0:
                    if not _salary_bounds_feasible(
                        used_salary=used_salary + cand_salary,
                        slots_left=slots_left_after,
                        cfg=cfg,
                        pairs=pairs,
                        prefix=prefix,
                        pos=pos_map,
                        exclude_name=str(name),
                    ):
                        continue
                else:
                    total_after = used_salary + cand_salary
                    if total_after < float(cfg.min_salary):
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

    # Precompute an index of all flex salaries for CPT feasibility checks.
    flex_salary_index = _build_sorted_salary_index([str(n) for n in universe], salary)

    for lineup_idx in range(field_size):
        built = False
        last_err: Exception | None = None
        for attempt in range(int(cfg.max_attempts_per_lineup)):
            relaxed = attempt >= int(cfg.relax_quotas_after_attempts)
            # Transactional local copies: only commit if lineup is accepted.
            local_cpt = dict(remaining_cpt)
            local_flex = dict(remaining_flex)
            try:
                # Sample CPT.
                cpt = _sample_cpt(
                    rng=rng,
                    universe=universe,
                    remaining_cpt=local_cpt,
                    proj=proj,
                    salary=salary,
                    pos=pos,
                    cfg=cfg,
                    relaxed_quotas=relaxed,
                    flex_salary_index=flex_salary_index,
                )
                local_cpt[cpt] = local_cpt.get(cpt, 0) - 1

                # Sample FLEX sequence.
                flex_players = _sample_flex_sequence(
                    rng=rng,
                    universe=universe,
                    remaining_flex=local_flex,
                    proj=proj,
                    salary=salary,
                    pos=pos,
                    team=team,
                    corr_lookup=corr_lookup,
                    cfg=cfg,
                    current_lineup=[cpt],
                    relaxed_quotas=relaxed,
                )

                total_salary = _lineup_salary(
                    cpt=cpt, flex_players=flex_players, salary_by_name=salary
                )
                if total_salary > float(cfg.salary_cap) or total_salary < float(cfg.min_salary):
                    raise RuntimeError(
                        "Constructed lineup violates salary constraints "
                        f"(salary={total_salary:.1f}, min={cfg.min_salary}, cap={cfg.salary_cap})."
                    )

                # Commit quota decrements.
                remaining_cpt[cpt] = remaining_cpt.get(cpt, 0) - 1
                for p in flex_players:
                    remaining_flex[p] = remaining_flex.get(p, 0) - 1

                lineup_record: Dict[str, str] = {"cpt": cpt}
                for i, name in enumerate(flex_players, start=1):
                    lineup_record[f"flex{i}"] = name
                lineups.append(lineup_record)
                built = True
                break
            except Exception as exc:
                last_err = exc
                continue

        if not built:
            salaries = [float(salary.get(str(n), 0.0)) for n in universe]
            s_min = min(salaries) if salaries else 0.0
            s_max = max(salaries) if salaries else 0.0
            raise RuntimeError(
                "Failed to construct a valid field lineup under salary constraints "
                f"after {cfg.max_attempts_per_lineup} attempts "
                f"(lineup_index={lineup_idx}, min_salary={cfg.min_salary}, "
                f"salary_cap={cfg.salary_cap}, universe_size={len(universe)}, "
                f"salary_min={s_min:.1f}, salary_max={s_max:.1f}). "
                f"Last error: {last_err}"
            )

    df = pd.DataFrame(lineups, columns=LINEUP_COLS)
    return df
