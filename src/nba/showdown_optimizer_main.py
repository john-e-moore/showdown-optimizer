from __future__ import annotations

"""
CLI entry point for the NBA Showdown lineup optimizer.

This mirrors the NFL optimizer CLI but:
  - Uses NBA config defaults and writes to `outputs/nba/lineups`.
  - Uses the shared Showdown optimizer core via the NFL-adaptor
    `src.lineup_optimizer`.
  - Applies no NBA-specific DFS constraints by default (stacking is optional).
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from glob import glob
from pathlib import Path
import json
import time

import pandas as pd

from .. import showdown_constraints  # only for generic team stack helper
from ..shared.lineup_optimizer import (
    Lineup,
    load_players_from_sabersim,
    optimize_showdown_lineups,
)
from . import config


STACK_PATTERNS: tuple[str, ...] = ("5|1", "4|2", "3|3", "2|4", "1|5")
STACK_PATTERN_COUNTS: dict[str, tuple[int, int]] = {
    "5|1": (5, 1),
    "4|2": (4, 2),
    "3|3": (3, 3),
    "2|4": (2, 4),
    "1|5": (1, 5),
}


def _format_lineup(lineup: Lineup, idx: int) -> str:
    lines: list[str] = []
    header = f"Lineup {idx + 1}: salary={lineup.salary()} proj={lineup.projection():.2f}"
    lines.append(header)

    # CPT first
    cpt = lineup.cpt
    lines.append(
        f"  CPT  | {cpt.name:25s} {cpt.team:3s} {cpt.position:4s} "
        f"salary={cpt.cpt_salary:5d} proj={1.5 * cpt.dk_proj:6.2f}"
    )

    # UTIL players in order for NBA Showdown
    for p in lineup.flex:
        lines.append(
            f"  UTIL | {p.name:25s} {p.team:3s} {p.position:4s} "
            f"salary={p.dk_salary:5d} proj={p.dk_proj:6.2f}"
        )

    return "\n".join(lines)


def _parse_stack_weights(raw: str | None) -> dict[str, float]:
    """
    Parse a stack-weights string into normalized weights over STACK_PATTERNS.

    Examples:
        None -> equal weights over all patterns.
        "5|1=0.3,4|2=0.25,3|3=0.2,2|4=0.15,1|5=0.1"
    """
    if raw is None:
        # Equal weights over all patterns.
        return {pattern: 1.0 / len(STACK_PATTERNS) for pattern in STACK_PATTERNS}

    weights: dict[str, float] = {pattern: 0.0 for pattern in STACK_PATTERNS}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid stack weight component {part!r}; expected 'PATTERN=weight'."
            )
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in STACK_PATTERNS:
            raise ValueError(
                f"Unknown stack pattern {key!r} in --stack-weights. "
                f"Expected one of {list(STACK_PATTERNS)}."
            )
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight {value!r} for pattern {key!r}; expected a float."
            ) from exc
        if weight < 0:
            raise ValueError(
                f"Negative weight {weight!r} for pattern {key!r} is not allowed."
            )
        weights[key] = weight

    total = sum(weights.values())
    if total <= 0:
        raise ValueError(
            "All stack weights are zero or negative; please specify positive weights."
        )

    return {pattern: (weights[pattern] / total) for pattern in STACK_PATTERNS}


def _allocate_lineups_across_patterns(
    num_lineups: int, weights: dict[str, float]
) -> dict[str, int]:
    """
    Given total num_lineups and normalized weights, compute integer lineup
    counts per stack pattern that sum exactly to num_lineups.
    """
    if num_lineups <= 0:
        return {pattern: 0 for pattern in STACK_PATTERNS}

    # Initial floor allocation.
    counts: dict[str, int] = {}
    remaining = num_lineups
    for pattern in STACK_PATTERNS:
        w = max(weights.get(pattern, 0.0), 0.0)
        n = int(num_lineups * w)
        counts[pattern] = n
        remaining -= n

    # Distribute any leftover lineups to patterns with positive weight in a
    # fixed priority order.
    positive_patterns = [p for p in STACK_PATTERNS if weights.get(p, 0.0) > 0.0]
    if not positive_patterns:
        positive_patterns = list(STACK_PATTERNS)

    idx = 0
    while remaining > 0:
        pattern = positive_patterns[idx % len(positive_patterns)]
        counts[pattern] += 1
        remaining -= 1
        idx += 1

    return counts


def _build_stack_pattern_label(
    pattern: str,
    team_a: str,
    team_b: str,
    team_a_count: int,
    team_b_count: int,
) -> str:
    """
    Human-readable label describing which stack pattern/run produced a lineup.
    """
    if team_a_count > team_b_count:
        return f"{pattern}_{team_a}-heavy"
    if team_b_count > team_a_count:
        return f"{pattern}_{team_b}-heavy"
    return pattern


def _optimize_for_stack_pattern(
    pattern: str,
    n: int,
    *,
    team_a: str,
    team_b: str,
    team_a_count: int,
    team_b_count: int,
    base_constraint_builders: list,
    player_pool,
    salary_cap: int,
    chunk_size: int,
    use_warm_start: bool,
    solver_max_seconds: float | None,
    solver_rel_gap: float | None,
) -> tuple[list[Lineup], str, dict[str, object]]:
    """
    Helper to optimize a batch of lineups for a single stack pattern.

    Returns:
        (lineups, label, metrics_dict)
    """
    stack_constraint = showdown_constraints.build_team_stack_constraint(
        team_a=team_a,
        team_b=team_b,
        team_a_count=team_a_count,
        team_b_count=team_b_count,
    )
    pattern_constraint_builders = list(base_constraint_builders) + [stack_constraint]

    start = time.perf_counter()
    lineups = optimize_showdown_lineups(
        player_pool=player_pool,
        num_lineups=n,
        salary_cap=salary_cap,
        constraint_builders=pattern_constraint_builders,
        chunk_size=chunk_size,
        use_warm_start=use_warm_start,
        max_seconds=solver_max_seconds,
        rel_gap=solver_rel_gap,
    )
    elapsed = time.perf_counter() - start

    label = _build_stack_pattern_label(
        pattern, team_a, team_b, team_a_count, team_b_count
    )
    metrics: dict[str, object] = {
        "pattern": pattern,
        "seconds": elapsed,
        "num_lineups_requested": int(n),
        "num_lineups_generated": int(len(lineups)),
        "team_a": team_a,
        "team_b": team_b,
        "team_a_count": int(team_a_count),
        "team_b_count": int(team_b_count),
    }
    return lineups, label, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NBA Showdown (Captain Mode) lineup optimizer"
    )
    parser.add_argument(
        "--sabersim-glob",
        type=str,
        default=config.SABERSIM_CSV,
        help=(
            "Glob pattern for Sabersim NBA Showdown projections CSV. "
            "Must resolve to exactly one file."
        ),
    )
    parser.add_argument(
        "--num-lineups",
        type=int,
        default=20,
        help="Number of optimal lineups to generate.",
    )
    parser.add_argument(
        "--salary-cap",
        type=int,
        default=50_000,
        help="DraftKings Showdown salary cap.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help=(
            "Number of lineups to solve per MILP chunk when generating many "
            "lineups. Set to 0 or a negative value to disable chunking and "
            "use a single growing model with growing no-duplicate constraints."
        ),
    )
    parser.add_argument(
        "--stack-mode",
        type=str,
        choices=["none", "multi"],
        default="none",
        help=(
            "Stacking mode: 'none' (default) runs a single optimization pass. "
            "'multi' splits --num-lineups across 5|1, 4|2, 3|3, 2|4, 1|5 "
            "team stack patterns and runs one pass per pattern."
        ),
    )
    parser.add_argument(
        "--stack-weights",
        type=str,
        default=None,
        help=(
            "Optional weights for multi-stack mode, e.g. "
            "'5|1=0.3,4|2=0.25,3|3=0.2,2|4=0.15,1|5=0.1'. "
            "If omitted in multi-stack mode, all patterns are weighted equally."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads to use for parallel optimization when "
            "--parallel-mode is not 'none'."
        ),
    )
    parser.add_argument(
        "--parallel-mode",
        type=str,
        choices=["none", "by_stack_pattern"],
        default="none",
        help=(
            "Parallelization strategy when generating many lineups. "
            "'none' runs everything sequentially. 'by_stack_pattern' "
            "parallelizes across stack patterns in multi-stack mode."
        ),
    )
    parser.add_argument(
        "--use-warm-start",
        action="store_true",
        help=(
            "Enable CBC warm starts when using the single-model path "
            "(chunk-size <= 0), reusing solution information across lineups."
        ),
    )
    parser.add_argument(
        "--solver-max-seconds",
        type=float,
        default=None,
        help=(
            "Optional per-solve time limit in seconds for the CBC solver. "
            "When set, solves may stop early and return the best incumbent "
            "solution found so far."
        ),
    )
    parser.add_argument(
        "--solver-rel-gap",
        type=float,
        default=None,
        help=(
            "Optional relative optimality gap for CBC (e.g., 0.005 for 0.5%). "
            "Allows early termination once the MIP gap is small enough."
        ),
    )
    parser.add_argument(
        "--output-excel",
        type=str,
        default=None,
        help=(
            "Optional explicit path for the output Excel workbook. "
            "If omitted, a timestamped file is written under outputs/nba/lineups/."
        ),
    )

    args = parser.parse_args()

    sabersim_pattern = args.sabersim_glob
    print(f"Using Sabersim projections from pattern: {sabersim_pattern!r}")

    # Resolve projections path to a single CSV file.
    matches = sorted(glob(sabersim_pattern))
    if not matches:
        raise FileNotFoundError(
            f"No Sabersim CSVs found matching pattern: {sabersim_pattern!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Expected exactly one Sabersim CSV for optimizer, "
            f"but found {len(matches)}: {matches}"
        )

    csv_path = matches[0]

    # Load players for ownership/opponent metadata and optimization.
    player_pool = load_players_from_sabersim(csv_path)
    players_by_id = {p.player_id: p for p in player_pool.players}
    teams = sorted({p.team for p in player_pool.players})

    # No NBA-specific DFS constraints by default; stacking constraints are
    # added only in multi-stack mode below.
    base_constraint_builders: list = []

    # Optimize lineups (pattern can be a concrete path) and time the run.
    start_time = time.perf_counter()

    stack_pattern_labels: list[str] = []
    pattern_metrics: list[dict[str, object]] = []

    if args.stack_mode == "none":
        lineups = optimize_showdown_lineups(
            player_pool=player_pool,
            num_lineups=args.num_lineups,
            salary_cap=args.salary_cap,
            constraint_builders=base_constraint_builders,
            chunk_size=args.chunk_size,
            use_warm_start=args.use_warm_start,
            max_seconds=args.solver_max_seconds,
            rel_gap=args.solver_rel_gap,
        )
        stack_pattern_labels = ["none"] * len(lineups)
    else:
        # Multi-stack mode uses team-level stack constraints across the two teams.
        if len(teams) != 2:
            raise ValueError(
                "Multi-stack mode requires exactly two distinct teams in the slate."
            )

        team_a, team_b = teams
        weights = _parse_stack_weights(args.stack_weights)
        pattern_counts = _allocate_lineups_across_patterns(
            args.num_lineups, weights
        )

        all_lineups: list[Lineup] = []
        all_labels: list[str] = []

        # Build jobs for each requested stack pattern in a fixed order.
        pattern_jobs: list[tuple[str, int, int, int]] = []
        for pattern in STACK_PATTERNS:
            n = pattern_counts.get(pattern, 0)
            if n <= 0:
                continue
            team_a_count, team_b_count = STACK_PATTERN_COUNTS[pattern]
            pattern_jobs.append((pattern, n, team_a_count, team_b_count))

        # Parallelize across patterns when requested; otherwise run sequentially.
        if args.parallel_mode == "by_stack_pattern" and args.num_workers > 1:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        _optimize_for_stack_pattern,
                        pattern,
                        n,
                        team_a=team_a,
                        team_b=team_b,
                        team_a_count=team_a_count,
                        team_b_count=team_b_count,
                        base_constraint_builders=base_constraint_builders,
                        player_pool=player_pool,
                        salary_cap=args.salary_cap,
                        chunk_size=args.chunk_size,
                        use_warm_start=args.use_warm_start,
                        solver_max_seconds=args.solver_max_seconds,
                        solver_rel_gap=args.solver_rel_gap,
                    )
                    for (pattern, n, team_a_count, team_b_count) in pattern_jobs
                ]

                # Preserve deterministic ordering by iterating in submission order.
                for (pattern, _n, _a_count, _b_count), fut in zip(pattern_jobs, futures):
                    pattern_lineups, label, metrics = fut.result()
                    if not pattern_lineups:
                        continue
                    pattern_metrics.append(metrics)
                    for lu in pattern_lineups:
                        all_lineups.append(lu)
                        all_labels.append(label)
        else:
            for pattern, n, team_a_count, team_b_count in pattern_jobs:
                pattern_lineups, label, metrics = _optimize_for_stack_pattern(
                    pattern,
                    n,
                    team_a=team_a,
                    team_b=team_b,
                    team_a_count=team_a_count,
                    team_b_count=team_b_count,
                    base_constraint_builders=base_constraint_builders,
                    player_pool=player_pool,
                    salary_cap=args.salary_cap,
                    chunk_size=args.chunk_size,
                    use_warm_start=args.use_warm_start,
                    solver_max_seconds=args.solver_max_seconds,
                    solver_rel_gap=args.solver_rel_gap,
                )
                if not pattern_lineups:
                    continue
                pattern_metrics.append(metrics)
                for lu in pattern_lineups:
                    all_lineups.append(lu)
                    all_labels.append(label)

        # Deduplicate by player composition across all runs.
        dedup_lineups: list[Lineup] = []
        dedup_labels: list[str] = []
        seen: set[tuple[str, ...]] = set()

        for lu, label in zip(all_lineups, all_labels):
            key = tuple(sorted(lu.as_tuple_ids()))
            if key in seen:
                continue
            seen.add(key)
            dedup_lineups.append(lu)
            dedup_labels.append(label)

        lineups = dedup_lineups
        stack_pattern_labels = dedup_labels

    elapsed = time.perf_counter() - start_time

    num_generated = len(lineups)
    opt_time_parts = [
        f"seconds={elapsed:.3f}",
        f"num_lineups_requested={args.num_lineups}",
        f"num_lineups_generated={num_generated}",
        f"stack_mode={args.stack_mode}",
        f"chunk_size={args.chunk_size}",
        f"parallel_mode={args.parallel_mode}",
        f"num_workers={args.num_workers}",
    ]
    if args.solver_max_seconds is not None:
        opt_time_parts.append(f"solver_max_seconds={args.solver_max_seconds}")
    if args.solver_rel_gap is not None:
        opt_time_parts.append(f"solver_rel_gap={args.solver_rel_gap}")
    opt_time_line = "OPT_TIME " + " ".join(opt_time_parts)
    print(opt_time_line)

    if not lineups:
        print("No feasible lineups found.")
        if elapsed < 60.0:
            print(f"Optimization completed in {elapsed:.2f}s.")
        else:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            print(f"Optimization completed in {minutes}m {seconds}s.")
        return

    if elapsed < 60.0:
        print(f"Optimization completed in {elapsed:.2f}s.")
    else:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"Optimization completed in {minutes}m {seconds}s.")

    print(f"Generated {num_generated} lineups.")

    # ------------------------------------------------------------------
    # Compute exposure statistics across generated lineups.
    # ------------------------------------------------------------------
    n_lineups = len(lineups)
    cpt_counts = defaultdict(int)
    flex_counts = defaultdict(int)

    for lineup in lineups:
        cpt_counts[lineup.cpt.player_id] += 1
        for p in lineup.flex:
            flex_counts[p.player_id] += 1

    if len(teams) == 2:
        opp_map = {teams[0]: teams[1], teams[1]: teams[0]}
    else:
        opp_map = {t: "" for t in teams}

    # Build exposure table based on how often each player appears in CPT/UTIL.
    exposure_rows = []
    for p in player_pool.players:
        pid = p.player_id
        cpt_share = cpt_counts[pid] / n_lineups if n_lineups > 0 else 0.0
        flex_share = flex_counts[pid] / n_lineups if n_lineups > 0 else 0.0
        total_share = cpt_share + flex_share

        exposure_rows.append(
            {
                "player": p.name,
                "team": p.team,
                "opponent": opp_map.get(p.team, ""),
                "cpt_exposure": 100.0 * cpt_share,
                "flex_exposure": 100.0 * flex_share,
                "total_exposure": 100.0 * total_share,
            }
        )

    exposure_df = pd.DataFrame(exposure_rows)
    exposure_df.sort_values(
        by=["total_exposure", "player"], ascending=[False, True], inplace=True
    )

    # ------------------------------------------------------------------
    # Build ownership table from Sabersim "My Own" column.
    # ------------------------------------------------------------------
    sabersim_df = pd.read_csv(csv_path)
    required_cols = {"Name", "Team", "My Proj", "My Own"}
    missing_cols = required_cols.difference(sabersim_df.columns)
    if missing_cols:
        raise KeyError(
            f"Sabersim CSV missing required columns {sorted(missing_cols)}. "
            "Expected at least 'Name', 'Team', 'My Proj', and 'My Own'."
        )

    def _to_pct(value: float) -> float:
        """
        Interpret Sabersim 'My Own' as already expressed in percentage units.

        Values like 0.67 are treated as 0.67% (not 67%), so we do not rescale.
        """
        return float(value)

    # For each (Name, Team) pair, identify CPT vs flex-style rows by projection:
    #   - Larger 'My Proj' => CPT row
    #   - Other row => flex-style row
    ownership_rows = []
    player_cpt_own_pct_by_id: dict[str, float] = {}
    player_flex_own_pct_by_id: dict[str, float] = {}

    grouped = sabersim_df.groupby(["Name", "Team"])
    own_by_name_team: dict[tuple[str, str], tuple[float, float]] = {}
    for (name, team), g in grouped:
        if len(g) == 0:
            continue
        g_sorted = g.sort_values(by="My Proj", ascending=False)
        cpt_row = g_sorted.iloc[0]
        cpt_raw = float(cpt_row["My Own"])
        cpt_pct = _to_pct(cpt_raw)

        if len(g_sorted) > 1:
            flex_row = g_sorted.iloc[1]
            flex_raw = float(flex_row["My Own"])
            flex_pct = _to_pct(flex_raw)
        else:
            flex_pct = 0.0

        own_by_name_team[(name, team)] = (cpt_pct, flex_pct)

    for p in player_pool.players:
        key = (p.name, p.team)
        cpt_pct, flex_pct = own_by_name_team.get(key, (0.0, 0.0))
        total_pct = cpt_pct + flex_pct

        player_cpt_own_pct_by_id[p.player_id] = cpt_pct
        player_flex_own_pct_by_id[p.player_id] = flex_pct

        ownership_rows.append(
            {
                "player": p.name,
                "team": p.team,
                "opponent": opp_map.get(p.team, ""),
                "cpt_ownership": cpt_pct,
                "flex_ownership": flex_pct,
                "total_ownership": total_pct,
            }
        )

    ownership_df = pd.DataFrame(ownership_rows)
    ownership_df.sort_values(
        by=["total_ownership", "player"], ascending=[False, True], inplace=True
    )

    # ------------------------------------------------------------------
    # Build lineups summary table.
    # ------------------------------------------------------------------
    lineup_rows = []
    for idx, lineup in enumerate(lineups):
        # Compute stack pattern (e.g., 5|1, 4|2, 3|3) from team counts.
        all_players = [lineup.cpt] + list(lineup.flex)
        team_counts = Counter(p.team for p in all_players)
        counts_sorted = sorted(team_counts.values(), reverse=True)
        stack_str = "|".join(str(c) for c in counts_sorted)

        def fmt_player_with_own(player_id: str, *, slot: str) -> str:
            player = players_by_id[player_id]
            if slot == "CPT":
                raw_pct = player_cpt_own_pct_by_id.get(player_id, 0.0)
            else:
                raw_pct = player_flex_own_pct_by_id.get(player_id, 0.0)
            pct_str = f"{raw_pct:.1f}"
            return f"{player.name} ({pct_str}%)"

        row = {
            "rank": idx + 1,
            "lineup_projection": lineup.projection(),
            "lineup_salary": lineup.salary(),
            "stack": stack_str,
            "target_stack_pattern": (
                stack_pattern_labels[idx] if idx < len(stack_pattern_labels) else ""
            ),
            "cpt": fmt_player_with_own(lineup.cpt.player_id, slot="CPT"),
        }
        for j, p in enumerate(lineup.flex, start=1):
            col_name = f"util{j}"
            row[col_name] = fmt_player_with_own(p.player_id, slot="UTIL")

        lineup_rows.append(row)

    lineups_df = pd.DataFrame(lineup_rows)

    # ------------------------------------------------------------------
    # Write Excel workbook with projections, ownership, exposure, and lineups.
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_excel:
        output_path = Path(args.output_excel)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        lineups_dir = Path(config.OUTPUTS_DIR) / "lineups"
        lineups_dir.mkdir(parents=True, exist_ok=True)
        output_path = lineups_dir / f"lineups_{timestamp}.xlsx"

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        sabersim_df.to_excel(writer, sheet_name="Projections", index=False)
        ownership_df.to_excel(writer, sheet_name="Ownership", index=False)
        exposure_df.to_excel(writer, sheet_name="Exposure", index=False)
        lineups_df.to_excel(writer, sheet_name="Lineups", index=False)

    # ------------------------------------------------------------------
    # Persist machine-readable optimization metrics alongside the workbook.
    # ------------------------------------------------------------------
    metrics: dict[str, object] = {
        "optimizer_seconds": elapsed,
        "num_lineups_requested": int(args.num_lineups),
        "num_lineups_generated": int(num_generated),
        "stack_mode": args.stack_mode,
        "chunk_size": int(args.chunk_size),
        "parallel_mode": args.parallel_mode,
        "num_workers": int(args.num_workers),
        "solver_max_seconds": args.solver_max_seconds,
        "solver_rel_gap": args.solver_rel_gap,
        "timestamp": timestamp,
        "output_excel": str(output_path),
    }
    if pattern_metrics:
        metrics["per_pattern"] = pattern_metrics

    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    try:
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote optimizer metrics to {metrics_path}")
    except OSError as exc:
        print(f"Warning: failed to write optimizer metrics JSON: {exc}")

    print(f"Wrote NBA lineup workbook to {output_path}")


if __name__ == "__main__":
    main()


