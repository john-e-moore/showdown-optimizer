from __future__ import annotations

"""
CLI entry point for the NFL Showdown lineup optimizer.

Example:
    python -m src.showdown_optimizer_main \
      --sabersim-glob "data/sabersim/NFL_*.csv" \
      --num-lineups 50 \
      --salary-cap 50000
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
import time

import pandas as pd

from . import config
from .lineup_constraints import build_custom_constraints
from .lineup_optimizer import (
    Lineup,
    load_players_from_sabersim,
    optimize_showdown_lineups,
)


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

    # FLEX players in order
    for p in lineup.flex:
        lines.append(
            f"  FLEX | {p.name:25s} {p.team:3s} {p.position:4s} "
            f"salary={p.dk_salary:5d} proj={p.dk_proj:6.2f}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFL Showdown (Captain Mode) lineup optimizer"
    )
    parser.add_argument(
        "--sabersim-glob",
        type=str,
        default=config.SABERSIM_CSV,
        help=(
            "Glob pattern for Sabersim Showdown projections CSV. "
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

    # Load players for ownership/opponent metadata.
    player_pool = load_players_from_sabersim(csv_path)
    players_by_id = {p.player_id: p for p in player_pool.players}

    # Prepare any custom DFS constraints from the configuration module.
    constraint_builders = build_custom_constraints()

    # Optimize lineups (pattern can be a concrete path) and time the run.
    start_time = time.perf_counter()
    lineups = optimize_showdown_lineups(
        projections_path_pattern=csv_path,
        num_lineups=args.num_lineups,
        salary_cap=args.salary_cap,
        constraint_builders=constraint_builders,
    )
    elapsed = time.perf_counter() - start_time

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

    print(f"Generated {len(lineups)} lineups.")

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

    # Map team -> opponent (assume standard two-team Showdown slate).
    teams = sorted({p.team for p in player_pool.players})
    if len(teams) == 2:
        opp_map = {teams[0]: teams[1], teams[1]: teams[0]}
    else:
        opp_map = {t: "" for t in teams}

    # Build exposure table based on how often each player appears in CPT/FLEX.
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
    # Build ownership table from Sabersim \"My Own\" column.
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

    # For each (Name, Team) pair, identify CPT vs FLEX rows by projection:
    #   - Larger 'My Proj' => CPT row
    #   - Other row => FLEX row
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
            "cpt": fmt_player_with_own(lineup.cpt.player_id, slot="CPT"),
        }
        for j, p in enumerate(lineup.flex, start=1):
            col_name = f"flex{j}"
            row[col_name] = fmt_player_with_own(p.player_id, slot="FLEX")

        lineup_rows.append(row)

    lineups_df = pd.DataFrame(lineup_rows)

    # ------------------------------------------------------------------
    # Write Excel workbook with projections, ownership, exposure, and lineups.
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lineups_dir = Path(config.OUTPUTS_DIR) / "lineups"
    lineups_dir.mkdir(parents=True, exist_ok=True)
    output_path = lineups_dir / f"lineups_{timestamp}.xlsx"

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        sabersim_df.to_excel(writer, sheet_name="Projections", index=False)
        ownership_df.to_excel(writer, sheet_name="Ownership", index=False)
        exposure_df.to_excel(writer, sheet_name="Exposure", index=False)
        lineups_df.to_excel(writer, sheet_name="Lineups", index=False)

    print(f"Wrote lineup workbook to {output_path}")


if __name__ == "__main__":
    main()


