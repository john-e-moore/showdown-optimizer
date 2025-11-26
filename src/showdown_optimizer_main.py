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
from pathlib import Path

from . import config
from .lineup_optimizer import Lineup, optimize_showdown_lineups


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

    lineups = optimize_showdown_lineups(
        projections_path_pattern=sabersim_pattern,
        num_lineups=args.num_lineups,
        salary_cap=args.salary_cap,
        constraint_builders=None,
    )

    if not lineups:
        print("No feasible lineups found.")
        return

    print(f"Generated {len(lineups)} lineups:")
    print("-" * 72)
    for i, lineup in enumerate(lineups):
        print(_format_lineup(lineup, i))
        print("-" * 72)


if __name__ == "__main__":
    main()


