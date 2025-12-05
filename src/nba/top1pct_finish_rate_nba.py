from __future__ import annotations

"""
NBA wrapper for estimating top 1% finish probabilities for Showdown lineups.

This mirrors the NFL CLI but delegates the core logic to
`src.shared.top1pct_core.run_top1pct` and uses NBA paths/config.
"""

import argparse
from pathlib import Path
from typing import List

from ..shared import top1pct_core
from . import config


def run(
    field_size: int,
    lineups_excel: str | None = None,
    corr_excel: str | None = None,
    num_sims: int = 20_000,
    random_seed: int | None = None,
    field_var_shrink: float = 0.7,
    field_z_score: float = 2.0,
    flex_var_factor: float = 3.5,
) -> Path:
    return top1pct_core.run_top1pct(
        field_size=field_size,
        outputs_dir=config.OUTPUTS_DIR,
        lineups_excel=lineups_excel,
        corr_excel=corr_excel,
        num_sims=num_sims,
        random_seed=random_seed if random_seed is not None else config.SIM_RANDOM_SEED,
        field_var_shrink=field_var_shrink,
        field_z_score=field_z_score,
        flex_var_factor=flex_var_factor,
    )


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate top 1% finish probabilities for NBA Showdown lineups "
            "using correlated player outcomes and ownership."
        )
    )
    parser.add_argument(
        "--field-size",
        type=int,
        required=True,
        help="Total number of lineups in the contest field (e.g., 23529).",
    )
    parser.add_argument(
        "--lineups-excel",
        type=str,
        default=None,
        help=(
            "Path to lineups Excel workbook under outputs/nba/lineups/. "
            "If omitted, the most recent .xlsx file in that directory is used."
        ),
    )
    parser.add_argument(
        "--corr-excel",
        type=str,
        default=None,
        help=(
            "Path to correlations Excel workbook under outputs/nba/correlations/. "
            "If omitted, the most recent .xlsx file in that directory is used."
        ),
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=100_000,
        help="Number of Monte Carlo simulations to run (default: 100000).",
    )
    parser.add_argument(
        "--field-var-shrink",
        type=float,
        default=0.7,
        help=(
            "Multiplicative shrinkage factor for the modeled field variance "
            "(0 < value <= 1, default: 0.7)."
        ),
    )
    parser.add_argument(
        "--field-z",
        type=float,
        default=2.0,
        help=(
            "Z-score used for the upper tail of the field score distribution "
            "(default: 2.0)."
        ),
    )
    parser.add_argument(
        "--flex-var-factor",
        type=float,
        default=3.5,
        help=(
            "Effective variance factor for the aggregate non-CPT flex-style "
            "component (<= 5.0; default: 3.5)."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        field_size=args.field_size,
        lineups_excel=args.lineups_excel,
        corr_excel=args.corr_excel,
        num_sims=args.num_sims,
        random_seed=args.random_seed,
        field_var_shrink=args.field_var_shrink,
        field_z_score=args.field_z,
        flex_var_factor=args.flex_var_factor,
    )


if __name__ == "__main__":
    main()


