from __future__ import annotations

"""
Estimate top 1% finish probability for NFL Showdown lineups.

This script:
  1. Loads a lineup workbook from outputs/nfl/lineups/*.xlsx.
  2. Loads a correlation workbook from outputs/nfl/correlations/*.xlsx.
  3. Builds a multivariate normal model of player DK scores using
     means + std devs + correlation matrix.
  4. Uses ownership projections to approximate the field score distribution.
  5. Estimates, for each lineup, the probability of finishing in the
     top 1% of the modeled field.

Usage example:

    python -m src.nfl.top1pct_finish_rate --field-size 23529

You can also specify input workbooks explicitly:

    python -m src.nfl.top1pct_finish_rate \\
        --field-size 23529 \\
        --lineups-excel outputs/nfl/lineups/lineups_YYYYMMDD_HHMMSS.xlsx \\
        --corr-excel outputs/nfl/correlations/showdown_corr_matrix.xlsx
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
    field_model: str = "mixture",
    run_dir: Path | None = None,
) -> Path:
    """
    NFL wrapper around the shared top1pct core.

    This preserves the existing CLI while delegating the math and IO to
    `shared.top1pct_core.run_top1pct`.
    """
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
        field_model=field_model,
        run_dir=run_dir,
    )


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate top 1% finish probabilities for NFL Showdown lineups "
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
            "Path to lineups Excel workbook under outputs/nfl/lineups/. "
            "If omitted, the most recent .xlsx file in that directory is used."
        ),
    )
    parser.add_argument(
        "--corr-excel",
        type=str,
        default=None,
        help=(
            "Path to correlations Excel workbook under outputs/nfl/correlations/. "
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
            "(default: 2.0, slightly below the canonical 99th percentile 2.326)."
        ),
    )
    parser.add_argument(
        "--flex-var-factor",
        type=float,
        default=3.5,
        help=(
            "Effective variance factor for the aggregate FLEX component "
            "(<= 5.0; default: 3.5)."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for reproducibility. "
            "Defaults to config.SIM_RANDOM_SEED when omitted."
        ),
    )
    parser.add_argument(
        "--field-model",
        type=str,
        choices=["mixture", "explicit"],
        default="mixture",
        help=(
            "Field modeling approach: 'mixture' uses the existing analytic "
            "ownership-mixture approximation; 'explicit' simulates a "
            "quota-balanced field of lineups and uses empirical thresholds."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Optional directory in which to write the top1pct workbook. "
            "When provided, the output Excel file is placed directly in this "
            "directory instead of under outputs/nfl/top1pct/."
        ),
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
        field_model=args.field_model,
        run_dir=Path(args.run_dir) if args.run_dir is not None else None,
    )


if __name__ == "__main__":
    main()



