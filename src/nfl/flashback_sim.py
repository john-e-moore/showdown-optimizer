from __future__ import annotations

"""
Flashback contest simulation for completed NFL Showdown slates.

This CLI is a thin wrapper around the sport-agnostic core in
`src.shared.flashback_core`.
"""

import argparse
from typing import List

from ..shared import flashback_core
from . import build_corr_matrix_from_projections, config, simulation_corr


def run(
    contest_csv: str | None = None,
    sabersim_csv: str | None = None,
    num_sims: int = 100_000,
    random_seed: int | None = None,
    payouts_csv: str | None = None,
):
    """
    Execute the flashback contest simulation pipeline for NFL.
    """
    return flashback_core.run_flashback(
        contest_csv=contest_csv,
        sabersim_csv=sabersim_csv,
        num_sims=num_sims,
        random_seed=random_seed,
        payouts_csv=payouts_csv,
        config_module=config,
        load_sabersim_projections=build_corr_matrix_from_projections.load_sabersim_projections,
        simulate_corr_matrix_from_projections=simulation_corr.simulate_corr_matrix_from_projections,
        name_col=build_corr_matrix_from_projections.SABERSIM_NAME_COL,
        team_col=build_corr_matrix_from_projections.SABERSIM_TEAM_COL,
        salary_col=build_corr_matrix_from_projections.SABERSIM_SALARY_COL,
        dk_proj_col=build_corr_matrix_from_projections.SABERSIM_DK_PROJ_COL,
        flex_role_label="FLEX",
    )


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flashback contest simulation for completed NFL Showdown slates "
            "using Sabersim projections and a correlation matrix."
        )
    )
    parser.add_argument(
        "--contest-csv",
        type=str,
        default=None,
        help=(
            "Path to contest standings CSV under data/nfl/contests/. "
            "If omitted, the most recent .csv file in that directory is used."
        ),
    )
    parser.add_argument(
        "--sabersim-csv",
        type=str,
        default=None,
        help=(
            "Path to Sabersim projections CSV under data/nfl/sabersim/. "
            "If omitted, the most recent .csv file in that directory is used."
        ),
    )
    parser.add_argument(
        "--payouts-csv",
        type=str,
        default=None,
        help=(
            "Path to DraftKings payout JSON for this contest under data/nfl/payouts/. "
            "Despite the name, this flag expects the JSON format used in "
            "payouts-*.json. If omitted, a default payouts-{contest_id}.json "
            "is inferred from the contest filename or downloaded from DraftKings."
        ),
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=100_000,
        help="Number of Monte Carlo simulations to run (default: 100000).",
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
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        contest_csv=args.contest_csv,
        sabersim_csv=args.sabersim_csv,
        num_sims=args.num_sims,
        random_seed=args.random_seed,
        payouts_csv=args.payouts_csv,
    )


if __name__ == "__main__":
    main()



