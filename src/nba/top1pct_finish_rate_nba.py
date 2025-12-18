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
    field_size: int | None,
    lineups_excel: str | None = None,
    corr_excel: str | None = None,
    num_sims: int = 20_000,
    random_seed: int | None = None,
    field_var_shrink: float = 0.7,
    field_z_score: float = 2.0,
    flex_var_factor: float = 3.5,
    field_model: str = "mixture",
    run_dir: Path | None = None,
    contest_id: str | None = None,
    payouts_json: str | None = None,
    dkentries_csv: str | None = None,
    sim_batch_size: int = 200,
) -> Path:
    return top1pct_core.run_top1pct(
        field_size=field_size,
        outputs_dir=config.OUTPUTS_DIR,
        data_dir=config.DATA_DIR,
        lineups_excel=lineups_excel,
        corr_excel=corr_excel,
        num_sims=num_sims,
        random_seed=random_seed if random_seed is not None else config.SIM_RANDOM_SEED,
        field_var_shrink=field_var_shrink,
        field_z_score=field_z_score,
        flex_var_factor=flex_var_factor,
        field_model=field_model,
        run_dir=run_dir,
        contest_id=contest_id,
        payouts_json=payouts_json,
        dkentries_csv=dkentries_csv,
        sim_batch_size=sim_batch_size,
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
        default=None,
        help=(
            "Total number of lineups in the contest field (e.g., 23529). "
            "Optional when --contest-id is provided (field size is inferred)."
        ),
    )
    parser.add_argument(
        "--contest-id",
        type=str,
        default=None,
        help=(
            "Optional DraftKings contest id. When provided, the pipeline "
            "downloads contest payout info and computes EV payout + EV ROI per "
            "lineup."
        ),
    )
    parser.add_argument(
        "--payouts-json",
        type=str,
        default=None,
        help=(
            "Optional path to a cached DraftKings contest JSON file. When "
            "provided, bypasses the download and reads this file directly."
        ),
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
        "--dkentries-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a DKEntries CSV template. When provided (or when a "
            "latest DKEntries*.csv can be found under data/nba/dkentries/), the "
            "output workbook will include a DK_Lineups sheet formatted as "
            "'Name (dk_player_id)' for copy/paste into DKEntries."
        ),
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=100_000,
        help="Number of Monte Carlo simulations to run (default: 100000).",
    )
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=200,
        help=(
            "Simulation batch size for streaming scoring (default: 200). "
            "Smaller values use less memory but may be slower."
        ),
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
            "directory instead of under outputs/nba/top1pct/."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.field_size is None and args.contest_id is None:
        raise SystemExit("Error: --field-size is required unless --contest-id is provided.")
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
        contest_id=args.contest_id,
        payouts_json=args.payouts_json,
        dkentries_csv=args.dkentries_csv,
        sim_batch_size=args.sim_batch_size,
    )


if __name__ == "__main__":
    main()


