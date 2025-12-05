from __future__ import annotations

"""
NBA wrapper for diversifying Showdown top1pct lineups.

This mirrors the NFL CLI but delegates the selection and exposure logic
to `src.shared.diversify_core.run_diversify` and uses NBA paths.
"""

import argparse
from typing import List

from ..shared import diversify_core
from . import config


def run(
    num_lineups: int,
    *,
    min_top1_pct: float = 1.0,
    max_overlap: int = 4,
    top1pct_excel: str | None = None,
):
    return diversify_core.run_diversify(
        num_lineups=num_lineups,
        outputs_dir=config.OUTPUTS_DIR,
        min_top1_pct=min_top1_pct,
        max_overlap=max_overlap,
        top1pct_excel=top1pct_excel,
    )


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a diversified subset of NBA Showdown lineups based on "
            "top 1% finish rate and player-overlap constraints."
        )
    )
    parser.add_argument(
        "--num-lineups",
        type=int,
        required=True,
        help="Number of lineups (X) to select.",
    )
    parser.add_argument(
        "--min-top1-pct",
        type=float,
        default=1.0,
        help=(
            "Minimum top1_pct_finish_rate (in percent) required to keep a "
            "lineup (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=4,
        help=(
            "Maximum number of overlapping players allowed between any pair of "
            "selected lineups (0–6 for Showdown; default: 4)."
        ),
    )
    parser.add_argument(
        "--top1pct-excel",
        type=str,
        default=None,
        help=(
            "Optional path to a top1pct workbook under outputs/nba/top1pct/. "
            "If omitted, the most recent .xlsx file in that directory is used."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        num_lineups=args.num_lineups,
        min_top1_pct=args.min_top1_pct,
        max_overlap=args.max_overlap,
        top1pct_excel=args.top1pct_excel,
    )


if __name__ == "__main__":
    main()


