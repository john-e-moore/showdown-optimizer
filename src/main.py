from __future__ import annotations

"""
Entry point for the NFL Showdown simulation-based correlation pipeline.

This script:
  1. Loads Sabersim projections for a single Showdown slate.
  2. Runs a Monte Carlo simulator to generate joint DK fantasy outcomes.
  3. Builds a player correlation matrix from the simulated DK points.
  4. Writes an Excel file with projections and the correlation matrix.
"""

import argparse
from pathlib import Path

import pandas as pd

from . import (
    build_corr_matrix_from_projections,
    config,
    simulation_corr,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFL Showdown player correlation pipeline"
    )
    parser.add_argument(
        "--sabersim-csv",
        type=str,
        default=config.SABERSIM_CSV,
        help="Path to Sabersim Showdown projections CSV.",
    )
    parser.add_argument(
        "--output-excel",
        type=str,
        default=config.OUTPUT_CORR_EXCEL,
        help="Path to output Excel file containing projections and correlation matrix.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=config.SIM_N_GAMES,
        help="Number of Monte Carlo simulations to run.",
    )

    args = parser.parse_args()

    config.ensure_directories()

    print(f"Loading Sabersim projections from {args.sabersim_csv}...")
    sabersim_df = build_corr_matrix_from_projections.load_sabersim_projections(
        args.sabersim_csv
    )

    print(
        f"Building player correlation matrix via simulation "
        f"({args.n_sims} simulated games)..."
    )
    corr_matrix = simulation_corr.simulate_corr_matrix_from_projections(
        sabersim_df, n_sims=args.n_sims
    )

    print(f"Writing Excel output to {args.output_excel}...")
    output_path = Path(args.output_excel)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        sabersim_df.to_excel(
            writer, sheet_name="Sabersim_Projections", index=False
        )
        corr_matrix.to_excel(
            writer, sheet_name="Correlation_Matrix", index=True
        )

    print("Done.")


if __name__ == "__main__":
    main()


