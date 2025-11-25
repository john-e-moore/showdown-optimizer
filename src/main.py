from __future__ import annotations

"""
Entry point for the NFL Showdown correlation pipeline.

On first run (or with --retrain), this script:
  1. Loads historical nflverse-style Parquet data.
  2. Computes DK offensive fantasy points and per-player z-scores.
  3. Builds a pairwise training dataset.
  4. Trains a correlation regression model and saves it to disk.

On every run, it:
  1. Loads the trained model.
  2. Loads Sabersim projections for a single Showdown slate.
  3. Builds a player correlation matrix for that slate.
  4. Writes an Excel file with projections and the correlation matrix.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import (
    build_corr_matrix_from_projections,
    build_pairwise_dataset,
    config,
    data_loading,
    fantasy_scoring,
    feature_engineering,
    train_corr_model,
)


def _train_historical_model() -> None:
    """
    Run the full historical pipeline and train the correlation model.
    """
    print("Loading historical player-game stats and games...")
    player_games = data_loading.load_player_game_stats(
        config.NFL_PLAYER_GAMES_PARQUET
    )
    games = data_loading.load_games(config.NFL_GAMES_PARQUET)

    print("Computing DK offensive fantasy points...")
    player_games = fantasy_scoring.compute_dk_points_offense(player_games)

    print("Computing per-player DK stats and z-scores...")
    player_games_z = feature_engineering.add_player_dk_stats(
        player_games, config.MIN_PLAYER_GAMES
    )

    print("Building pairwise training dataset...")
    pairwise_df = build_pairwise_dataset.build_pairwise_dataset(
        player_games_z, games
    )

    print("Training correlation regression model...")
    train_corr_model.train_corr_regressor(pairwise_df)
    print(f"Model saved to {config.CORR_MODEL_PATH}")


def _load_model() -> object:
    """
    Load the trained correlation model pipeline from disk.
    """
    if not Path(config.CORR_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Correlation model not found at {config.CORR_MODEL_PATH}. "
            "Run with --retrain to train it from historical data."
        )
    return joblib.load(config.CORR_MODEL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFL Showdown player correlation pipeline"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the correlation model from historical data.",
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

    args = parser.parse_args()

    config.ensure_directories()

    if args.retrain or not Path(config.CORR_MODEL_PATH).exists():
        _train_historical_model()

    print("Loading trained correlation model...")
    model = _load_model()

    print(f"Loading Sabersim projections from {args.sabersim_csv}...")
    sabersim_df = build_corr_matrix_from_projections.load_sabersim_projections(
        args.sabersim_csv
    )

    print("Building player correlation matrix...")
    corr_matrix = (
        build_corr_matrix_from_projections.build_corr_matrix_from_sabersim(
            model, sabersim_df
        )
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


