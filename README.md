## NFL Showdown Correlation Pipeline

This project implements the first-stage pipeline for modeling player-to-player
fantasy point correlations for DraftKings NFL Showdown contests.

The pipeline now supports two ways to estimate player-to-player correlations:
- **Simulation-based (default)**: Monte Carlo simulator that uses Sabersim
  box-score projections to generate many joint game outcomes and computes an
  empirical correlation matrix of DraftKings fantasy points.
- **ML-based**: Historical regression model trained on nflverse data using
  a z-score product target to approximate correlations.

At a high level, the project:
- Ingests historical nflverse-style Parquet data.
- Computes DraftKings-style offensive fantasy points and per-player z-scores.
- Builds a pairwise training dataset using the z-score product trick.
- Trains a tree-based regression model to predict pairwise correlations.
- Applies either the simulation engine or the ML model to a Sabersim Showdown
  projections CSV to produce a player correlation matrix and writes it to an
  Excel file.

### Project layout

- `data/nfl_raw/`: nflverse Parquet files (player stats and games/schedule).
- `data/nfl_processed/`: processed player-game data with DK points and z-scores.
- `data/sabersim/`: Sabersim Showdown projections CSVs.
- `models/`: trained correlation model (`corr_model.pkl`).
- `outputs/correlations/`: Excel outputs with projections + correlation matrix.
- `src/`: Python package with all pipeline code.

### Dependencies

Use Python 3.10+ and install dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries:
- `pandas`, `numpy` for data handling.
- `scikit-learn`, `joblib` for modeling.
- `xlsxwriter` for Excel output.
- `pyarrow` for Parquet IO.
- `nfl_data_py` for downloading historical NFL data.

### Configuration

Edit `src/config.py` if needed to point to your local Parquet files and
Sabersim CSVs. By default it expects:

- Player stats Parquet: `data/nfl_raw/player_stats.parquet`
- Games/schedule Parquet: `data/nfl_raw/games.parquet`
- Sabersim CSV: `data/sabersim/NFL_2025-11-24-815pm_DK_SHOWDOWN_CAR-@-SF.csv`
- Output Excel: `outputs/correlations/showdown_corr_matrix.xlsx`

You can also adjust:
- `MIN_PLAYER_GAMES` (minimum games per player for stable z-scores).
- Season ranges for train/validation/test splits.
- Offensive positions to include.
- Simulation settings:
  - `DEFAULT_CORR_METHOD` (defaults to `"simulation"`).
  - `SIM_N_GAMES` (number of Monte Carlo simulations).
  - `SIM_RANDOM_SEED` (seed or `None` for nondeterministic).
  - `SIM_DIRICHLET_K_YARDS`, `SIM_DIRICHLET_K_RECEPTIONS`,
    `SIM_DIRICHLET_K_TDS` (how tightly simulated stat shares cluster around
    projections).

### Downloading historical NFL data

Before running the correlation pipeline, populate the raw Parquet files using
`nfl_data_py`:

```bash
python -m src.download_nfl_data --start-season 2005 --end-season 2024
```

This will create:
- `data/nfl_raw/player_stats.parquet`
- `data/nfl_raw/games.parquet`

The downloader normalizes columns so they align with `src/config.py` and
`src/data_loading.py`.

### Running the correlation pipeline (simulation-based, default)

From the project root:

```bash
python -m src.main
```

By default this will:
1. Load Sabersim projections from `config.SABERSIM_CSV`.
2. Run `SIM_N_GAMES` Monte Carlo simulations of team-level stat pools and
   allocate them to players using Dirichlet/multinomial sampling.
3. Compute the empirical correlation matrix of DK fantasy points across all
   FLEX players in that slate.
4. Write an Excel file to `config.OUTPUT_CORR_EXCEL` with:
   - Sheet `Sabersim_Projections`: FLEX-only Sabersim projections.
   - Sheet `Correlation_Matrix`: symmetric correlation matrix with
     player names as both rows and columns.

You can override the Sabersim CSV, number of simulations, and output paths:

```bash
python -m src.main \
  --sabersim-csv data/sabersim/your_showdown_file.csv \
  --n-sims 8000 \
  --output-excel outputs/correlations/your_corr_matrix.xlsx
```

### Using the historical ML model instead of simulation

If you want to use the regression model trained on historical nflverse data,
run:

```bash
python -m src.main --corr-method ml
```

On the first ML run, if no model exists at `models/corr_model.pkl`, the script
will:
1. Load historical data from `data/nfl_raw/`.
2. Compute DK offensive fantasy points and per-player z-scores.
3. Build the pairwise training dataset.
4. Train the correlation regression model and save it.

You can force retraining with:

```bash
python -m src.main --corr-method ml --retrain
```

