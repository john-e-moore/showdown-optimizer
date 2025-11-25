## NFL Showdown Correlation Pipeline

This project implements the first-stage pipeline for modeling player-to-player
fantasy point correlations for DraftKings NFL Showdown contests.

The pipeline:
- Ingests historical nflverse-style Parquet data.
- Computes DraftKings-style offensive fantasy points and per-player z-scores.
- Builds a pairwise training dataset using the z-score product trick.
- Trains a tree-based regression model to predict pairwise correlations.
- Applies the model to a Sabersim Showdown projections CSV to produce a
  player correlation matrix and writes it to an Excel file.

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

### Running the correlation pipeline

From the project root:

```bash
python -m src.main
```

On the first run, if no model exists at `models/corr_model.pkl`, the script
will:
1. Load historical data from `data/nfl_raw/`.
2. Compute DK offensive fantasy points and per-player z-scores.
3. Build the pairwise training dataset.
4. Train the correlation regression model and save it.

Then it will:
1. Load Sabersim projections from `config.SABERSIM_CSV`.
2. Build a correlation matrix across all FLEX players in that slate.
3. Write an Excel file to `config.OUTPUT_CORR_EXCEL` with:
   - Sheet `Sabersim_Projections`: FLEX-only Sabersim projections.
   - Sheet `Correlation_Matrix`: symmetric correlation matrix with
     player names as both rows and columns.

You can force retraining with:

```bash
python -m src.main --retrain
```

You can also override the Sabersim CSV and output paths:

```bash
python -m src.main \
  --sabersim-csv data/sabersim/your_showdown_file.csv \
  --output-excel outputs/correlations/your_corr_matrix.xlsx
```



