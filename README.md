## NFL Showdown Correlation Pipeline

This project implements a simulation-based pipeline for modeling player-to-player
fantasy point correlations for DraftKings NFL Showdown contests.

The core of the project is a Monte Carlo simulator that uses Sabersim
box-score projections to generate many joint game outcomes and computes an
empirical correlation matrix of DraftKings fantasy points. A separate helper
script can optionally download and normalize historical NFL data for your own
analysis, but no ML model training is required for the correlation pipeline.

### Project layout

- `data/nfl_raw/`: nflverse Parquet files (player stats and games/schedule).
- `data/sabersim/`: Sabersim Showdown projections CSVs.
- `outputs/correlations/`: Excel outputs with projections + correlation matrix.
- `src/`: Python package with all pipeline and diagnostics code.

### Dependencies

Use Python 3.10+ and install dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries:
- `pandas`, `numpy` for data handling.
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
- Offensive positions to include.
- Simulation settings:
  - `SIM_N_GAMES` (number of Monte Carlo simulations).
  - `SIM_RANDOM_SEED` (seed or `None` for nondeterministic).
  - `SIM_DIRICHLET_K_YARDS`, `SIM_DIRICHLET_K_RECEPTIONS`,
    `SIM_DIRICHLET_K_TDS` (how tightly simulated stat shares cluster around
    projections).

### Downloading historical NFL data (optional)

If you want local Parquet files of historical NFL games and player stats for
your own analysis, you can populate `data/nfl_raw/` using `nfl_data_py`:

```bash
python -m src.download_nfl_data --start-season 2005 --end-season 2024
```

This will create:
- `data/nfl_raw/player_stats.parquet`
- `data/nfl_raw/games.parquet`

The downloader normalizes columns so they align with `src/config.py`.

### Running the correlation pipeline (simulation-based)

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

### Running the lineup optimizer (MILP-based)

From the project root, after installing dependencies:

```bash
python -m src.showdown_optimizer_main \
  --sabersim-glob "data/sabersim/NFL_*.csv" \
  --num-lineups 20 \
  --salary-cap 50000
```

This will:
- Load Sabersim projections from the CSV matched by `--sabersim-glob`.
- Build valid DraftKings Showdown lineups (1 CPT, 5 FLEX, 6 distinct players).
- Maximize mean projected DK fantasy points under the given salary cap.

You can also control how many lineups are solved per MILP model using the
`--chunk-size` flag:

```bash
python -m src.showdown_optimizer_main \
  --sabersim-glob "data/sabersim/NFL_*.csv" \
  --num-lineups 5000 \
  --salary-cap 50000 \
  --chunk-size 50
```

With chunking enabled, the optimizer:
- Solves for lineups in batches of `--chunk-size`, rebuilding a fresh MILP for
  each batch to keep model size roughly constant.
- Applies a projection cap between batches so that each new batch targets
  lineups strictly below the worst projection from the previous batch.

Setting `--chunk-size` to `0` (or a negative value) disables chunking and
reverts to the original behavior that uses a single model with a growing set of
no-duplicate constraints.


### Estimating top 1% finish probabilities for lineups

After generating a correlation workbook and a lineup workbook, you can
estimate the probability that each lineup finishes in the top 1% of a
large-field DraftKings Showdown contest.

From the project root:

```bash
python -m src.top1pct_finish_rate --field-size 23529
```

This will:
- Automatically pick the most recent `.xlsx` in `outputs/lineups/` as the
  lineup workbook.
- Automatically pick the most recent `.xlsx` in `outputs/correlations/` as
  the correlation workbook.
- Simulate correlated DK outcomes for all players.
- Use ownership projections to approximate the field's score distribution in
  each simulation.
- Write an Excel file to `outputs/top1pct/` with:
  - Sheet `Lineups_Top1Pct`: original lineups plus a new column
    `top1_pct_finish_rate` (percentage of simulations where the lineup beats
    the modeled top 1% cutoff).
  - Sheet `Meta`: basic metadata including field size, number of simulations,
    and paths to the input workbooks.

You can also specify the input workbooks explicitly:

```bash
python -m src.top1pct_finish_rate \
  --field-size 23529 \
  --lineups-excel outputs/lineups/lineups_YYYYMMDD_HHMMSS.xlsx \
  --corr-excel outputs/correlations/showdown_corr_matrix.xlsx
```

The reported probabilities can be interpreted as the chance that a lineup
finishes inside the top `floor(field_size * 0.01)` entries, under the
ownership and correlation model described in
`prompts/02_top1pct_finish_rate.md`.

