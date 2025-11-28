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

You can optionally enable **multi-stack mode** to force a mix of team stacks
across the generated lineups. For example, to generate 1000 lineups split
equally across 5|1, 4|2, 3|3, 2|4, and 1|5 team stacks:

```bash
python -m src.showdown_optimizer_main \
  --sabersim-glob "data/sabersim/NFL_*.csv" \
  --num-lineups 1000 \
  --salary-cap 50000 \
  --chunk-size 50 \
  --stack-mode multi
```

In `--stack-mode multi`, the optimizer:
- Assumes a standard two-team Showdown slate.
- Splits `--num-lineups` across the five stack patterns
  (`5|1`, `4|2`, `3|3`, `2|4`, `1|5`) according to configurable weights.
- Runs the MILP once per pattern with an additional team-level stack constraint.
- Deduplicates lineups across all runs and writes a **single** Excel workbook
  as before.

You can customize the relative frequency of each stack pattern using
`--stack-weights`. For example:

```bash
python -m src.showdown_optimizer_main \
  --sabersim-glob "data/sabersim/NFL_*.csv" \
  --num-lineups 1000 \
  --salary-cap 50000 \
  --chunk-size 50 \
  --stack-mode multi \
  --stack-weights "5|1=0.4,4|2=0.3,3|3=0.2,2|4=0.1,1|5=0.0"
```

`--stack-weights` values are normalized to sum to 1. Patterns omitted or set to
0 receive no lineups (other than potential rounding leftovers, which are
distributed among patterns with positive weights).

In the resulting `Lineups` sheet, an extra column
`target_stack_pattern` records which stack configuration/run produced each
lineup (e.g., `5|1_KC-heavy`, `3|3`, `2|4_SF-heavy`). The existing `stack`
column still reflects the actual team counts (e.g., `5|1`, `4|2`, `3|3`) and is
used by downstream tools like `src.top1pct_finish_rate.py`.


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

You can optionally tune the calibration of the field distribution:

- `--field-var-shrink`: multiplicative shrinkage on the modeled field variance
  (default `0.7`). Values closer to `0` pull the top 1% cutoff closer to the
  mean field score; values near `1` use the raw variance.
- `--field-z`: z-score applied to the (shrunken) field standard deviation
  (default `2.0`, slightly below the canonical 99th percentile `2.326`).
- `--flex-var-factor`: effective variance factor for the aggregate FLEX
  component (default `3.5`, vs. `5.0` for five independent FLEX slots).

These knobs are intended to produce **realistic relative rankings** of lineups
by top-1% finish probability (e.g., best lineups a few percent, weakest near
zero), not perfectly calibrated absolute probabilities.

