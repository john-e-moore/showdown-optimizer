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

### Selecting diversified lineups based on top 1% finish rate

Once `outputs/top1pct/` contains a workbook with `Lineups_Top1Pct` (from
`src.top1pct_finish_rate`), you can select a diversified subset of lineups that:

- Filters out weak lineups with `top1_pct_finish_rate < 1%`.
- Prefers higher `top1_pct_finish_rate` lineups.
- Enforces a maximum allowed player overlap between any pair of selected lineups.

From the project root:

```bash
python -m src.diversify_lineups \
  --num-lineups 50 \
  --min-top1-pct 1.0 \
  --max-overlap 4
```

This will:

- Load the most recent `.xlsx` in `outputs/top1pct/` and read the
  `Lineups_Top1Pct` sheet.
- Keep only lineups with `top1_pct_finish_rate >= min-top1-pct`.
- Represent each lineup as the set of its six players (CPT + 5 FLEX).
- Greedily select up to `--num-lineups` lineups in descending
  `top1_pct_finish_rate` (breaking ties by `lineup_projection` when available),
  only accepting a lineup if its overlap with every already-selected lineup is
  at most `--max-overlap` shared players.
- Write the selected lineups to
  `outputs/top1pct/top1pct_diversified_{num_lineups}.xlsx` with sheet
  `Lineups_Diversified`.

Typical choices:

- Use `--min-top1-pct 1.0` to discard lineups with very low modeled upside.
- Use `--max-overlap` between 3 and 5 to control how aggressively you diversify
  player combinations across the final portfolio.

### Flashback contest simulation for completed Showdown contests

You can analyze a **completed** DraftKings Showdown contest using the same
correlation engine via `src.flashback_sim`. This script:

- Loads a finished contest CSV from `data/contests/`.
- Loads matching Sabersim projections from `data/sabersim/`.
- Rebuilds a player correlation matrix from the Sabersim projections.
- Simulates correlated DK scores for all players and scores every contest lineup.
- Writes an Excel workbook under `outputs/flashback/` with:
  - `Standings`: original contest CSV.
  - `Simulation`: CPT/FLEX players, **Sim ROI**, Top 1% / Top 5% / Top 20% finish
    rates, and average DK points per lineup (plus actual points/ROI when the
    payout file is available).
  - `Entrant summary`: average metrics across entries per entrant, including
    **Avg. Sim ROI** when payouts are available.
  - `Player summary`: draft rates and performance by role (CPT vs FLEX),
    including **CPT Sim ROI** and **FLEX Sim ROI** when payouts are available.

From the project root, if you have a contest CSV in `data/contests/` and a
matching Sabersim CSV in `data/sabersim/`, you can run:

```bash
python -m src.flashback_sim
```

To specify the exact inputs and number of simulations:

```bash
python -m src.flashback_sim \
  --contest-csv data/contests/my_contest.csv \
  --sabersim-csv data/sabersim/my_slate.csv \
  --num-sims 20000 \
  --payouts-csv data/payouts/payouts-123456789.json
```

If `--contest-csv` or `--sabersim-csv` are omitted, the script will
automatically pick the most recent `.csv` in the corresponding directory.

If `--payouts-csv` is omitted, the script will first look for a DraftKings payout
JSON named `payouts-{contest_id}.json` under `data/payouts/`, where
`{contest_id}` comes from the contest standings filename (e.g.,
`contest-standings-185418998.csv` → `payouts-185418998.json`). If that file does
not exist, `src.flashback_sim` will automatically call the DraftKings payouts
endpoint
`https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json`,
save the response to `data/payouts/payouts-{contest_id}.json`, and then use it
for ROI computation. If the download fails (e.g., network error, non-200 HTTP
status, or malformed JSON), the script falls back to skipping ROI computation as
before. The **Sim ROI** values are defined as:

\[
\text{ROI} = \frac{\mathbb{E}[\text{payout}] - \text{entry fee}}{\text{entry fee}}
\]

so `0.5` means +50% ROI and negative values mean losing money on average.

### End-to-end pipeline with run_full.sh

You can run the full NFL pipeline (correlation → lineups → top1% →
diversified portfolio → DKEntries fill) with the provided helper script:

```bash
./run_full.sh PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM]
```

Where:

- `FIELD_SIZE` (optional) is the contest size used for the top 1% threshold.
- `NUM_LINEUPS` (optional) is the number of MILP-optimized lineups to generate.
- `SALARY_CAP` (optional) is the DK Showdown salary cap (default `50000`).
- `STACK_MODE` (optional) controls stacking behavior for the optimizer
  (`none` or `multi`).
- `STACK_WEIGHTS` (optional) are the multi-stack pattern weights passed through
  to `src.showdown_optimizer_main`.
- `DIVERSIFIED_NUM` (optional) is the number of diversified lineups to select.
  If provided explicitly, it is used as-is. When omitted, the script defaults
  to the number of entries in the latest `DKEntries*.csv` under
  `data/dkentries/`.

The script will:

1. Build a correlation workbook under `outputs/correlations/`.
2. Generate a lineup workbook under `outputs/lineups/`.
3. Estimate top 1% finish probabilities into `outputs/top1pct/`.
4. Select a diversified subset of `DIVERSIFIED_NUM` lineups based on
   `top1_pct_finish_rate` and player-overlap constraints.
5. Fill the latest DKEntries CSV template from `data/dkentries/` with the
   selected diversified lineups, writing a DK-upload-ready CSV (with lineup
   slots formatted as `{player_name} ({player_id})`) under `outputs/dkentries/`.

### NBA Showdown pipeline

The NBA side mirrors the NFL layout with:

- `data/nba/sabersim/`: Sabersim NBA Showdown projections CSVs.
- `data/nba/contests/`: DraftKings NBA Showdown contest CSVs.
- `data/nba/dkentries/`: NBA DKEntries templates.
- `data/nba/payouts/`: DraftKings payout JSON files.
- `outputs/nba/correlations/`: NBA correlation workbooks.
- `outputs/nba/lineups/`: NBA lineup workbooks.
- `outputs/nba/top1pct/`: NBA top 1% and diversified lineups.
- `outputs/nba/dkentries/`: Filled NBA DKEntries CSVs.
- `outputs/nba/flashback/`: NBA flashback analysis workbooks.

Key NBA entrypoints:

- Correlations: `python -m src.nba.main --sabersim-csv ... --output-excel ...`
- Optimizer: `python -m src.nba.showdown_optimizer_main --sabersim-glob ...`
- Top 1%: `python -m src.nba.top1pct_finish_rate_nba --field-size ...`
- Diversification: `python -m src.nba.diversify_lineups_nba --num-lineups ...`
- DKEntries fill: `python -m src.nba.fill_dkentries_nba`
- Flashback: `python -m src.nba.flashback_sim_nba`

For a full end-to-end NBA run (correlations → optimizer → top 1% →
diversification → DKEntries fill), use:

```bash
./run_full_nba.sh PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM]
```

