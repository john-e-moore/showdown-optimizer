## Showdown Optimizer – Overview

This repository is organized around a small number of Python packages under `src/`
to clearly separate shared logic from sport-specific code:

- **`src/shared/` – sport-agnostic cores and utilities**
  - `config_base.py`: project root, data/outputs/diagnostics/model directories, and
    helpers like `get_data_dir_for_sport`.
  - `optimizer_core.py`: generic Showdown MILP optimizer (no NFL/NBA assumptions).
  - `lineup_optimizer.py`: thin adapter that loads Sabersim CSVs into the shared
    optimizer core.
  - `dkentries_core.py`: shared logic for working with DraftKings DKEntries CSVs
    (resolving templates, mapping Name/ID/roster positions, fee-aware lineup
    assignment, etc.).
  - `top1pct_core.py`: core engine for estimating top 1% finish rates given
    correlation matrices, ownership, and projections.
  - `flashback_core.py`: core engine for flashback simulations on completed
    contests (parsing exported lineups, simulating correlated scores, computing
    ROI and finish rates).

- **`src/nfl/` – NFL-specific configuration and CLIs**
  - `config.py`: NFL paths (e.g. `data/nfl`, `outputs/nfl`), column names, and
    simulation hyperparameters.
  - `data_loading.py`, `diagnostics.py`: nflverse-style Parquet loaders and
    diagnostics that use the NFL config.
  - `build_corr_matrix_from_projections.py`, `simulation_corr.py`: NFL Sabersim
    loader and Monte Carlo correlation engine.
  - **Primary CLIs (used via `python -m src.nfl.<module>`):**
    - `main`: build correlations from Sabersim.
    - `showdown_optimizer_main`: NFL Showdown optimizer.
    - `top1pct_finish_rate`: top 1% finish-rate estimation for NFL.
    - `diversify_lineups`: select diversified NFL lineups from top1% output.
    - `flashback_sim`: flashback contest simulation for completed NFL slates.
    - `fill_dkentries`: fill DKEntries templates with diversified NFL lineups.
    - `download_nfl_data`: download and normalize historical NFL data.

- **`src/nba/` – NBA-specific configuration and CLIs**
  - `config.py`: NBA paths (e.g. `data/nba`, `outputs/nba`) and any NBA-specific
    knobs.
  - `sabersim_parser.py`, `simulation_corr.py`, `stat_sim.py`: NBA Sabersim
    parsing and correlation/stat simulation.
  - **Primary CLIs (used via `python -m src.nba.<module>`):**
    - `main`: NBA correlation pipeline.
    - `showdown_optimizer_main`: NBA Showdown optimizer.
    - `top1pct_finish_rate_nba`: NBA top 1% finish-rate estimation.
    - `diversify_lineups_nba`: select diversified NBA lineups from top1% output.
    - `flashback_sim_nba`: flashback contest simulation for NBA.
    - `fill_dkentries_nba`: fill DKEntries templates with diversified NBA lineups.

In day-to-day use, prefer the sport-specific entrypoints under `src/nfl` and
`src/nba` (e.g. `python -m src.nfl.showdown_optimizer_main ...`) while treating
`src/shared` as an implementation detail that you rarely need to call directly.

The `run_full.sh` script wires many of these steps together into an end‑to‑end
pipeline, but each module can also be run individually as documented below.

---

## Environment and Installation

- **Python**: Use a recent Python 3 (3.10+ recommended).
- **Install dependencies** from the project root:

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below assume you run them **from the project root**
(`showdown-optimizer/`) so that `src/` is importable.

---

## NFL Modules – How to Run (with Flags)

All NFL commands use the form:

```bash
python -m src.nfl.<module> [FLAGS...]
```

### NFL correlation pipeline – `src.nfl.main`

Build a player correlation matrix from a Sabersim Showdown CSV.

- **Command:**

```bash
python -m src.nfl.main \
  --sabersim-csv data/nfl/sabersim/NFL_<slate>.csv \
  --output-excel outputs/nfl/correlations/showdown_corr_matrix.xlsx \
  --n-sims 100000
```

- **Flags:**
  - **`--sabersim-csv`**: Path to Sabersim Showdown projections CSV.
    - Default: `src.nfl.config.SABERSIM_CSV` (typically under `data/nfl/sabersim/`).
  - **`--output-excel`**: Output Excel with projections + correlation matrix.
    - Default: `src.nfl.config.OUTPUT_CORR_EXCEL` (under `outputs/nfl/correlations/`).
  - **`--n-sims`**: Number of Monte Carlo simulations to run.
    - Default: `src.nfl.config.SIM_N_GAMES`.

### NFL Showdown optimizer – `src.nfl.showdown_optimizer_main`

Generate optimized NFL Showdown lineups from a Sabersim CSV.

- **Basic example:**

```bash
python -m src.nfl.showdown_optimizer_main \
  --sabersim-glob "data/nfl/sabersim/NFL_*.csv" \
  --num-lineups 50 \
  --salary-cap 50000
```

- **Flags:**
  - **`--sabersim-glob`**: Glob/path to Sabersim Showdown projections CSV.
    - Must resolve to **exactly one** file.
    - Default: `str(config.SABERSIM_DIR / "NFL_*.csv")`.
  - **`--num-lineups`**: Number of lineups to generate.
    - Default: `20`.
  - **`--salary-cap`**: DraftKings Showdown salary cap.
    - Default: `50000`.
  - **`--chunk-size`**: Lineups solved per MILP chunk when generating many lineups.
    - Default: `50`.
    - Set `<= 0` to disable chunking and use a single growing model.
  - **`--stack-mode`**: Team stacking mode.
    - Choices: `"none"` (default), `"multi"`.
    - `"none"`: single optimization pass with generic constraints.
    - `"multi"`: splits `--num-lineups` across patterns `5|1, 4|2, 3|3, 2|4, 1|5`.
  - **`--stack-weights`**: Optional weights for multi-stack mode.
    - Example: `"5|1=0.3,4|2=0.25,3|3=0.2,2|4=0.15,1|5=0.1"`.
    - If omitted in `"multi"` mode, all stack patterns are weighted equally.

- **Output:**
  - Writes an Excel workbook under `outputs/nfl/lineups/lineups_<timestamp>.xlsx`
    with sheets for projections, ownership, exposure, and lineups.

### NFL top 1% finish rate – `src.nfl.top1pct_finish_rate`

Estimate top 1% finish probabilities for a set of lineups using correlations +
ownership.

- **Minimal example (auto-resolve latest inputs):**

```bash
python -m src.nfl.top1pct_finish_rate --field-size 23529
```

- **Explicit input example:**

```bash
python -m src.nfl.top1pct_finish_rate \
  --field-size 23529 \
  --lineups-excel outputs/nfl/lineups/lineups_YYYYMMDD_HHMMSS.xlsx \
  --corr-excel outputs/nfl/correlations/showdown_corr_matrix.xlsx \
  --num-sims 100000 \
  --field-var-shrink 0.7 \
  --field-z 2.0 \
  --flex-var-factor 3.5
```

- **Flags:**
  - **`--field-size`** (required): Total number of lineups in the contest field.
  - **`--lineups-excel`**: Path to lineups workbook under `outputs/nfl/lineups/`.
    - Default: most recent `.xlsx` in that directory.
  - **`--corr-excel`**: Path to correlations workbook under
    `outputs/nfl/correlations/`.
    - Default: most recent `.xlsx` in that directory.
  - **`--num-sims`**: Number of Monte Carlo simulations.
    - Default: `100000`.
  - **`--field-var-shrink`**: Shrinkage factor for modeled field variance
    (\(0 < \text{value} \le 1\)).
    - Default: `0.7`.
  - **`--field-z`**: Z-score for upper tail of field score distribution.
    - Default: `2.0`.
  - **`--flex-var-factor`**: Effective variance factor for aggregate FLEX
    component (<= 5.0).
    - Default: `3.5`.
  - **`--random-seed`**: Optional random seed for reproducibility.
    - Default: `None` (falls back to `config.SIM_RANDOM_SEED`).

- **Output:**
  - Writes a top1% workbook under `outputs/nfl/top1pct/` (exact path is printed).

### NFL lineup diversification – `src.nfl.diversify_lineups`

Select a diversified subset of top1%-rated lineups, enforcing a max overlap.

- **Example:**

```bash
python -m src.nfl.diversify_lineups \
  --num-lineups 150 \
  --min-top1-pct 1.0 \
  --max-overlap 4
```

- **Flags:**
  - **`--num-lineups`** (required): Number of lineups to select.
  - **`--min-top1-pct`**: Minimum `top1_pct_finish_rate` (in percent)
    required to keep a lineup.
    - Default: `1.0`.
  - **`--max-overlap`**: Maximum overlapping players allowed between any pair
    of selected lineups (\(0\)–\(6\) for Showdown).
    - Default: `4`.
  - **`--max-flex-overlap`**: Optional maximum number of overlapping FLEX
    players allowed between any pair of selected lineups (\(0\)–\(5\)).
    - Default: omitted (no FLEX-specific cap; only `--max-overlap` is used).
  - **`--top1pct-excel`**: Path to a top1% workbook under `outputs/nfl/top1pct/`.
    - Default: most recent `.xlsx` in that directory.

- **Output:**
  - Writes a diversified lineups workbook under `outputs/nfl/top1pct/`
    (sheet `Lineups_Diversified`).

### NFL flashback simulation – `src.nfl.flashback_sim`

Run a flashback contest simulation for a completed NFL Showdown slate.

- **Example:**

```bash
python -m src.nfl.flashback_sim \
  --contest-csv data/nfl/contests/contest_<id>.csv \
  --sabersim-csv data/nfl/sabersim/NFL_<slate>.csv \
  --payouts-csv data/nfl/payouts/payouts_<id>.json \
  --num-sims 100000
```

- **Flags:**
  - **`--contest-csv`**: Contest standings CSV under `data/nfl/contests/`.
    - Default: most recent `.csv` in that directory.
  - **`--sabersim-csv`**: Sabersim projections CSV under `data/nfl/sabersim/`.
    - Default: most recent `.csv` in that directory.
  - **`--payouts-csv`**: DraftKings payout JSON under `data/nfl/payouts/`.
    - Despite the name, this flag expects a JSON file (`payouts-*.json`).
    - Default: inferred from contest filename or downloaded if missing.
  - **`--num-sims`**: Number of Monte Carlo simulations.
    - Default: `100000`.
  - **`--random-seed`**: Optional random seed.
    - Default: `None` (falls back to `config.SIM_RANDOM_SEED`).

- **Output:**
  - Writes flashback outputs under `outputs/nfl/flashback/` (exact paths printed).

### NFL DKEntries filler – `src.nfl.fill_dkentries`

Fill a DKEntries CSV template with diversified lineups and write an
upload‑ready file plus exposure summaries.

- **Example:**

```bash
python -m src.nfl.fill_dkentries \
  --dkentries-csv data/nfl/dkentries/DKEntries_<slate>.csv \
  --diversified-excel outputs/nfl/top1pct/top1pct_YYYYMMDD_HHMMSS.xlsx
```

- **Flags:**
  - **`--dkentries-csv`**: Explicit DKEntries CSV path.
    - Default: latest `DKEntries*.csv` under `data/nfl/dkentries/`.
  - **`--diversified-excel`**: Top1% workbook containing `Lineups_Diversified`.
    - Default: latest `.xlsx` under `outputs/nfl/top1pct/`.
  - **`--output-csv`**: Explicit output CSV path.
    - Default: timestamped folder under `outputs/nfl/dkentries/`.
  - **`--cpt-own-weight`**: Weight on CPT dollar-exposure matching vs lineup
    strength. Higher values make realized CPT dollar exposure more closely
    track projected field CPT ownership.
    - Default: `1.0`.
  - **`--flex-own-weight`**: Weight on FLEX exposure diversification penalty.
    Set `> 0` to encourage FLEX dollar exposure to track projected field FLEX
    ownership.
    - Default: `0.0` (disabled).
  - **`--max-flex-overlap`**: Optional maximum number of overlapping FLEX
    players allowed between any pair of assigned lineups. If omitted, no hard
    FLEX overlap cap is enforced at assignment time.

- **Output:**
  - Creates a timestamped directory under `outputs/nfl/dkentries/<YYYYMMDD_HHMMSS>/`
    containing:
    - `dkentries.csv`: DK upload‑ready file.
    - `diversified.csv`: diversified lineups used.
    - `ownership.csv`: realized ownership summary with projected field
      ownership (`field_ownership`) and realized lineup/dollar exposure
      (`lineup_exposure`, `dollar_exposure`) by player and role.

### NFL historical data downloader – `src.nfl.download_nfl_data`

Download historical NFL weekly stats and game schedules via `nfl_data_py` and
write normalized Parquet files.

- **Example:**

```bash
python -m src.nfl.download_nfl_data \
  --start-season 2005 \
  --end-season 2024 \
  --overwrite
```

- **Flags:**
  - **`--start-season`**: First season to include.
    - Default: `2005`.
  - **`--end-season`**: Last season to include (inclusive).
    - Default: current year.
  - **`--overwrite`**: If present, overwrite existing Parquet outputs.

- **Output:**
  - Writes Parquet files to locations configured in `src.nfl.config`, typically
    under `data/nfl/`.

---

## NBA Modules – How to Run (with Flags)

All NBA commands use the form:

```bash
python -m src.nba.<module> [FLAGS...]
```

### NBA correlation pipeline – `src.nba.main`

Build a player correlation matrix from an NBA Sabersim Showdown CSV.

- **Example:**

```bash
python -m src.nba.main \
  --sabersim-csv data/nba/sabersim/NBA_<slate>.csv \
  --output-excel outputs/nba/correlations/showdown_corr_matrix.xlsx \
  --n-sims 100000
```

- **Flags:**
  - **`--sabersim-csv`**: Path to NBA Sabersim projections CSV.
    - Default: `src.nba.config.SABERSIM_CSV`.
  - **`--output-excel`**: Output Excel with projections + correlation matrix.
    - Default: `src.nba.config.OUTPUT_CORR_EXCEL`.
  - **`--n-sims`**: Number of Monte Carlo simulations.
    - Default: `src.nba.config.SIM_N_GAMES`.

### NBA Showdown optimizer – `src.nba.showdown_optimizer_main`

Generate optimized NBA Showdown lineups from a Sabersim CSV.

- **Example:**

```bash
python -m src.nba.showdown_optimizer_main \
  --sabersim-glob "data/nba/sabersim/NBA_*.csv" \
  --num-lineups 50 \
  --salary-cap 50000 \
  --stack-mode multi \
  --stack-weights "5|1=0.4,4|2=0.3,3|3=0.2,2|4=0.1"
```

- **Flags:**
  - **`--sabersim-glob`**: Glob/path to NBA Sabersim Showdown CSV.
    - Must resolve to **exactly one** file.
    - Default: `config.SABERSIM_CSV`.
  - **`--num-lineups`**: Number of lineups to generate.
    - Default: `20`.
  - **`--salary-cap`**: DraftKings Showdown salary cap.
    - Default: `50000`.
  - **`--chunk-size`**: Lineups solved per MILP chunk.
    - Default: `50`; set `<= 0` to disable chunking.
  - **`--stack-mode`**: Team stacking mode.
    - Choices: `"none"` (default), `"multi"`.
  - **`--stack-weights`**: Optional weights for multi-stack mode.
    - Same semantics as NFL optimizer.

- **Output:**
  - Writes an Excel workbook under `outputs/nba/lineups/lineups_<timestamp>.xlsx`.

### NBA top 1% finish rate – `src.nba.top1pct_finish_rate_nba`

Estimate top 1% finish probabilities for NBA Showdown lineups.

- **Example:**

```bash
python -m src.nba.top1pct_finish_rate_nba \
  --field-size 23529 \
  --num-sims 100000
```

- **Flags:**
  - **`--field-size`** (required): Total number of lineups in the contest field.
  - **`--lineups-excel`**: Lineups workbook under `outputs/nba/lineups/`.
    - Default: most recent `.xlsx` there.
  - **`--corr-excel`**: Correlations workbook under `outputs/nba/correlations/`.
    - Default: most recent `.xlsx` there.
  - **`--num-sims`**: Number of Monte Carlo simulations.
    - Default: `100000`.
  - **`--field-var-shrink`**: Shrinkage factor for field variance.
    - Default: `0.7`.
  - **`--field-z`**: Z-score for upper tail of field score distribution.
    - Default: `2.0`.
  - **`--flex-var-factor`**: Effective variance factor for aggregate non‑CPT
    component.
    - Default: `3.5`.
  - **`--random-seed`**: Optional random seed.
    - Default: `None` (falls back to `config.SIM_RANDOM_SEED`).

- **Output:**
  - Writes a top1% workbook under `outputs/nba/top1pct/`.

### NBA lineup diversification – `src.nba.diversify_lineups_nba`

Select a diversified subset of NBA Showdown lineups based on top1% and overlap.

- **Example:**

```bash
python -m src.nba.diversify_lineups_nba \
  --num-lineups 150 \
  --min-top1-pct 1.0 \
  --max-overlap 4
```

- **Flags:**
  - **`--num-lineups`** (required): Number of lineups to select.
  - **`--min-top1-pct`**: Minimum `top1_pct_finish_rate` in percent.
    - Default: `1.0`.
  - **`--max-overlap`**: Maximum overlapping players allowed (0–6).
    - Default: `4`.
  - **`--top1pct-excel`**: NBA top1% workbook under `outputs/nba/top1pct/`.
    - Default: most recent `.xlsx` in that directory.

- **Output:**
  - Writes a diversified lineups workbook under `outputs/nba/top1pct/`.

### NBA flashback simulation – `src.nba.flashback_sim_nba`

Run a flashback contest simulation for a completed NBA Showdown slate.

- **Example:**

```bash
python -m src.nba.flashback_sim_nba \
  --contest-csv data/nba/contests/contest_<id>.csv \
  --sabersim-csv data/nba/sabersim/NBA_<slate>.csv \
  --payouts-csv data/nba/payouts/payouts_<id>.json \
  --num-sims 100000
```

- **Flags:**
  - **`--contest-csv`**: Contest standings CSV under `data/nba/contests/`.
    - Default: most recent `.csv` in that directory.
  - **`--sabersim-csv`**: Sabersim projections CSV under `data/nba/sabersim/`.
    - Default: most recent `.csv` in that directory.
  - **`--payouts-csv`**: DraftKings payout JSON under `data/nba/payouts/`.
    - Same semantics as NFL flashback script.
  - **`--num-sims`**: Number of Monte Carlo simulations.
    - Default: `100000`.
  - **`--random-seed`**: Optional random seed.
    - Default: `None` (falls back to `config.SIM_RANDOM_SEED`).

- **Output:**
  - Writes flashback outputs under `outputs/nba/flashback/`.

### NBA DKEntries filler – `src.nba.fill_dkentries_nba`

Fill an NBA DKEntries CSV with diversified lineups and write an upload‑ready file
plus ownership summaries.

- **Example:**

```bash
python -m src.nba.fill_dkentries_nba \
  --dkentries-csv data/nba/dkentries/DKEntries_<slate>.csv \
  --diversified-excel outputs/nba/top1pct/top1pct_YYYYMMDD_HHMMSS.xlsx
```

- **Flags:**
  - **`--dkentries-csv`**: Explicit NBA DKEntries CSV path.
    - Default: latest `DKEntries*.csv` under `data/nba/dkentries/`.
  - **`--diversified-excel`**: Top1% workbook containing `Lineups_Diversified`.
    - Default: latest `.xlsx` under `outputs/nba/top1pct/`.
  - **`--output-csv`**: Explicit output CSV path.
    - Default: timestamped folder under `outputs/nba/dkentries/`.

- **Output:**
  - Creates a timestamped directory under `outputs/nba/dkentries/<YYYYMMDD_HHMMSS>/`
    containing:
    - `dkentries.csv`
    - `diversified.csv`
    - `ownership.csv`

