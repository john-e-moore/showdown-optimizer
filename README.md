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

## End-to-end scripts

These scripts run the full optimizer → simulation → diversification → DKEntries fill loop.

### NFL – `run_full.sh`

- **Top1% (legacy)**

```bash
./run_full.sh data/nfl/sabersim/NFL_<slate>.csv 23529 150
```

- **EV ROI (contest payouts)**

```bash
./run_full.sh data/nfl/sabersim/NFL_<slate>.csv --contest-id <DK_CONTEST_ID>
```

### NBA – `run_full_nba.sh`

- **Top1% (legacy)**

```bash
./run_full_nba.sh data/nba/sabersim/NBA_<slate>.csv outputs/nba/correlations/my_corr_matrix.xlsx 9803 1000
```

- **EV ROI (contest payouts)**

```bash
./run_full_nba.sh data/nba/sabersim/NBA_<slate>.csv outputs/nba/correlations/my_corr_matrix.xlsx --contest-id <DK_CONTEST_ID>
```

Notes:
- When `--contest-id` is provided, the scripts will download DraftKings contest JSON (cached under `data/<sport>/payouts/`) and diversify by `ev_roi`.
- In EV ROI mode, DKEntries filling will also prefer higher `ev_roi` lineups when it needs to break ties during fee-aware assignment (legacy mode prefers `top1_pct_finish_rate`).
- You can optionally pass `--payouts-json <path>` to use a pre-downloaded contest JSON file instead of downloading.

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
    - Set `<= 0` to disable chunking and reuse a single growing model with
      growing no-duplicate constraints across all lineups (often faster for
      large runs when combined with CBC warm starts).
  - **`--stack-mode`**: Team stacking mode.
    - Choices: `"none"` (default), `"multi"`.
    - `"none"`: single optimization pass with generic constraints.
    - `"multi"`: splits `--num-lineups` across patterns `5|1, 4|2, 3|3, 2|4, 1|5`.
  - **`--stack-weights`**: Optional weights for multi-stack mode.
    - Example: `"5|1=0.3,4|2=0.25,3|3=0.2,2|4=0.15,1|5=0.1"`.
    - If omitted in `"multi"` mode, all stack patterns are weighted equally.
  - **`--num-workers`**: Number of worker threads used when parallelizing
    multi-stack runs across stack patterns.
    - Default: `1` (no parallelism).
  - **`--parallel-mode`**: Parallelization strategy.
    - Choices: `"none"` (default), `"by_stack_pattern"`.
    - `"by_stack_pattern"` runs each stack pattern in its own worker when
      `--stack-mode multi` and `--num-workers > 1`.
  - **`--use-warm-start`**: Enable CBC warm starts in the single-model path
    (`--chunk-size <= 0`), allowing the solver to reuse solution information
    as additional lineups are generated.
  - **`--solver-max-seconds`**: Optional per-solve time limit in seconds
    passed through to CBC.
  - **`--solver-rel-gap`**: Optional relative optimality gap for CBC
    (e.g., `0.005` for 0.5% MIP gap); allows early termination once the
    solver is “good enough” instead of fully optimal.

- **Profiling:**
  - Every optimizer run prints a machine-parsable summary line of the form:
    - `OPT_TIME seconds=<float> num_lineups_requested=<int> num_lineups_generated=<int> stack_mode=<str> chunk_size=<int> parallel_mode=<str> num_workers=<int> [solver_max_seconds=... solver_rel_gap=...]`
  - A sidecar JSON file `lineups_<timestamp>_metrics.json` is written next to
    the lineups workbook with the same high-level metrics plus per-pattern
    timings in multi-stack mode.

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

- **EV ROI example (download DK payouts + infer field size):**

```bash
python -m src.nfl.top1pct_finish_rate \
  --contest-id <DK_CONTEST_ID> \
  --field-model explicit \
  --num-sims 100000
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
  - **`--field-size`**: Total number of lineups in the contest field.
    - Required unless `--contest-id` is provided (then inferred from DK contest metadata).
  - **`--contest-id`**: DraftKings contest id. When provided, download contest payout
    info and compute per-lineup `avg_payout` and `ev_roi`.
  - **`--payouts-json`**: Optional path to a cached DraftKings contest JSON file.
  - **`--sim-batch-size`**: Simulation batch size used for streaming scoring.
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
  - When run with `--contest-id`, the `Lineups_Top1Pct` sheet includes:
    - `avg_payout`: average simulated payout (in dollars)
    - `ev_roi`: \((\text{avg_payout} - \text{entry_fee}) / \text{entry_fee}\)

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
  - **`--sort-by`**: Column to sort candidate lineups by before greedy selection.
    - Default: `top1_pct_finish_rate`.
    - Use `ev_roi` when top1% scoring was run with `--contest-id`.
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

### NFL field-style augmentation – `src.nfl.augment_lineups_with_field`

Optionally augment an optimizer lineups workbook with additional
quota-balanced field-style lineups before running top 1% scoring.

- **Example:**

```bash
python -m src.nfl.augment_lineups_with_field \
  --lineups-excel outputs/nfl/lineups/lineups_YYYYMMDD_HHMMSS.xlsx \
  --corr-excel outputs/nfl/correlations/showdown_corr_matrix.xlsx \
  --extra-lineups 500
```

- **Flags:**
  - **`--lineups-excel`** (required): Optimizer lineups workbook to augment.
  - **`--corr-excel`** (required): Correlations workbook providing
    `Sabersim_Projections` and `Correlation_Matrix` sheets.
  - **`--extra-lineups`** (required): Number of additional field-style
    CPT+FLEX lineups to generate and append.
  - **`--output-excel`**: Optional explicit path for the augmented workbook.
    - Default: sibling file with `_augmented` suffix next to the input file.
  - **`--random-seed`**: Optional RNG seed for reproducibility.

- **Output:**
  - Writes an augmented lineups workbook whose `Lineups` sheet contains both
    optimizer lineups and additional field-style lineups tagged via the
    `target_stack_pattern` column (e.g., `"field"` for augmented entries).

### NFL flashback simulation – `src.nfl.flashback_sim`

Run a flashback contest simulation for a completed NFL Showdown slate.

- **Example:**

```bash
python -m src.nfl.flashback_sim \
  --contest-csv data/nfl/contests/contest_<id>.csv \
  --sabersim-csv data/nfl/sabersim/NFL_<slate>.csv \
  --corr-excel outputs/nfl/correlations/showdown_corr_matrix.xlsx \
  --payouts-csv data/nfl/payouts/payouts_<id>.json \
  --num-sims 100000
```

- **Flags:**
  - **`--contest-csv`**: Contest standings CSV under `data/nfl/contests/`.
    - Default: most recent `.csv` in that directory.
  - **`--sabersim-csv`**: Sabersim projections CSV under `data/nfl/sabersim/`.
    - Default: most recent `.csv` in that directory.
  - **`--corr-excel`**: Correlations workbook (absolute path, or relative to `outputs/nfl/correlations/`).
    - If omitted, correlations are computed from the Sabersim projections and a timestamped workbook is written under `outputs/nfl/correlations/`.
    - Workbook must contain sheets `Sabersim_Projections` and `Correlation_Matrix`.
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
    - Default: `50`.
    - Set `<= 0` to disable chunking and reuse a single growing model with
      no-duplicate constraints across all lineups (often faster for large
      runs when combined with CBC warm starts).
  - **`--stack-mode`**: Team stacking mode.
    - Choices: `"none"` (default), `"multi"`.
  - **`--stack-weights`**: Optional weights for multi-stack mode.
    - Same semantics as NFL optimizer.
  - **`--num-workers`**: Number of worker threads used when parallelizing
    multi-stack runs across stack patterns.
    - Default: `1` (no parallelism).
  - **`--parallel-mode`**: Parallelization strategy.
    - Choices: `"none"` (default), `"by_stack_pattern"`.
    - `"by_stack_pattern"` runs each stack pattern in its own worker when
      `--stack-mode multi` and `--num-workers > 1`.
  - **`--use-warm-start`**: Enable CBC warm starts in the single-model path
    (`--chunk-size <= 0`), allowing the solver to reuse solution information
    as additional lineups are generated.
  - **`--solver-max-seconds`**: Optional per-solve time limit in seconds
    passed through to CBC.
  - **`--solver-rel-gap`**: Optional relative optimality gap for CBC
    (e.g., `0.005` for 0.5% MIP gap); allows early termination once the
    solver is “good enough” instead of fully optimal.

- **Profiling:**
  - Every optimizer run prints a machine-parsable summary line of the form:
    - `OPT_TIME seconds=<float> num_lineups_requested=<int> num_lineups_generated=<int> stack_mode=<str> chunk_size=<int> parallel_mode=<str> num_workers=<int> [solver_max_seconds=... solver_rel_gap=...]`
  - A sidecar JSON file `lineups_<timestamp>_metrics.json` is written next to
    the lineups workbook with the same high-level metrics plus per-pattern
    timings in multi-stack mode.

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
  - **`--field-size`**: Total number of lineups in the contest field.
    - Required unless `--contest-id` is provided (then inferred from DK contest metadata).
  - **`--contest-id`**: DraftKings contest id. When provided, download contest payout
    info and compute per-lineup `avg_payout` and `ev_roi`.
  - **`--payouts-json`**: Optional path to a cached DraftKings contest JSON file.
  - **`--sim-batch-size`**: Simulation batch size used for streaming scoring.
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
  - When run with `--contest-id`, the `Lineups_Top1Pct` sheet includes:
    - `avg_payout`: average simulated payout (in dollars)
    - `ev_roi`: \((\text{avg_payout} - \text{entry_fee}) / \text{entry_fee}\)

### NBA field-style augmentation – `src.nba.augment_lineups_with_field`

Optionally augment an optimizer lineups workbook with additional
quota-balanced field-style lineups before running top 1% scoring.

- **Example:**

```bash
python -m src.nba.augment_lineups_with_field \
  --lineups-excel outputs/nba/lineups/lineups_YYYYMMDD_HHMMSS.xlsx \
  --corr-excel outputs/nba/correlations/showdown_corr_matrix.xlsx \
  --extra-lineups 500
```

- **Flags:**
  - **`--lineups-excel`** (required): Optimizer lineups workbook to augment.
  - **`--corr-excel`** (required): Correlations workbook providing
    `Sabersim_Projections` and `Correlation_Matrix` sheets.
  - **`--extra-lineups`** (required): Number of additional field-style
    CPT+UTIL lineups to generate and append.
  - **`--output-excel`**: Optional explicit path for the augmented workbook.
    - Default: sibling file with `_augmented` suffix next to the input file.
  - **`--random-seed`**: Optional RNG seed for reproducibility.

- **Output:**
  - Writes an augmented lineups workbook whose `Lineups` sheet contains both
    optimizer lineups and additional field-style lineups tagged via the
    `target_stack_pattern` column (e.g., `"field"` for augmented entries).

### NBA lineup diversification – `src.nba.diversify_lineups_nba`

Select a diversified subset of NBA Showdown lineups based on top1% and overlap.

– **Example:**

```bash
python -m src.nba.diversify_lineups_nba \
  --num-lineups 150 \
  --min-top1-pct 1.0 \
  --max-overlap 4 \
  --max-flex-overlap 4 \
  --cpt-field-cap-multiplier 1.5
```

– **Flags:**
  - **`--num-lineups`** (required): Number of lineups to select.
  - **`--min-top1-pct`**: Minimum `top1_pct_finish_rate` in percent.
    - Default: `1.0`.
  - **`--sort-by`**: Column to sort candidate lineups by before greedy selection.
    - Default: `top1_pct_finish_rate`.
    - Use `ev_roi` when top1% scoring was run with `--contest-id`.
  - **`--max-overlap`**: Maximum overlapping players allowed (0–6).
    - Default: `4`.
  - **`--max-flex-overlap`**: Optional maximum overlapping FLEX/UTIL players allowed
    between any pair of selected lineups (0–5). If omitted, only total overlap
    (`--max-overlap`) is enforced.
  - **`--cpt-field-cap-multiplier`**: Multiple of projected field CPT ownership
    used as a max CPT share cap within the diversified set. Set `<= 0` to
    disable CPT caps.
    - Default: `2.0`.
  - **`--top1pct-excel`**: NBA top1% workbook under `outputs/nba/top1pct/`.
    - Default: most recent `.xlsx` in that directory, or a run-scoped workbook
      under `--output-dir` when provided.
  - **`--lineups-excel`**: Optional lineups workbook whose Projections sheet
    provides projected field ownership. If omitted, a run-scoped `lineups_*.xlsx`
    under the output run directory is preferred when available; otherwise, the
    latest `.xlsx` under `outputs/nba/lineups/` is used.
  - **`--output-dir`**: Optional directory in which to write `diversified.csv`
    and `ownership.csv` sidecar files for this diversification run.

- **Output:**
  - Writes a diversified lineups workbook under `outputs/nba/top1pct/`.

### NBA flashback simulation – `src.nba.flashback_sim_nba`

Run a flashback contest simulation for a completed NBA Showdown slate.

- **Example:**

```bash
python -m src.nba.flashback_sim_nba \
  --contest-csv data/nba/contests/contest_<id>.csv \
  --sabersim-csv data/nba/sabersim/NBA_<slate>.csv \
  --corr-excel data/nba/correlations/NBA_<slate>_corr_matrix.xlsx \
  --payouts-csv data/nba/payouts/payouts_<id>.json \
  --num-sims 100000
```

- **Flags:**
  - **`--contest-csv`**: Contest standings CSV under `data/nba/contests/`.
    - Default: most recent `.csv` in that directory.
  - **`--sabersim-csv`**: Sabersim projections CSV under `data/nba/sabersim/`.
    - Default: most recent `.csv` in that directory.
  - **`--corr-excel`**: Correlations Excel workbook under `data/nba/correlations/`.
    - Default: most recent `.xlsx` in that directory.
    - Must contain sheets `Sabersim_Projections` and `Correlation_Matrix`.
    - If missing/unreadable, the flashback sim errors (generate the workbook or pass an explicit path).
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

