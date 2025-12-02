<!-- 0f1d0d6f-c004-4184-bca2-9461a3c33a39 df0eeedf-93a2-4987-9b19-6fc3519f117f -->
# Add NBA Showdown support alongside NFL

## High-level goals

- **Keep the existing NFL pipeline working as-is**, while adding NBA as a parallel sport.
- **Refactor the codebase into shared vs sport-specific modules**, minimizing duplication.
- **First milestone**: support building an NBA correlation matrix from a Sabersim NBA projections CSV under `data/nba/sabersim/`.
- **Later milestones**: extend the NBA side to cover lineup optimization, top 1% simulation, diversification, DKEntries filling, and flashback analysis.

## Step 1: Introduce sport-aware directory layout

- **Data directories**
- Create `data/nfl/` and `data/nba/` (with subdirs as needed, e.g. `data/nfl/sabersim/`, `data/nba/sabersim/`, `data/nfl/contests/`, etc.).
- Move existing NFL data from current locations (e.g. `data/sabersim/`, `data/contests/`, `data/nfl_raw/`, `data/dkentries/`) into `data/nfl/...` and update any absolute/relative paths in the code and README.
- **Outputs and diagnostics**
- Create `outputs/nfl/` and `outputs/nba/` with sport-specific subfolders mirroring current usage, e.g.:
- `outputs/nfl/correlations/`, `outputs/nfl/lineups/`, `outputs/nfl/top1pct/`, `outputs/nfl/flashback/`, `outputs/nfl/dkentries/`.
- `outputs/nba/correlations/`, `outputs/nba/lineups/`, `outputs/nba/top1pct/`, `outputs/nba/flashback/`, `outputs/nba/dkentries/`.
- If you use a `diagnostics/` tree, mirror this convention with `diagnostics/nfl/` and `diagnostics/nba/`.
- **Keep the top-level stable**
- Preserve the existing top-level folders (`data/`, `outputs/`, `diagnostics/`, `src/`, `run_full.sh`, etc.), only introducing `nfl/` and `nba/` *inside* those domain folders.

## Step 2: Restructure `src/` into shared and sport packages

- **Create shared and sport-specific packages**
- Add `src/shared/` for completely sport-agnostic code.
- Add `src/nfl/` and `src/nba/` for sport-specific logic, configs, and CLIs.
- **Identify and move shared components** (mostly by inspecting `src/build_corr_matrix_from_projections.py` and related modules):
- Simulation engine (Dirichlet / multinomial sampling, Monte Carlo loops, random seeding helpers).
- Generic correlation matrix computation (converting simulated samples into a correlation matrix and DataFrame/Excel outputs).
- Generic utility functions (e.g., date/time helpers, logging, common validation, common CLI argument parsing) that don’t intrinsically depend on NFL.
- Excel/IO helpers used by multiple scripts (writing workbooks, auto-selecting latest files by pattern).
- **Leave NFL wiring in an NFL-specific layer**
- Create `src/nfl/corr_pipeline.py` (or similar) that:
- Parses NFL Sabersim projections.
- Calls the shared simulation/correlation functions with NFL-specific configuration (positions, team IDs, stat model knobs, path conventions).
- Writes to `outputs/nfl/correlations/...` by default.
- Update any NFL entrypoint modules (e.g. `src/main.py` if it is NFL-specific) to live under `src/nfl/` or to clearly delegate to the new `src/nfl/` pipeline functions.

## Step 3: Introduce shared + sport-specific configuration

- **Shared base config**
- Add something like `src/shared/config_base.py` with sport-agnostic settings and helpers:
- Common simulation parameters (e.g. `SIM_N_GAMES`, `SIM_RANDOM_SEED` defaults).
- Base directory helpers (e.g. functions that build paths under `data/{sport}/...` and `outputs/{sport}/...`).
- Any generic constants used by both sports.
- **NFL config module**
- Add/rename `src/nfl/config.py` to focus on NFL-specific parameters:
- NFL default Sabersim glob/path (now under `data/nfl/sabersim/`).
- NFL stat categories and their Dirichlet `k` parameters.
- NFL positional set and role filters (CPT/FLEX, QB/RB/WR/TE/K/DEF, etc.).
- Default NFL output paths (e.g. `outputs/nfl/correlations/showdown_corr_matrix.xlsx`).
- Update existing NFL scripts to import from this module and from `shared.config_base` instead of the old monolithic `config.py`.
- **NBA config module**
- Create `src/nba/config.py` that mirrors the NFL config but with NBA specifics:
- NBA default Sabersim glob/path (under `data/nba/sabersim/`).
- NBA stat categories and any initial modeling assumptions (you can start with a simple model and refine later).
- NBA positional set/filters (e.g. G/F/C, roster constraints for Showdown CPT/FLEX if they differ from NFL).
- NBA default output paths (e.g. `outputs/nba/correlations/showdown_corr_matrix.xlsx`).

## Step 4: Implement NBA Sabersim parsing and data model

- **Create an NBA Sabersim parser**
- Add `src/nba/sabersim_parser.py` (or similarly named module) that:
- Reads an NBA Sabersim Showdown projections CSV (from `data/nba/sabersim/`).
- Normalizes column names and types into a sport-agnostic internal schema expected by the shared simulation (e.g. player ID, name, team, opponent, role (CPT/FLEX), projected fantasy points, salary, position).
- Handles any NBA-specific quirks (e.g. multi-position eligibility, different column naming conventions).
- **Define an internal NBA player model**
- Create NBA-specific helper functions (in `src/nba/model_utils.py`, for example) that:
- Map raw Sabersim rows into an internal `Player` dataclass or similar structure used by the shared engine.
- Determine CPT vs FLEX eligibility and any position-based filters for correlation (e.g. which positions to include).

## Step 5: Wire NBA into the shared correlation engine

- **Create an NBA correlation entrypoint module**
- Add `src/nba/build_corr_matrix.py` (or `src/nba/main.py`) that:
- Parses command-line args (Sabersim path, number of simulations, output path, RNG seed, etc.) mirroring the existing NFL CLI interface.
- Uses `src/nba/config.py` defaults when arguments are omitted.
- Calls the NBA Sabersim parser to produce normalized player inputs.
- Invokes shared simulation/correlation functions from `src/shared/`.
- Writes the resulting Excel output under `outputs/nba/correlations/`.
- **Align CLI behavior between NFL and NBA**
- Ensure the NBA entrypoint accepts essentially the same flags and options as the NFL correlation script (so your workflow is parallel, just with a different sport module and default paths).

## Step 6: Verify NBA correlation pipeline end-to-end

- **Manual test with a real NBA Sabersim CSV**
- After you upload an NBA projections CSV to `data/nba/sabersim/`, run the NBA correlation entrypoint (e.g. `python -m src.nba.build_corr_matrix ...`).
- Confirm that:
- The script reads the NBA CSV successfully and recognizes players/columns.
- The Monte Carlo simulation runs without errors at a modest `SIM_N_GAMES` (e.g. 1,000–5,000) to start.
- An Excel file appears under `outputs/nba/correlations/` with the expected sheets and structure (mirroring the NFL version but with NBA players).
- **Sanity-check outputs**
- Spot-check a few players/teams to ensure projections and correlations look reasonable (no obviously broken scaling, NaNs, or empty matrices).
- Fix any sport-specific assumptions that don’t translate well from NFL.

## Step 7: Extend NBA to the rest of the pipeline (future milestones)

- **Lineup optimizer for NBA**
- Extract shared MILP/optimizer code into `src/shared/optimizer_core.py`.
- Create `src/nfl/showdown_optimizer_main.py` and `src/nba/showdown_optimizer_main.py` that each:
- Parse their sport’s Sabersim projections.
- Apply sport-specific roster rules (NBA Showdown lineup constraints, multi-position eligibility rules, etc.).
- Use shared optimizer core to generate lineups to `outputs/{sport}/lineups/`.
- **Top 1% finish simulation & diversification for NBA**
- Move sport-agnostic pieces of `src.top1pct_finish_rate` and `src.diversify_lineups` into `src/shared/`.
- Add NBA-aware wrappers (or extend existing CLIs with a `--sport nba` flag) to:
- Use the correct correlation/lineups workbooks under `outputs/nba/...`.
- Write NBA results to `outputs/nba/top1pct/` and `outputs/nba/top1pct_diversified_*.xlsx`.
- **DKEntries filling and flashback analysis for NBA**
- Mirror NFL’s DKEntries utilities in an NBA context (either via shared code with a sport flag or separate thin NBA wrappers).
- Add NBA-oriented flashback scripts that:
- Read NBA contest CSVs from `data/nba/contests/`.
- Use NBA correlations and projections to simulate outcomes and write `outputs/nba/flashback/` workbooks.

## Step 8: CLI scripts and documentation

- **Shell entrypoints**
- Keep the existing `run_full.sh` semantics for NFL.
- Add a parallel `run_full_nba.sh` (or a unified script that accepts a `SPORT` argument) that:
- Points to `data/nba/sabersim/...` by default.
- Calls `src.nba.build_corr_matrix`, `src.nba.showdown_optimizer_main`, `src.nba.top1pct_finish_rate`, `src.nba.diversify_lineups`, and NBA DKEntries filler when those steps exist.
- **Update `README.md`**
- Document the new directory structure (`data/nfl/...` vs `data/nba/...`, `outputs/nfl/...` vs `outputs/nba/...`).
- Add a short “NBA Showdown” section mirroring the NFL instructions:
- Where to put NBA Sabersim CSVs.
- How to run the NBA correlation-only pipeline initially (and later the full NBA pipeline once implemented).
- Any NBA-specific caveats (e.g., differences in positions or stacking behavior).

### To-dos

- [ ] Restructure data and outputs into sport-aware folders (data/nfl, data/nba, outputs/nfl, outputs/nba) and update any hard-coded paths.
- [ ] Create src/shared package and move sport-agnostic simulation, correlation, IO, and utility code into it.
- [ ] Create src/nfl package with NFL-specific config and correlation pipeline wrapper that delegates to shared code.
- [ ] Create src/nba config and Sabersim parser mirroring the NFL structure but with NBA-specific assumptions.
- [ ] Implement an NBA correlation entrypoint module that uses the shared engine and NBA parser to write outputs/nba/correlations workbooks.
- [ ] Run the NBA correlation pipeline on a real Sabersim NBA CSV and verify outputs for plausibility.
- [ ] Refactor optimizer, top1% simulation, diversification, DKEntries filling, and flashback code into shared + per-sport layers for both NFL and NBA.
- [ ] Add sport-aware shell entrypoints (e.g., run_full_nfl.sh, run_full_nba.sh) and update README with NFL vs NBA usage.