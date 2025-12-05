<!-- 0f1d0d6f-c004-4184-bca2-9461a3c33a39 410f6f4a-e39f-470a-9df0-95b3db8b1777 -->
# NBA pipeline extension and two-sport refactor

### Goal

Bring the NBA side up to parity with NFL for the full DraftKings Showdown pipeline (correlations → optimizer → top 1% → diversification → DKEntries → flashback), while cleaning up the code layout into **shared**, **NFL**, and **NBA** layers and making the CLIs sport-aware.

---

## Step 1: Confirm and tidy the directory structure

- **Sport-inside-domain layout**
- Ensure the project consistently uses:
- `data/nfl/...`, `data/nba/...` (with `sabersim/`, `contests/`, `dkentries/`, `payouts/`, etc.).
- `outputs/nfl/...`, `outputs/nba/...` (with `correlations/`, `lineups/`, `top1pct/`, `flashback/`, `dkentries/`).
- **Configs go through `shared/config_base`**
- Verify `src/config.py` (NFL) and `src/nba/config.py` (NBA):
- Import from `src/shared/config_base.py` for `PROJECT_ROOT`, `DATA_DIR`, `OUTPUTS_DIR`, etc.
- Point `DATA_DIR` to `data/nfl` vs `data/nba`, `OUTPUTS_DIR` to `outputs/nfl` vs `outputs/nba`.
- **Eliminate hardcoded root paths**
- Grep for bare `outputs/` and `data/` in the codebase.
- Replace remaining references with `config.DATA_DIR`, `config.OUTPUTS_DIR` (NFL) or `nba.config.DATA_DIR`, `nba.config.OUTPUTS_DIR` (NBA), except where genuinely top-level (e.g. README examples).

---

## Step 2: Extract a shared optimizer core

- **Create `src/shared/optimizer_core.py`**
- Move from `src/lineup_optimizer.py` into this shared module:
- Data structures and types:
- `Player`, `PlayerPool`, `Lineup`.
- `SLOTS`, `CPT_SLOT`, `FLEX_SLOTS`, `VarKey`, `VarDict`.
- Core MILP logic:
- Model creation and constraints: `build_showdown_model`, `add_single_cpt_constraint`,
`add_flex_count_constraint`, `add_unique_player_constraint`, `add_salary_cap_constraint`,
`add_eligibility_constraints`, `add_min_one_per_team_constraint`, `add_base_constraints`.
- Objective: `set_mean_projection_objective`.
- No-duplicate and projection-cap features: `_build_showdown_model_with_constraints`,
`_add_projection_cap_constraint`.
- Solution extraction and public API: `_extract_lineup_from_solution`, `solve_single_lineup`,
`optimize_showdown_lineups`.
- Generic constraint helpers such as `min_players_from_team`.
- **Update `src/showdown_constraints.py` to use the shared core**
- Change imports to pull `Player`, `PlayerPool`, `CPT_SLOT`, `SLOTS`, `ConstraintBuilder`, and
`min_players_from_team` from `shared.optimizer_core` instead of `lineup_optimizer`.
- Leave genuinely NFL-specific constraints (QB + pass-catchers, RB CPT vs opposing DST, team stacks)
in this NFL-only configuration module.
- **Slim `src/lineup_optimizer.py` to an NFL adaptor**
- Re-export shared core symbols:
- `Player`, `PlayerPool`, `Lineup`, `ConstraintBuilder`, `optimize_showdown_lineups` imported from
`shared.optimizer_core`.
- Keep only NFL-specific CSV loading logic:
- `_load_raw_sabersim_csv` and `load_players_from_sabersim` that parse NFL Sabersim CSVs and
construct a `PlayerPool`.

---

## Step 3: Create explicit NFL and NBA optimizer entrypoints

- **Refine NFL optimizer CLI** (`src/showdown_optimizer_main.py` or `src/nfl/showdown_optimizer_main.py`)
- Treat this as the NFL wrapper:
- Import optimizer primitives (`PlayerPool`, `Lineup`, `optimize_showdown_lineups`) from
`shared.optimizer_core`.
- Import NFL constraints from `showdown_constraints`.
- Use `lineup_optimizer.load_players_from_sabersim` to build the NFL `PlayerPool`.
- Ensure it writes lineups to `config.OUTPUTS_DIR / "lineups"` (i.e. `outputs/nfl/lineups`).
- **Add NBA optimizer CLI** (`src/nba/showdown_optimizer_main.py`)
- Implement a new entrypoint that:
- Mirrors NFL CLI flags where possible:
- `--sabersim-glob` (default: `nba.config.SABERSIM_CSV`).
- `--num-lineups`, `--salary-cap`, `--chunk-size`.
- Optionally `--stack-mode`, `--stack-weights` if you want NBA multi-stack support.
- Defines `load_players_from_sabersim_nba(path)` that:
- Calls `nba.sabersim_parser.load_sabersim_projections(path)` to get FLEX-only rows.
- Builds `shared.optimizer_core.Player` objects from Sabersim columns (`Name`, `Team`, `Pos`,
`Salary`, `My Proj`) and returns a `PlayerPool`.
- Calls `optimize_showdown_lineups` with the NBA `PlayerPool` and any NBA-specific
`ConstraintBuilder`s (can be empty initially).
- Writes NBA lineup workbooks to `nba.config.OUTPUTS_DIR / "lineups" / lineups_{timestamp}.xlsx`.

---

## Step 4: Make top 1% and diversification sport-aware

- **Extract shared top 1% core** (`src/shared/top1pct_core.py`)
- Move sport-agnostic logic from `src/top1pct_finish_rate.py` into this module:
- File helpers: `_resolve_latest_excel(directory, explicit)`.
- Workbook loaders: `_load_lineups_workbook`, `_load_corr_workbook`.
- Universe/correlation builders: `_build_player_universe`, `_build_full_correlation`.
- Simulation pieces: `_simulate_player_scores`, `_compute_field_thresholds`,
`_build_lineup_weights`.
- Public function:
- `run_top1pct(field_size, outputs_dir, lineups_excel=None, corr_excel=None, num_sims=..., random_seed=..., field_var_shrink=..., field_z=..., flex_var_factor=...)`, which:
  - Uses `outputs_dir / "lineups"` and `outputs_dir / "correlations"`.
  - Writes top-1% workbooks to `outputs_dir / "top1pct"`.
- **NFL wrapper** (`src/top1pct_finish_rate.py`)
- Keep the current CLI interface and user docs.
- Replace internal implementation with a thin wrapper that calls
`shared.top1pct_core.run_top1pct(..., outputs_dir=config.OUTPUTS_DIR, ...)`.
- **NBA wrapper** (`src/nba/top1pct_finish_rate_nba.py`)
- Add a new module that:
- Exposes the same CLI flags as the NFL version.
- Calls `run_top1pct(..., outputs_dir=nba.config.OUTPUTS_DIR, ...)`.

- **Extract shared diversification core** (`src/shared/diversify_core.py`)
- Move from `src/diversify_lineups.py` into this module:
- `_load_top1pct_lineups`, `_build_player_set`, `_greedy_diversified_selection`,
`_compute_exposure`.
- Public function `run_diversify(num_lineups, outputs_dir, min_top1_pct=..., max_overlap=..., top1pct_excel=None)` that:
- Reads from `outputs_dir / "top1pct"`.
- Writes `top1pct_diversified_{num_lineups}.xlsx` into the same directory.
- **NFL & NBA diversification wrappers**
- Update `src/diversify_lineups.py` to call
`shared.diversify_core.run_diversify(..., outputs_dir=config.OUTPUTS_DIR, ...)`.
- Add `src/nba/diversify_lineups_nba.py` that calls
`run_diversify(..., outputs_dir=nba.config.OUTPUTS_DIR, ...)`.

> Keep the Excel workbook schemas (sheet names and columns) identical between NFL and NBA so
> downstream tools (e.g. DKEntries filler) work uniformly.

---

## Step 5: Generalize DKEntries filling and flashback analysis

### DKEntries

- **Shared DKEntries core** (`src/shared/dkentries_core.py`)
- Extract from `src/dkentries_utils.py` and `src/fill_dkentries.py`:
- `resolve_latest_dkentries_csv(data_dir, explicit=None)`, which looks under
`data_dir / "dkentries"`.
- `count_real_entries(path)`.
- DK player dictionary helpers: `_find_mapping_columns(df)` and `_build_name_role_to_id_map(df)`.
- Lineup slot helpers: `_get_lineup_slot_columns(df)`.
- Core logic: `_assign_lineups_fee_aware(...)` and `_apply_assignments_to_dkentries(...)`.
- **NFL wrappers**
- `src/dkentries_utils.py`:
- Wrap `shared.dkentries_core.resolve_latest_dkentries_csv(config.DATA_DIR, explicit)`.
- `src/fill_dkentries.py`:
- Use shared helpers but default to NFL locations:
- Templates: `config.DATA_DIR / "dkentries"`.
- Outputs: `config.OUTPUTS_DIR / "dkentries"`.
- **NBA DKEntries wrapper** (`src/nba/fill_dkentries_nba.py`)
- Add a new module that:
- Mirrors the NFL CLI and behavior.
- Uses `nba.config.DATA_DIR / "dkentries"` for templates and
`nba.config.OUTPUTS_DIR / "dkentries"` for filled outputs.
- Uses the shared DKEntries core for all parsing and assignment logic.

### Flashback

- **Shared flashback core** (`src/shared/flashback_core.py`)
- Extract from `src/flashback_sim.py` the sport-agnostic pieces:
- Contest parsing: `ParsedLineup` dataclass, `_parse_lineup_string`, `_load_contest_lineups`.
- Player universe and weights: `_build_player_universe_and_weights`,
`_build_player_universe_from_sabersim_and_lineups`, `_build_full_correlation`, `_make_psd`,
`_simulate_player_scores`.
- Outcome modeling: `_compute_lineup_finish_rates`, `_compute_sim_roi`.
- Summaries: entrant and player summary builders.
- Orchestration function `run_flashback(contest_csv, sabersim_csv, num_sims, random_seed, payouts_csv, config_module, dk_scoring_fn)` that:
- Uses `config_module.DATA_DIR` / `config_module.OUTPUTS_DIR`.
- Uses `dk_scoring_fn` to translate means/variances into DK scoring.
- **NFL wrapper** (`src/flashback_sim.py`)
- Simplify to a thin CLI that:
- Parses NFL-specific flags.
- Calls `shared.flashback_core.run_flashback(..., config_module=src.config, dk_scoring_fn=nfl_offense_scoring)`.
- **NBA flashback wrapper** (`src/nba/flashback_sim_nba.py`)
- Add a new module that:
- Parses analogous CLI flags for NBA contests.
- Uses `nba.config` for paths (`data/nba/contests`, `data/nba/sabersim`, `outputs/nba/flashback`).
- Uses NBA DK scoring from `nba.stat_sim.compute_dk_points_nba` (or a compatible scoring wrapper).

---

## Step 6: CLI wrappers and README updates

### Shell entrypoints

- **NFL `run_full.sh`**
- Keep as the NFL end-to-end pipeline script, but verify it:
- Calls `python -m src.main` (NFL correlations, writing to `outputs/nfl/correlations`).
- Calls `python -m src.showdown_optimizer_main` (NFL optimizer → `outputs/nfl/lineups`).
- Calls `python -m src.top1pct_finish_rate` (NFL top 1% → `outputs/nfl/top1pct`).
- Calls `python -m src.diversify_lineups` (NFL diversification → `outputs/nfl/top1pct`).
- Calls `python -m src.fill_dkentries` (NFL DKEntries fill → `outputs/nfl/dkentries`).
- **Add `run_full_nba.sh`**
- Create a parallel script that mirrors the NFL steps but uses NBA modules:
- NBA correlation: `python -m src.nba.main` writing to `outputs/nba/correlations/...`.
- NBA optimizer: `python -m src.nba.showdown_optimizer_main` writing to `outputs/nba/lineups/...`.
- NBA top 1%: `python -m src.nba.top1pct_finish_rate_nba` writing to `outputs/nba/top1pct/...`.
- NBA diversification: `python -m src.nba.diversify_lineups_nba` writing `top1pct_diversified_*.xlsx` under `outputs/nba/top1pct/...`.
- NBA DKEntries fill: `python -m src.nba.fill_dkentries_nba` writing under `outputs/nba/dkentries/...`.

### README updates

- **NBA Showdown section**
- In `README.md`, add a new section paralleling the NFL documentation:
- **Data layout**: where to place NBA Sabersim CSVs and contest/DKEntries files (`data/nba/sabersim/`, `data/nba/contests/`, `data/nba/dkentries/`, `data/nba/payouts/`).
- **NBA correlation**: usage example for `python -m src.nba.main --sabersim-csv ... --n-sims ...`.
- **Full NBA pipeline**: usage example for `./run_full_nba.sh SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] ...`.
- **Module layout overview**
- Briefly describe:
- `src/shared/...` – shared core (config base, optimizer, top1%, DKEntries, flashback, etc.).
- NFL modules under `src/` – NFL config, optimizer CLI, flashback CLI, etc.
- NBA modules under `src/nba/` – NBA config, correlation/optimizer/top1%/diversify/DKEntries/flashback CLIs.

---

## Implementation todos

- **dirs-and-config**: Verify/clean up `data/{sport}` and `outputs/{sport}` usage across configs and code to ensure everything flows through `shared/config_base` and sport-specific config modules.
- **optimizer-core**: Extract MILP core from `src/lineup_optimizer.py` / `src/showdown_constraints.py` into `src/shared/optimizer_core.py` and update NFL optimizer to use it.
- **nba-optimizer-cli**: Implement `src/nba/showdown_optimizer_main.py` that uses the shared optimizer core and NBA Sabersim loader, and writes lineups to `outputs/nba/lineups`.
- **shared-top1pct-diversify**: Move generic top1%/diversify logic into `src/shared/` and add sport-aware wrappers or a `--sport` flag so NBA can use its own correlation/lineups directories.
- **shared-dkentries-flashback**: Extract shared DKEntries + flashback pieces into `src/shared/` and add NBA-aware entrypoints and paths.
- **cli-and-readme**: Add `run_full_nba.sh` (or a unified sport-aware full-run script) and update `README.md` with clear NFL vs NBA usage and directory structure.

### To-dos

- [x] Restructure data and outputs into sport-aware folders (data/nfl, data/nba, outputs/nfl, outputs/nba) and update any hard-coded paths.
- [x] Create src/shared package and move sport-agnostic simulation, correlation, IO, and utility code into it.
- [x] Create src/nfl package with NFL-specific config and correlation pipeline wrapper that delegates to shared code.
- [x] Create src/nba config and Sabersim parser mirroring the NFL structure but with NBA-specific assumptions.
- [x] Implement an NBA correlation entrypoint module that uses the shared engine and NBA parser to write outputs/nba/correlations workbooks.
- [x] Run the NBA correlation pipeline on a real Sabersim NBA CSV and verify outputs for plausibility.
- [x] Refactor optimizer, top1% simulation, diversification, DKEntries filling, and flashback code into shared + per-sport layers for both NFL and NBA.
- [x] Add sport-aware shell entrypoints (e.g., run_full_nfl.sh, run_full_nba.sh) and update README with NFL vs NBA usage.
- [ ] Review NFL correlation implementation and catalog which NBA Sabersim columns to use for stat simulation.
- [ ] Define the NBA stat schema (PTS, RB, AST, STL, BLK, TOV, FG3M, etc.) and a DK scoring function for Showdown.
- [ ] Implement nba/stat_sim.py to sample per-player stat lines from Sabersim projections and convert them to DK points.
- [ ] Implement an NBA-specific correlation builder that computes corr(DK_i, DK_j) from simulated DK-point matrices and wire it into src/nba/simulation_corr.py.
- [ ] Run the updated NBA pipeline on the OKC–GSW slate, inspect projections vs simulated means and spot-check correlations for sanity.
- [ ] Optionally refine the NBA stat model (bonuses, team coupling) and update README with a short description of the NBA correlation process.
- [ ] Restructure data and outputs into sport-aware folders (data/nfl, data/nba, outputs/nfl, outputs/nba) and update any hard-coded paths.
- [ ] Create src/shared package and move sport-agnostic simulation, correlation, IO, and utility code into it.
- [ ] Create src/nfl package with NFL-specific config and correlation pipeline wrapper that delegates to shared code.
- [ ] Create src/nba config and Sabersim parser mirroring the NFL structure but with NBA-specific assumptions.
- [ ] Implement an NBA correlation entrypoint module that uses the shared engine and NBA parser to write outputs/nba/correlations workbooks.
- [ ] Run the NBA correlation pipeline on a real Sabersim NBA CSV and verify outputs for plausibility.
- [ ] Refactor optimizer, top1% simulation, diversification, DKEntries filling, and flashback code into shared + per-sport layers for both NFL and NBA.
- [ ] Add sport-aware shell entrypoints (e.g., run_full_nfl.sh, run_full_nba.sh) and update README with NFL vs NBA usage.
- [x] Compare current simulated DK means/stds/quantiles to Sabersim dk_points and dk_std to understand variance gaps.
- [x] Specify game- and team-level multiplier distributions (e.g., lognormal/normal on log scale) and how they interact with existing Poisson stat draws.
- [x] Extend the NBA stat simulator to apply game/team multipliers to per-player stat draws while preserving long-run means.
- [x] Tune coupling variances and overdispersion so average simulated dk_std and tails roughly match Sabersim’s dk_std and quantiles.
- [x] Inspect refined NBA correlation matrices (teammate vs cross-team pairs) and compare to the previous independent-player model for sanity.
- [x] Add NBA coupling hyperparameters to config and update README to describe the new NBA correlation behavior and tuning knobs.
- [ ] Restructure data and outputs into sport-aware folders (data/nfl, data/nba, outputs/nfl, outputs/nba) and update any hard-coded paths.
- [ ] Create src/shared package and move sport-agnostic simulation, correlation, IO, and utility code into it.
- [ ] Create src/nfl package with NFL-specific config and correlation pipeline wrapper that delegates to shared code.
- [ ] Create src/nba config and Sabersim parser mirroring the NFL structure but with NBA-specific assumptions.
- [ ] Implement an NBA correlation entrypoint module that uses the shared engine and NBA parser to write outputs/nba/correlations workbooks.
- [ ] Run the NBA correlation pipeline on a real Sabersim NBA CSV and verify outputs for plausibility.
- [ ] Refactor optimizer, top1% simulation, diversification, DKEntries filling, and flashback code into shared + per-sport layers for both NFL and NBA.
- [ ] Add sport-aware shell entrypoints (e.g., run_full_nfl.sh, run_full_nba.sh) and update README with NFL vs NBA usage.
- [ ] Verify and clean up data/{sport} and outputs/{sport} usage across configs and code so all paths go through shared/config_base and the sport-specific config modules.
- [ ] Extract sport-agnostic MILP optimizer core into src/shared/optimizer_core.py and update the NFL optimizer to use it.
- [ ] Add an NBA optimizer CLI (e.g., src/nba/showdown_optimizer_main.py) that uses the shared core and NBA Sabersim loader and writes to outputs/nba/lineups.
- [ ] Refactor top1pct_finish_rate and diversify_lineups into shared math plus sport-aware wrappers or a --sport flag so NBA can run its own top1% and diversification pipeline.
- [ ] Generalize DKEntries filler and flashback_sim into shared components with NFL and NBA entrypoints and sport-specific paths/scoring where needed.
- [ ] Add a run_full_nba.sh (or unified sport-aware script) and update README.md with NFL vs NBA data layout and usage for the full pipelines.
- [ ] Restructure data and outputs into sport-aware folders (data/nfl, data/nba, outputs/nfl, outputs/nba) and update any hard-coded paths.
- [ ] Create src/shared package and move sport-agnostic simulation, correlation, IO, and utility code into it.
- [ ] Create src/nfl package with NFL-specific config and correlation pipeline wrapper that delegates to shared code.
- [ ] Create src/nba config and Sabersim parser mirroring the NFL structure but with NBA-specific assumptions.
- [ ] Implement an NBA correlation entrypoint module that uses the shared engine and NBA parser to write outputs/nba/correlations workbooks.
- [ ] Run the NBA correlation pipeline on a real Sabersim NBA CSV and verify outputs for plausibility.
- [ ] Refactor optimizer, top1% simulation, diversification, DKEntries filling, and flashback code into shared + per-sport layers for both NFL and NBA.
- [ ] Add sport-aware shell entrypoints (e.g., run_full_nfl.sh, run_full_nba.sh) and update README with NFL vs NBA usage.
- [ ] Verify and clean up data/{sport} and outputs/{sport} usage across configs and code so all paths go through shared/config_base and the sport-specific config modules.
- [ ] Extract sport-agnostic MILP optimizer core into src/shared/optimizer_core.py and update the NFL optimizer to use it.
- [ ] Add an NBA optimizer CLI (e.g., src/nba/showdown_optimizer_main.py) that uses the shared core and NBA Sabersim loader and writes to outputs/nba/lineups.
- [ ] Refactor top1pct_finish_rate and diversify_lineups into shared math plus sport-aware wrappers or a --sport flag so NBA can run its own top1% and diversification pipeline.
- [ ] Generalize DKEntries filler and flashback_sim into shared components with NFL and NBA entrypoints and sport-specific paths/scoring where needed.
- [ ] Add a run_full_nba.sh (or unified sport-aware script) and update README.md with NFL vs NBA data layout and usage for the full pipelines.
- [ ] Restructure data and outputs into sport-aware folders (data/nfl, data/nba, outputs/nfl, outputs/nba) and update any hard-coded paths.
- [ ] Create src/shared package and move sport-agnostic simulation, correlation, IO, and utility code into it.
- [ ] Create src/nfl package with NFL-specific config and correlation pipeline wrapper that delegates to shared code.
- [ ] Create src/nba config and Sabersim parser mirroring the NFL structure but with NBA-specific assumptions.
- [ ] Implement an NBA correlation entrypoint module that uses the shared engine and NBA parser to write outputs/nba/correlations workbooks.
- [ ] Run the NBA correlation pipeline on a real Sabersim NBA CSV and verify outputs for plausibility.
- [ ] Refactor optimizer, top1% simulation, diversification, DKEntries filling, and flashback code into shared + per-sport layers for both NFL and NBA.
- [ ] Add sport-aware shell entrypoints (e.g., run_full_nfl.sh, run_full_nba.sh) and update README with NFL vs NBA usage.
- [ ] Verify and clean up data/{sport} and outputs/{sport} usage across configs and code so all paths go through shared/config_base and the sport-specific config modules.
- [ ] Extract sport-agnostic MILP optimizer core into src/shared/optimizer_core.py and update the NFL optimizer to use it.
- [ ] Add an NBA optimizer CLI (e.g., src/nba/showdown_optimizer_main.py) that uses the shared core and NBA Sabersim loader and writes to outputs/nba/lineups.
- [ ] Refactor top1pct_finish_rate and diversify_lineups into shared math plus sport-aware wrappers or a --sport flag so NBA can run its own top1% and diversification pipeline.
- [ ] Generalize DKEntries filler and flashback_sim into shared components with NFL and NBA entrypoints and sport-specific paths/scoring where needed.
- [ ] Add a run_full_nba.sh (or unified sport-aware script) and update README.md with NFL vs NBA data layout and usage for the full pipelines.
- [ ] Restructure data and outputs into sport-aware folders (data/nfl, data/nba, outputs/nfl, outputs/nba) and update any hard-coded paths.
- [ ] Create src/shared package and move sport-agnostic simulation, correlation, IO, and utility code into it.
- [ ] Create src/nfl package with NFL-specific config and correlation pipeline wrapper that delegates to shared code.
- [ ] Create src/nba config and Sabersim parser mirroring the NFL structure but with NBA-specific assumptions.
- [ ] Implement an NBA correlation entrypoint module that uses the shared engine and NBA parser to write outputs/nba/correlations workbooks.
- [ ] Run the NBA correlation pipeline on a real Sabersim NBA CSV and verify outputs for plausibility.
- [ ] Refactor optimizer, top1% simulation, diversification, DKEntries filling, and flashback code into shared + per-sport layers for both NFL and NBA.
- [ ] Add sport-aware shell entrypoints (e.g., run_full_nfl.sh, run_full_nba.sh) and update README with NFL vs NBA usage.
- [ ] Compare current simulated DK means/stds/quantiles to Sabersim dk_points and dk_std to understand variance gaps.
- [ ] Specify game- and team-level multiplier distributions (e.g., lognormal/normal on log scale) and how they interact with existing Poisson stat draws.
- [ ] Extend the NBA stat simulator to apply game/team multipliers to per-player stat draws while preserving long-run means.
- [ ] Tune coupling variances and overdispersion so average simulated dk_std and tails roughly match Sabersim’s dk_std and quantiles.
- [ ] Inspect refined NBA correlation matrices (teammate vs cross-team pairs) and compare to the previous independent-player model for sanity.
- [ ] Add NBA coupling hyperparameters to config and update README to describe the new NBA correlation behavior and tuning knobs.