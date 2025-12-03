## Showdown Optimizer – Project Layout

This repository is organized around a small number of Python packages under `src/`
to clearly separate shared logic from sport-specific code:

- `src/shared/` – **Sport-agnostic cores and utilities**
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

- `src/nfl/` – **NFL-specific configuration and CLIs**
  - `config.py`: NFL paths (e.g. `data/nfl`, `outputs/nfl`), column names, and
    simulation hyperparameters.
  - `data_loading.py`, `diagnostics.py`: nflverse-style Parquet loaders and
    diagnostics that use the NFL config.
  - `build_corr_matrix_from_projections.py`, `simulation_corr.py`: NFL Sabersim
    loader and Monte Carlo correlation engine.
  - CLIs (used via `python -m src.nfl.<module>`):
    - `main`: build correlations from Sabersim (`src.nfl.main`).
    - `showdown_optimizer_main`: NFL Showdown optimizer.
    - `top1pct_finish_rate`: top 1% finish-rate estimation for NFL.
    - `diversify_lineups`: select diversified NFL lineups from top1% output.
    - `flashback_sim`: flashback contest simulation for completed NFL slates.
    - `fill_dkentries`: fill DKEntries templates with diversified NFL lineups.
    - `download_nfl_data`: download and normalize historical NFL data.

- `src/nba/` – **NBA-specific configuration and CLIs**
  - `config.py`: NBA paths (e.g. `data/nba`, `outputs/nba`) and any NBA-specific
    knobs.
  - `sabersim_parser.py`, `simulation_corr.py`, `stat_sim.py`: NBA Sabersim
    parsing and correlation/stat simulation.
  - CLIs (used via `python -m src.nba.<module>`):
    - `main`: NBA driver script (where applicable).
    - `showdown_optimizer_main`: NBA Showdown optimizer.
    - `top1pct_finish_rate_nba`: NBA top 1% finish-rate estimation.
    - `diversify_lineups_nba`: select diversified NBA lineups from top1% output.
    - `flashback_sim_nba`: flashback contest simulation for NBA.
    - `fill_dkentries_nba`: fill DKEntries templates with diversified NBA lineups.

In day-to-day use, prefer the sport-specific entrypoints under `src/nfl` and
`src/nba` (e.g. `python -m src.nfl.showdown_optimizer_main ...`) while treating
`src/shared` as an implementation detail that you rarely need to call directly.


