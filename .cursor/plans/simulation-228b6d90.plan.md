<!-- 228b6d90-d2fd-4281-a792-7538dc177e08 121404bc-a2b5-4bcf-89a4-a44257a53730 -->
## Slimming the project to simulation-only

### Goals

- Remove all code, models, and diagnostics related to the original ML regression-based correlation pipeline.
- Keep only the essentials needed to:
- Download and normalize historical NFL data (via the existing downloader script).
- Load a Sabersim Showdown CSV and build a simulation-based correlation matrix.

### 1. Identify and remove ML-specific modules and artifacts

- **Source modules to remove or deprecate** (after verifying no remaining references):
- `src/fantasy_scoring.py` (historical DK scoring for training).
- `src/feature_engineering.py` (z-scores, mu/sigma).
- `src/build_pairwise_dataset.py` (z_i * z_j training dataset).
- `src/train_corr_model.py` (regression model training and persistence).
- Any ML-only helpers in `src/data_loading.py` that are no longer used by `download_nfl_data.py`.
- **Model artifacts**:
- Remove `models/corr_model.pkl` and the `models/` directory if nothing else uses it.
- **Diagnostics**:
- Remove or ignore historical diagnostics under `diagnostics/fantasy_scoring/`, `diagnostics/pairwise_build/`, and `diagnostics/train_*` that are only relevant to the ML pipeline.

### 2. Simplify configuration to simulation + downloader

- In `src/config.py`:
- Keep base paths (`PROJECT_ROOT`, `DATA_DIR`, `NFL_RAW_DIR`, `SABERSIM_DIR`, `OUTPUTS_DIR`, `DIAGNOSTICS_DIR`).
- Keep any constants used by `download_nfl_data.py` (e.g., `NFL_PLAYER_GAMES_PARQUET`, `NFL_GAMES_PARQUET`, and column name mappings) if the downloader still depends on them.
- Keep simulation-related settings: `SABERSIM_CSV`, `OUTPUT_CORR_EXCEL`, `DEFAULT_CORR_METHOD` (can be simplified to always "simulation" or removed in favor of a simpler flag), `SIM_N_GAMES`, `SIM_RANDOM_SEED`, `SIM_DIRICHLET_K_*`, `SIM_EPS`.
- Remove ML-only settings: `PROCESSED_PLAYER_GAMES_PARQUET`, `CORR_MODEL_PATH`, `MIN_PLAYER_GAMES`, `TRAIN_SEASONS`, `VAL_SEASONS`, `TEST_SEASONS`, and any constants only used by the training pipeline.

### 3. Streamline the main entrypoint to simulation-only

- In `src/main.py`:
- Remove imports and logic for `build_pairwise_dataset`, `fantasy_scoring`, `feature_engineering`, and `train_corr_model`.
- Drop the `--retrain` and `--corr-method` flags; keep a minimal CLI with:
- `--sabersim-csv` (override `config.SABERSIM_CSV`).
- `--output-excel` (override `config.OUTPUT_CORR_EXCEL`).
- `--n-sims` (override `config.SIM_N_GAMES`).
- The main flow becomes:

1. Ensure output/diagnostics directories exist.
2. Load Sabersim projections via `load_sabersim_projections`.
3. Call `simulate_corr_matrix_from_projections` to get the correlation matrix.
4. Write projections + correlation matrix to Excel.

- Optionally add a separate small CLI entry (or flag) if you want to run `download_nfl_data` from `main.py`; otherwise, leave it as a standalone script.

### 4. Keep and lightly isolate the downloader

- Keep `src/download_nfl_data.py` mostly as-is, but:
- Confirm which pieces of `config` and `data_loading` it uses.
- If `data_loading.py` is mostly training-oriented, either:
- Keep only the parts needed by the downloader (e.g., column normalization), or
- Inline minimal loading/normalization logic into `download_nfl_data.py` and remove `data_loading.py` entirely.
- Ensure the README documents how to run the downloader as a separate step, if you still plan to use those Parquet files for other analysis.

### 5. Update diagnostics to focus on simulation

- Keep `src/diagnostics.py` but note it is now used only by the simulator.
- Retain and/or enhance current simulation diagnostics:
- `sim_input_players`, `sim_team_totals`, `dk_sim_vs_proj`, `dk_points_first_100`.
- Optionally remove or ignore old diagnostics directories to avoid confusion, or clearly separate them by step names.

### 6. Refresh README and cleanup

- Update `README.md` to:
- Describe the project as a **simulation-based** NFL Showdown correlation generator, with an optional historical data downloader.
- Remove references to training an ML regression model, z-scores, and `corr_model.pkl`.
- Show minimal usage examples:
- How to run `python -m src.download_nfl_data` (if still relevant).
- How to run `python -m src.main` with `--sabersim-csv`, `--n-sims`, and `--output-excel`.
- Optionally clean up `requirements.txt` to drop ML-only dependencies (e.g., `scikit-learn`, `joblib`) if nothing else uses them.

This plan will leave you with a lean codebase centered on your simulation engine and keep the historical downloader as a utility, while removing unused ML training code and artifacts.

### To-dos

- [ ] Add simulation-related configuration parameters and optional correlation method flag in src/config.py (and wire a matching CLI flag in main.py).
- [ ] Implement a helper to map Sabersim projection columns (yards, TDs, receptions) into canonical stat fields and compute team-level projected totals per stat and per team.
- [ ] Implement simulation engine in a new module (e.g., src/simulation_corr.py) that simulates team-level stat pools and allocates them to players with Dirichlet/multinomial sampling, enforcing basic passing and receiving consistency constraints.
- [ ] Integrate fantasy scoring into the simulator, run many simulated games, and compute the empirical player-by-player correlation matrix of DK points.
- [ ] Wire the simulator into main.py so the Showdown correlation matrix is built via simulation by default, with an optional ML fallback path for comparison.
- [ ] Add diagnostics to validate that simulated DK means are close to projections and that key player pairs (e.g., QB–WR, QB–RB) have sensible correlation signs and magnitudes.
- [ ] Update README.md to describe the simulation-based correlation method, configuration, and how to run it.