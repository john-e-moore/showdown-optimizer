<!-- f8f3fa04-3a10-4434-910d-a1156427f453 24f2a417-1b54-4524-b18c-e536eef2d256 -->
# NFL Showdown Lineup Optimizer Implementation Plan

### High-level goals

- **Add a MILP-based DraftKings Showdown (Captain Mode) lineup optimizer** that reads Sabersim projections from `data/sabersim/NFL_*.csv`, builds valid CPT + FLEX lineups, supports constraint hooks, and returns top N lineups by mean projection.
- **Integrate cleanly with the existing `src` package**, reusing Sabersim loading conventions and config paths, and expose a **CLI entry point** (e.g. `python -m src.showdown_optimizer_main`).

### Files to add or modify

- **New**: [`src/lineup_optimizer.py`](src/lineup_optimizer.py) – core domain models (`Player`, `PlayerPool`, `Lineup`) and MILP optimizer functions.
- **New**: [`src/showdown_optimizer_main.py`](src/showdown_optimizer_main.py) – CLI entry to run the optimizer from the command line.
- **Modify**: [`requirements.txt`](requirements.txt) – add `pulp` as the MILP solver dependency.
- **Optional/Light modify**: [`src/config.py`](src/config.py) – add small constants for default lineup/solver settings and possibly a helper for default Sabersim glob pattern.
- **Optional/Light modify**: [`README.md`](README.md) – brief section documenting how to run the optimizer.

### Step 1: Define core data models and Sabersim loading

- **Implement `Player`, `PlayerPool`, and `Lineup`** in `src/lineup_optimizer.py`:
  - **`Player`** dataclass: `player_id: str`, `name: str`, `team: str`, `position: str`, `dk_salary: int`, `dk_proj: float`, `dk_std: float | None`, `is_cpt_eligible: bool`, `is_flex_eligible: bool`.
  - **`PlayerPool`**: wraps `List[Player] `and provides helpers `by_team(team)`, `by_position(position)`, `get(player_id)`.
  - **`Lineup`**: fields `cpt: Player`, `flex: List[Player] `with methods `salary()`, `projection()`, `as_tuple_ids()` implementing CPT 1.5x rules from the prompt.
- **Add a loader to build `PlayerPool` from a Sabersim CSV**:
  - Either reuse or parallel the logic in [`src/build_corr_matrix_from_projections.py`](src/build_corr_matrix_from_projections.py) but adapted to the columns described in `prompts/01_lineup_optimizer.md`.
  - Implement a function like `load_players_from_sabersim(path: str) -> PlayerPool` that:
    - Reads the CSV with `pandas.read_csv`.
    - Maps the actual Sabersim columns (e.g. `Name`, `Team`, `Pos`, `Salary`, `My Proj`) into the canonical optimizer fields.
    - Handles optional columns `dk_std`, `is_cpt_eligible`, `is_flex_eligible` (defaulting as described in the prompt).
  - Keep this loader in `lineup_optimizer.py` (or a small helper in the same file) to avoid over-fragmenting modules.

### Step 2: Build the single-lineup MILP model with PuLP

- **Introduce an internal model builder** in `lineup_optimizer.py`, e.g.:
  - `def build_showdown_model(player_pool: PlayerPool, salary_cap: int) -> tuple[pulp.LpProblem, dict[tuple[str, str], pulp.LpVariable]]:`
- **Decision variables**:
  - For each `player` and slot `s` in `{"CPT", "F1", "F2", "F3", "F4", "F5"}`, create binary variable `x[(player.player_id, s)]`.
- **Base constraints (implemented as separate helper functions for clarity)**:
  - `add_single_cpt_constraint(prob, x, player_pool)` enforcing `sum_i x[i, CPT] == 1`.
  - `add_flex_count_constraint(prob, x, player_pool)` enforcing `sum_i,sum_FLEX x[i, Fj] == 5`.
  - `add_unique_player_constraint(prob, x, player_pool)` enforcing each player appears in at most one slot.
  - `add_salary_cap_constraint(prob, x, player_pool, salary_cap)` implementing CPT 1.5x salary and FLEX normal salary with `<= salary_cap`.
  - `add_eligibility_constraints(prob, x, player_pool)` zeroing out CPT/FLEX slots for ineligible players.
- **Objective**:
  - Implement `set_mean_projection_objective(prob, x, player_pool)` using 1.5x CPT projections and 1.0x FLEX projections as specified.
- Keep all these helpers **pure and deterministic**, returning the `prob` and `x` mapping and not doing any file IO or CLI work.

### Step 3: Custom DFS constraint hooks

- **Define a simple constraint-builder protocol** in `lineup_optimizer.py`:
  - Type alias: `ConstraintBuilder = Callable[[pulp.LpProblem, dict[tuple[str, str], pulp.LpVariable], PlayerPool], None]`.
  - The optimizer will accept a list of such callables and invoke them after base constraints are added.
- **Implement a small library of reusable constraints as examples**, matching the prompt:
  - `min_players_from_team(team: str, k: int) -> ConstraintBuilder` – builds a closure that enforces `team_count_T >= k` (or similarly `<=` if a max helper is desired).
  - `mutually_exclusive_groups(group1_pred, group2_pred) -> ConstraintBuilder` where predicates select players based on team/position; enforce `g1 + g2 <= 1`.
  - `if_qb_cpt_then_no_dst(team_qb: str, team_dst: str) -> ConstraintBuilder` implementing `qb_cpt + dst_any <= 1`.
- These helpers will live alongside the main optimizer so users (or future code) can compose constraint lists easily.

### Step 4: Solve for a single optimal lineup and extract solution

- **Implement a helper to solve once and extract a `Lineup`**:
  - `def solve_single_lineup(player_pool: PlayerPool, salary_cap: int, constraint_builders: list[ConstraintBuilder] | None = None) -> Lineup | None`.
  - Inside: call `build_showdown_model`, add base constraints and then call each builder in `constraint_builders or []`, set objective, run `prob.solve(pulp.PULP_CBC_CMD(msg=False))`.
  - Check solver status (e.g., `pulp.LpStatus[prob.status] == "Optimal"`); return `None` or raise a clear error if infeasible.
  - Extract all `x[(player_id, slot)] `with value `>= 0.5` into one CPT and five FLEX players; guard against any shape inconsistencies with assertions.

### Step 5: Generate top N lineups with no-duplicate constraints

- **Implement the iterative top-N routine described in the prompt**:
  - Public API:
    ```python
    def optimize_showdown_lineups(
        projections_path_pattern: str,
        num_lineups: int,
        salary_cap: int = 50000,
        constraint_builders: list[ConstraintBuilder] | None = None,
    ) -> list[Lineup]:
        ...
    ```

  - Steps:
    - Resolve `projections_path_pattern` to actual CSV path(s) (using `glob.glob` or `pathlib.Path.glob`); enforce exactly one file for now and raise an error if 0 or >1.
    - Load a `PlayerPool` from that CSV using the loader from Step 1.
    - Build an initial MILP model once (`build_showdown_model`), apply base constraints, and optionally custom constraints.
    - Loop `for k in range(num_lineups)`: solve, check optimal status, extract lineup `L` and store; then **add a no-duplicate constraint** based on `L`:
      - Compute `used_player_ids = L.as_tuple_ids()`.
      - Add `sum_{pid in used_player_ids, s in all_slots} x[pid, s] <= 5` to ensure at least one player changes in the next solution.
- Consider a simple optional `max_solve_time` parameter (later) by passing `timeLimit` into `PULP_CBC_CMD`, but keep the initial implementation straightforward.

### Step 6: CLI entry point for the optimizer

- **Create `src/showdown_optimizer_main.py`** with a `main()` function and `if __name__ == "__main__": main()` block:
  - Use `argparse` similar to [`src/main.py`](src/main.py) to define flags such as:
    - `--sabersim-glob` (default something like `"data/sabersim/NFL_*.csv"`).
    - `--num-lineups` (default e.g. 20).
    - `--salary-cap` (default 50000).
    - Later optionally: flags to toggle example constraints (e.g., min players per team) or point to a Python config file.
  - Inside `main()`, call `optimize_showdown_lineups(...)` and print lineups in a simple tabular or CSV-style format to stdout, including:
    - CPT / FLEX labels, player name, team, position, salary, projection.
    - Total salary and projection per lineup.
  - Ensure this module imports from `.lineup_optimizer` using relative imports to stay within the package.

### Step 7: Configuration and dependency wiring

- **Update `requirements.txt`** to include `pulp` with a pinned, compatible version.
- **Optionally extend `src/config.py`**:
  - Add small constants such as `DEFAULT_SALARY_CAP = 50000` and maybe `DEFAULT_SABERSIM_GLOB = str(SABERSIM_DIR / "NFL_*.csv")`.
  - These can be used by `showdown_optimizer_main.py` as defaults, keeping hard-coded paths centralized.
- **Add a brief README section** summarizing:
  - What the optimizer does and that it currently maximizes **mean projection** under CPT/FLEX rules.
  - How to run it, e.g.:
    ```bash
    python -m src.showdown_optimizer_main \
      --sabersim-glob "data/sabersim/NFL_2025-11-24-815pm_DK_SHOWDOWN_CAR-@-SF.csv" \
      --num-lineups 50
    ```


### Step 8: Validation and future integration

- **Smoke tests / validation**:
  - Run the optimizer on the existing Sabersim CSV used by the correlation pipeline and verify:
    - All lineups have exactly 6 distinct players, 1 CPT and 5 FLEX.
    - Salary never exceeds the cap.
    - No-duplicate constraint actually changes at least one player across lineups.
  - Optionally log or print basic stats (e.g., top lineup projection, number of unique players used).
- **Future hook for simulation-based front end**:
  - Keep the optimizer API (`optimize_showdown_lineups`) and `Lineup` structure generic so that later it can be driven by simulated distributions (e.g., plugging in mean/variance or correlation-aware constraints) without API changes.

### To-dos

- [ ] Define `Player`, `PlayerPool`, and `Lineup` classes in `src/lineup_optimizer.py` and implement a Sabersim-to-PlayerPool CSV loader.
- [ ] Implement PuLP-based single-lineup MILP model in `src/lineup_optimizer.py` with decision variables, base constraints, and objective.
- [ ] Add reusable constraint builder helpers and integrate them into the optimization flow.
- [ ] Implement iterative top-N lineup generation with no-duplicate constraints and a public `optimize_showdown_lineups` API.
- [ ] Create `src/showdown_optimizer_main.py` CLI to run the optimizer and print/export lineups.
- [ ] Update `requirements.txt` with `pulp` and add README documentation for running the optimizer.