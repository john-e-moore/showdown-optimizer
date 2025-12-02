<!-- b7dae0a8-de8f-4a34-a53e-87f690e95362 cca3d4a2-4528-4626-9b93-ee71407e3a25 -->
# Flashback output spreadsheet changes

### Scope

Implement TODO items 2–5 for the flashback Excel workbook produced by `src/flashback_sim.py`, using the raw Sabersim CSV for projections and ownership.

### 1. Add `Projections` tab as the first sheet

- **Load raw Sabersim CSV**: In `run()` in [`src/flashback_sim.py`](/home/john/showdown-optimizer/src/flashback_sim.py), read the original Sabersim CSV via `pd.read_csv(sabersim_path)` into `sabersim_raw_df` (separate from the FLEX-only `sabersim_df` used for correlations).
- **Write `Projections` sheet first**: When writing the Excel workbook, call `sabersim_raw_df.to_excel(..., sheet_name="Projections")` **before** the other sheets so the ordering is: `Projections`, `Standings`, `Simulation`, `Entrant summary`, `Player summary`.

### 2. Compute `Sum Ownership` and `Salary-weighted Ownership` on `Simulation`

- **Build per-player CPT/FLEX ownership and salary mapping**:
- From `sabersim_raw_df`, validate presence of `Name`, `Team`, `My Proj`, `My Own`, and `Salary` (similar to the checks in [`src/showdown_optimizer_main.py`](/home/john/showdown-optimizer/src/showdown_optimizer_main.py)).
- Group by `(Name, Team)`, sort each group by `My Proj` descending, treat the first row as CPT and (if present) second row as FLEX, extracting:
- `cpt_own_pct`, `flex_own_pct` from `My Own` (kept in the same percentage units as Sabersim, e.g., `0.67` meaning `0.67%`).
- `cpt_salary`, `flex_salary` from `Salary`.
- Store these in a dict keyed by `(name, team)` for easy lookup.
- **Map contest lineup players to teams**:
- Reuse or extend the existing `name_to_team` mapping already built in `run()` (from `sabersim_df["Name"]`/`["Team"]`) so each `CPT`/`Flex1`–`Flex5` name in `simulation_df` can be paired with its Sabersim team.
- **Per-lineup ownership metrics**:
- For each row of `simulation_df`, compute:
- **`Sum Ownership`**: `cpt_own_pct(CPT_name, team) + sum_j flex_own_pct(Flexj_name, team)` using the CPT/FLEX-specific ownership values.
- **`Salary-weighted Ownership`**: \(\text{SWO}_\text{raw} = cpt\_salary \cdot cpt\_own\_pct + \sum_j flex\_salary_j \cdot flex\_own\_pct_j\).
- **Human-readable scaling**: define the stored `Salary-weighted Ownership` as `SWO = SWO_raw / 1000.0` so values are in "salary-thousands × ownership-percent" units and readable on the sheet; format as a numeric column (user-facing behavior can be tuned later if desired).
- Append both columns to `simulation_df` before writing the `Simulation` sheet.

### 3. Add `Entries` column to `Entrant summary`

- **Extend `_build_entrant_summary`** in `src/flashback_sim.py`:
- Keep the current groupby on `"Entrant"` and mean aggregations for `Top 1%`, `Top 5%`, `Top 20%`, and `Avg Points`.
- Also compute the number of contest entries per entrant via `grouped.size()` or a `count` aggregation and merge it in as an `Entries` column.
- Reorder columns so the DataFrame has: `Entrant`, `Entries`, `Avg. Top 1%`, `Avg. Top 5%`, `Avg. Top 20%`, `Avg Points`.

### 4. Add projected ownership columns to `Player summary`

- **Prepare player-level projected ownership**:
- From the same Sabersim ownership mapping built in step 2, derive a dict keyed by `player_name` with aggregated `cpt_proj_own` and `flex_proj_own` (e.g., summing across teams if that ever occurs; in practice names are unique within a showdown slate).
- **Update `_build_player_summary`** in `src/flashback_sim.py`:
- Change the function signature to accept the projected ownership dict (or a small DataFrame) alongside `simulation_df`.
- When constructing each per-player row (already computing `CPT draft %`, `FLEX draft %`, and role-specific top-1/top-20 rates), look up that player's `cpt_proj_own` and `flex_proj_own` from the Sabersim mapping (defaulting to `0.0` if missing).
- Insert two new columns:
- `CPT proj ownership` immediately after `CPT draft %`.
- `FLEX proj ownership` immediately after `FLEX draft %`.
- Keep values in the same units as the Sabersim `My Own` field (e.g., `0.7` meaning `0.7%`).

### 5. Wiring and documentation

- **Wire helpers in `run()`**: After building `simulation_df`, compute the new per-lineup ownership metrics, then build `entrant_summary_df` and `player_summary_df` using the updated helpers and Sabersim ownership mapping.
- **Update documentation**:
- In [`prompts/flashback.md`](/home/john/showdown-optimizer/prompts/flashback.md), update the description of the `Simulation`, `Entrant summary`, and `Player summary` tabs to mention `Sum Ownership`, `Salary-weighted Ownership`, `Entries`, and the new projected ownership columns.
- Optionally add a short note in [`README.md`](/home/john/showdown-optimizer/README.md) explaining the meaning and scaling of `Salary-weighted Ownership` on the flashback workbook.

### To-dos

- [ ] Load raw Sabersim CSV in flashback_sim.run and add a first-sheet `Projections` tab to the flashback workbook.
- [ ] From raw Sabersim data, build CPT/FLEX ownership and salary mappings and add `Sum Ownership` and scaled `Salary-weighted Ownership` columns to the `Simulation` sheet.
- [ ] Extend `_build_entrant_summary` to include an `Entries` column and place it as the second column in the `Entrant summary` sheet.
- [ ] Extend `_build_player_summary` to incorporate `CPT proj ownership` and `FLEX proj ownership` columns sourced from Sabersim projected ownership.