---
name: nba-diversification-knobs
overview: Add max-flex-overlap and CPT field-cap controls for diversification to the NBA pipeline and wire them into the run_full_nba.sh script.
todos: []
---

# Add diversification knobs to NBA full-run pipeline

### 1. Mirror NFL diversification options in NBA CLI

- **Inspect NFL diversification wrapper**: Review `src/nfl/diversify_lineups.py` to document how `max_flex_overlap` and `cpt_field_cap_multiplier` are exposed via CLI flags and passed into `shared.diversify_core.run_diversify`.
- **Extend NBA diversify wrapper**: Update [`src/nba/diversify_lineups_nba.py`](src/nba/diversify_lineups_nba.py) to accept the same additional parameters in `run(...)` (e.g., `max_flex_overlap`, `cpt_field_cap_multiplier`, `lineups_excel`, `output_dir`) and forward them to `diversify_core.run_diversify` and any needed NBA-specific ownership logic.
- **Expand NBA CLI flags**: Add corresponding CLI options to `_parse_args` (e.g., `--max-flex-overlap`, `--cpt-field-cap-multiplier`, optional `--lineups-excel`, `--output-dir`) and wire them into `main()` so they reach the updated `run(...)` function.

### 2. Provide CPT field-cap support for NBA

- **Extract or replicate field-ownership mapping**: Reuse or factor out the logic from [`src/nba/fill_dkentries_nba.py`](src/nba/fill_dkentries_nba.py) that builds per-player projected CPT ownership (`field_own_cpt`) from the lineups workbook `Projections` sheet into a helper that NBA diversification can call.
- **Compute CPT max-share map**: In the NBA `run(...)` wrapper, when `cpt_field_cap_multiplier > 0`, build a `cpt_max_share` dict (player -> max CPT share) analogous to NFL: `max_share = (field_own_cpt / 100) * multiplier`, clipped to 1.0, and pass it into `diversify_core.run_diversify`.
- **Keep sensible defaults**: Choose NBA defaults that mirror current NFL script behavior (e.g., `max_flex_overlap=None` unless specified, `cpt_field_cap_multiplier` default of around `1.5–2.0`), while making them fully overrideable via CLI and the run script.

### 3. Wire new knobs into `run_full_nba.sh`

- **Add optional script arguments**: Extend [`run_full_nba.sh`](run_full_nba.sh) to accept extra optional positional arguments for `MAX_FLEX_OVERLAP` and `CPT_FIELD_CAP_MULTIPLIER` after `DIVERSIFIED_NUM`, with documented defaults that match your preferred NFL settings.
- **Thread flags into NBA diversify CLI call**: Update the Step 3 call to `python -m src.nba.diversify_lineups_nba` to include `--max-flex-overlap` and `--cpt-field-cap-multiplier` only when the corresponding script args are provided (or fall back to defaults inside the script).
- **Optionally support run-scoped outputs**: If desired to match NFL, add a per-run `RUN_DIR` for NBA and pass `--output-dir` (and `--lineups-excel` when needed) so diversified CSV sidecars and ownership summaries can live alongside each run.

### 4. Documentation and validation

- **Update README**: Amend the NBA section in [`README.md`](README.md) to mention the new diversification flags for `src.nba.diversify_lineups_nba` and the additional optional arguments accepted by `run_full_nba.sh`.
- **Sanity checks**: Run the NBA pipeline on a small test slate with different combinations of `max-flex-overlap` and `cpt-field-cap-multiplier` to confirm: (a) constraints are honored (exposure and overlap look reasonable), and (b) NBA outputs remain compatible with `fill_dkentries_nba` (no schema changes).