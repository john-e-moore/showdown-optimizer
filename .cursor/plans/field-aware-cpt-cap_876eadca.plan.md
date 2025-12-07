---
name: field-aware-cpt-cap
overview: Introduce a field-ownership-aware CPT exposure cap into the diversification step so that the selected diversified lineups cannot be dominated by a single CPT beyond a configurable multiple of its projected field CPT ownership.
todos:
  - id: extend-core-api
    content: Extend `diversify_core` to accept an optional per-player CPT max-share map and enforce it during greedy selection.
    status: completed
  - id: build-field-caps
    content: In `nfl/diversify_lineups`, load projected field CPT ownership from the lineups workbook and convert it into a CPT max-share cap map using a configurable multiplier.
    status: completed
  - id: wire-nfl-wrapper
    content: Wire `diversify_lineups.run` and its CLI to conditionally build and pass CPT caps into `diversify_core.run_diversify`, with safe fallbacks when ownership data is unavailable.
    status: completed
  - id: update-pipeline
    content: Update `run_full.sh` to pass the new CPT field-cap multiplier flag so end-to-end runs use the field-aware CPT cap.
    status: completed
  - id: validate-and-tune
    content: Re-run the pipeline on a recent slate, verify diversified CPT exposure vs field ownership, and adjust the multiplier if needed.
    status: completed
---

## Field-Ownership-Aware CPT Cap for Diversification

### 1. Extend diversification core to accept CPT caps
- **Update `diversify_core._greedy_diversified_selection`** in `src/shared/diversify_core.py`:
  - Add an optional argument like `cpt_max_share: dict[str, float] | None = None` that maps player name to a maximum allowed share (0–1) within the diversified set.
  - When iterating candidates, extract the CPT name for the row (using existing `_parse_player_name` on `row["cpt"]`).
  - Maintain `cpt_counts: Dict[str, int]` and `selected_count` alongside `selected_rows`.
  - Before accepting a candidate (after overlap checks), compute the hypothetical future CPT share: `(cpt_counts[cpt_name] + 1) / (selected_count + 1)`. If `cpt_max_share` is provided and `future_share > cpt_max_share.get(cpt_name, 1.0)`, skip this lineup and move to the next candidate.
  - Update `cpt_counts` and `selected_count` whenever a lineup is accepted.
- **Update `diversify_core.run_diversify`** to accept an optional `cpt_max_share` parameter and pass it through to `_greedy_diversified_selection`.

### 2. Build field-ownership-based CPT caps in NFL wrapper
- **Add ownership-loading helper in `src/nfl/diversify_lineups.py`**:
  - Implement a small helper (or reuse the logic pattern from `_load_field_ownership_mapping` in `fill_dkentries.py`) that:
    - Resolves the relevant lineups workbook: prefer a run-scoped `lineups_*.xlsx` under `output_dir` when provided (consistent with how `top1pct` is resolved); otherwise fall back to the latest workbook under `outputs/nfl/lineups/`.
    - Reads the `Projections` sheet and checks for required columns (`Name`, `Team`, `My Proj`, `My Own`).
    - Groups by `(Name, Team)` and extracts a CPT field ownership percentage per player (mirroring the grouping and selection logic used in `fill_dkentries._load_field_ownership_mapping`).
- **Derive CPT max-share map from field ownership**:
  - For each player with a CPT field ownership `field_own_cpt` (in percent), compute a max lineup share cap:
    - `max_share = min(1.0, (field_own_cpt / 100.0) * cpt_field_cap_multiplier)` where `cpt_field_cap_multiplier` is a configurable multiplier (e.g. 2.0 by default).
  - Build `cpt_max_share: Dict[str, float]` keyed by raw player name (matching the names used in the lineups CSV and diversification code).
  - For players missing from the ownership map, either omit them from `cpt_max_share` or give them a default cap of `1.0` so they are effectively uncapped.

### 3. Wire NFL diversification to use the CPT caps
- **Extend `src/nfl/diversify_lineups.run` signature** to accept an optional `cpt_field_cap_multiplier: float | None` (or similar) and an optional explicit `lineups_excel` path.
- **Inside `run`**, when calling `diversify_core.run_diversify`:
  - If `cpt_field_cap_multiplier` is `None` or `<= 0`, call the core exactly as today (no CPT caps) to preserve current behavior.
  - Otherwise, resolve the lineups workbook, build the field-ownership map and the `cpt_max_share` dict, and pass `cpt_max_share` into `run_diversify`.
  - Handle errors gracefully: if the lineups workbook or required columns are missing, log/print a clear warning and fall back to running without CPT caps rather than failing the whole run.

### 4. Expose configuration via CLI and pipeline
- **Update `_parse_args` in `src/nfl/diversify_lineups.py`**:
  - Add a new argument such as `--cpt-field-cap-multiplier` (float, default e.g. `2.0`), documented as “Multiple of projected field CPT ownership to use as a max CPT share cap in the diversified set; set `<= 0` to disable.”
  - Optionally add `--lineups-excel` override if you want to decouple from automatic run-directory resolution.
- **Update `run_full.sh` Step 4 call** to `python -m src.nfl.diversify_lineups` to include a chosen value for `--cpt-field-cap-multiplier` so the new behavior is active in the end-to-end pipeline.

### 5. Validate behavior and adjust tuning
- **Sanity-check on your latest run**:
  - Re-run the pipeline with the new CPT cap enabled and inspect `outputs/nfl/runs/<ts>/diversified.csv` to ensure the CPT mix includes more non-Jacobs captains.
  - Confirm that `outputs/nfl/runs/<ts>/ownership.csv` (from `fill_dkentries`) now shows CPT dollar exposure much closer to field targets, especially for previously overrepresented captains.
- **Tune the multiplier**:
  - If CPT exposure is still too concentrated (e.g., Jacob still far above field), lower `--cpt-field-cap-multiplier`.
  - If the algorithm struggles to reach the target number of diversified lineups because caps are too tight, raise the multiplier slightly or accept that you may have fewer than `num_lineups` under extreme constraints.