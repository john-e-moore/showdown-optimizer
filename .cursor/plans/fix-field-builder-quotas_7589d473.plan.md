---
name: fix-field-builder-quotas
overview: Normalize and robustly round CPT/FLEX ownership quotas in the quota-balanced field builder so that total CPT capacity is sufficient for the requested field size, eliminating runtime errors when sampling CPT.
todos: []
---

# Robust CPT/FLEX Quota Fix for Field Builder

## Goals

- **Ensure total CPT quota is at least equal to `field_size`** so the builder never runs out of CPT candidates before generating all field lineups.
- **Make FLEX quotas consistent with 5 slots per lineup**, improving realism of the field composition.
- **Remove the pandas `FutureWarning`** about multi-key indexing in the ownership aggregation.

## High-Level Approach

- Update `_compute_player_quantas` in `src/shared/field_builder.py` to:
- Aggregate per-player ownership as before, but use list-style column selection to avoid the `FutureWarning`.
- Convert CPT and FLEX ownership to raw fractions, handle degenerate cases where totals are zero.
- **Normalize** CPT fractions to sum to 1 (and similarly normalize FLEX fractions to sum to 1 for quota purposes).
- Use a **Hamilton / largest-remainder rounding scheme** to:
  - Allocate integer CPT quotas whose sum is exactly `field_size`.
  - Allocate integer FLEX quotas whose sum is approximately `5 * field_size`.
- Keep the rest of the builder (CPT / FLEX sampling, correlation logic) unchanged so that behavior is identical except for improved quota stability.

## Detailed Steps

### 1. Refactor ownership aggregation and normalization

- In `_compute_player_quantas` in `field_builder.py`:
- Replace the `groupby` indexing `own.groupby("player")["cpt_ownership", "flex_ownership"]` with `own.groupby("player")[["cpt_ownership", "flex_ownership"]]` to eliminate the deprecation warning.
- After aggregation, construct raw ownership fractions:
  - `p_cpt_raw = own["cpt_ownership"] / 100.0`.
  - `p_flex_raw = own["flex_ownership"] / 100.0`.
- Compute totals `total_cpt`, `total_flex`.
  - If `total_cpt <= 0`, fall back to uniform CPT fractions across players.
  - Otherwise, define `p_cpt = p_cpt_raw / total_cpt` so that `sum(p_cpt) == 1`.
  - Similarly, define `p_flex` normalized by `total_flex`, with a uniform fallback when `total_flex <= 0`.

### 2. Implement Hamilton rounding for CPT quotas

- Compute target CPT counts as a float vector: `target_cpt = p_cpt * field_size`.
- Take the floor: `base_cpt = floor(target_cpt)` (integer array).
- Compute `remaining_cpt_slots = field_size - base_cpt.sum()`.
- This may be positive or zero; if negative (due to numerical quirks), clamp to zero.
- Compute fractional remainders: `remainders_cpt = target_cpt - base_cpt`.
- Sort indices by decreasing `remainders_cpt`; for the top `remaining_cpt_slots` indices, increment `base_cpt[idx] += 1`.
- Use `base_cpt` as the final CPT quotas, guaranteeing `sum(base_cpt) == field_size`.

### 3. Implement proportional rounding for FLEX quotas

- Define total flex slots as `total_flex_slots = 5 * field_size`.
- Compute `target_flex = p_flex * total_flex_slots` and `base_flex = floor(target_flex)`.
- Compute `remaining_flex_slots = total_flex_slots - base_flex.sum()` and clamp to non-negative.
- Compute `remainders_flex = target_flex - base_flex` and, as with CPT, distribute the remaining slots by giving `+1` to the players with largest fractional remainders until all remaining slots are assigned.
- Use `base_flex` as the final FLEX quotas; their sum will be exactly `5 * field_size`.

### 4. Wire new quotas into the existing builder

- At the end of `_compute_player_quantas`, map the integer arrays back to dictionaries:
- `cpt_quota[name] = int(base_cpt[i])` and `flex_quota[name] = int(base_flex[i]) `for each row `i` in `own`.
- Leave the rest of `build_quota_balanced_field` and downstream sampling logic unchanged, so they operate on these robust quotas.

### 5. Quick verification / diagnostics (manual)

- After implementation, run small ad-hoc checks (no permanent code needed, but easy to do in a REPL or temporary script):
- For a real slate’s ownership DataFrame and a chosen `field_size`, call `_compute_player_quantas` and verify:
  - `sum(cpt_quota.values()) == field_size`.
  - `sum(flex_quota.values()) == 5 * field_size`.
- Run `python -m src.nfl.top1pct_finish_rate --field-size <contest_size> --num-sims 20000 --field-model explicit` and confirm there is no longer a `RuntimeError` about CPT quotas.
- Optionally, compute realized CPT/FLEX usage in the generated field and compare to input ownerships to sanity check the distribution.