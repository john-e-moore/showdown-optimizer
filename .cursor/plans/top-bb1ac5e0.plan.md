<!-- bb1ac5e0-73ca-4d02-b37d-70353e6d5b01 ec5f171d-2278-4a80-9601-208a523806c9 -->
## Calibrate the Top 1% Field Threshold Model

### 1. Introduce configurable calibration knobs

- **New parameters**: Add scalar calibration factors to the `top1pct_finish_rate` pipeline (in `src/top1pct_finish_rate.py`):
- `field_var_shrink` \(\alpha \in (0, 1]\): multiplicative shrinkage on the modeled field variance, i.e. `var_field_eff = alpha * var_field`.
- `field_z_score` \(z > 0\): quantile multiplier for the Normal tail; default slightly below the canonical 2.326 (e.g. 2.0–2.1).
- Optionally, `flex_var_factor` \(k \le 5\): factor on FLEX variance to reduce the impact of five independent FLEX draws, e.g. `var_F_total = k * var_F` instead of `5 * var_F`.
- **CLI exposure**: Add optional CLI flags (with sensible defaults tuned for your current slate) such as:
- `--field-var-shrink` (float, default 0.6–0.8),
- `--field-z` (float, default 2.0),
- `--flex-var-factor` (float, default 3.0–5.0).
- **Plumbing**: Thread these parameters from `_parse_args` into `run`, then down into `_compute_field_thresholds` (or a new helper) where the adjusted variance and z are applied.

### 2. Modify the field distribution and threshold computation

- **Variance adjustment**: In `_compute_field_thresholds`, replace:
- `var_F_total = 5.0 * var_F` with `var_F_total = flex_var_factor * var_F`.
- `var_field = var_C + var_F_total` with `var_field_eff = field_var_shrink * (var_C + var_F_total)`.
- **Softened quantile**: Replace the fixed `z_0_99 = 2.326...` with `field_z_score`, and compute:
- `std_field = np.sqrt(var_field_eff)`
- `T_field_0_99[s] = mu_field[s] + field_z_score * std_field[s]`.
- **Defaults aimed at ranking quality**: Start with conservative-but-not-extreme defaults (e.g. `field_var_shrink=0.7`, `field_z=2.0`, `flex_var_factor=3.5`) that will pull the 99th-percentile threshold closer to the best lineup scores while preserving ordering sensitivity across lineups.

### 3. Reuse diagnostics to sanity-check ranking behavior

- **Leverage existing outputs**: Use the already-written diagnostics in `diagnostics/top1pct/` to compare behavior before and after calibration:
- `lineup_scores.csv`: to verify that thresholds now sit closer to the upper tail of candidate scores rather than far beyond them.
- `lineup_top1_summary.csv`: to confirm that:
  - the **best** lineups have noticeably higher `top1_pct_finish_rate` (on the order of 1–5% depending on calibration),
  - mid-tier lineups have lower but non-zero values,
  - the worst lineups are near 0.
- **Empirical tuning loop**: For one or two slates, experiment with a small grid of (`field_var_shrink`, `field_z`, `flex_var_factor`) settings and visually inspect how the *relative ranking* (vs. mean projection, vs. stack type, etc.) behaves until it aligns with your intuition (best lineups ~3% top‑1, weakest just above 0).

### 4. Document calibration controls

- **README update**: Add a brief subsection under the top‑1% pipeline section of `README.md` describing:
- that the field distribution is a calibrated approximation (not a full fake field),
- the meaning of `--field-var-shrink`, `--field-z`, and `--flex-var-factor`,
- recommended starting values and how to tune them to your own historical experience.
- **Inline docstrings/comments**: In `src/top1pct_finish_rate.py`, clearly comment that these knobs are there to get realistic **relative rankings** of lineups by top‑1% finish probability, not necessarily perfectly calibrated absolute probabilities.

### To-dos

- [ ] Add CLI flags and function parameters for field variance shrinkage, z-score, and optional FLEX variance factor in `src/top1pct_finish_rate.py`, threaded into the field threshold computation.
- [ ] Modify `_compute_field_thresholds` (or equivalent) to apply variance shrinkage, FLEX variance factor, and configurable z-score when computing per-simulation 99th percentile field thresholds.
- [ ] Use existing diagnostics (`lineup_scores`, `lineup_top1_summary`) on at least one slate to pick reasonable default calibration values that yield plausible top-1% rates and a sensible ranking of lineups.
- [ ] Update `README.md` and inline comments to explain the calibrated field model, its knobs, and how to interpret the resulting top-1% estimates.