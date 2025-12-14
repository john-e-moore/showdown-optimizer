from __future__ import annotations

"""
Sport-agnostic core for estimating top 1% finish probabilities for
DraftKings Showdown lineups.

This module intentionally has no dependency on sport-specific config
modules. Callers must provide:

  - outputs_dir: root outputs directory for the sport
                 (e.g. outputs/nfl or outputs/nba).

It operates purely on the standard Excel workbook schemas produced by
the optimizer and correlation pipelines.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .field_builder import build_quota_balanced_field
from .dk_contest_api import load_contest_payouts_and_size


LINEUPS_SHEET_NAME = "Lineups"
OWNERSHIP_SHEET_NAME = "Ownership"
PROJECTIONS_SHEET_NAME = "Projections"
CORR_PROJECTIONS_SHEET_NAME = "Sabersim_Projections"
CORR_MATRIX_SHEET_NAME = "Correlation_Matrix"


def _resolve_latest_excel(directory: Path, explicit: str | None) -> Path:
    """
    Resolve an Excel file path, preferring an explicit argument when provided.

    If explicit is None, pick the most recent *.xlsx file in `directory`.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Specified Excel file does not exist: {path}")
        return path

    candidates = sorted(directory.glob("*.xlsx"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .xlsx files found in directory: {directory}")
    return candidates[-1]


def _parse_player_name(cell: str) -> str:
    """
    Extract the raw player name from a lineup cell like 'Deebo Samuel (34.5%)'.
    """
    if not isinstance(cell, str):
        return str(cell)
    # Split at the first ' (' if present.
    split_idx = cell.find(" (")
    if split_idx == -1:
        return cell.strip()
    return cell[:split_idx].strip()


def _load_lineups_workbook(
    path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load lineups, ownership, and projections from a lineups workbook.

    Returns:
        lineups_df: DataFrame from 'Lineups' sheet.
        ownership_df: DataFrame from 'Ownership' sheet.
        projections_df: DataFrame from 'Projections' sheet.
    """
    xls = pd.ExcelFile(path)

    try:
        lineups_df = pd.read_excel(xls, sheet_name=LINEUPS_SHEET_NAME)
    except ValueError as exc:  # sheet not found
        raise KeyError(
            f"Lineups workbook missing '{LINEUPS_SHEET_NAME}' sheet: {path}"
        ) from exc

    try:
        ownership_df = pd.read_excel(xls, sheet_name=OWNERSHIP_SHEET_NAME)
    except ValueError as exc:
        raise KeyError(
            f"Lineups workbook missing '{OWNERSHIP_SHEET_NAME}' sheet: {path}"
        ) from exc

    try:
        projections_df = pd.read_excel(xls, sheet_name=PROJECTIONS_SHEET_NAME)
    except ValueError as exc:
        raise KeyError(
            f"Lineups workbook missing '{PROJECTIONS_SHEET_NAME}' sheet: {path}"
        ) from exc

    return lineups_df, ownership_df, projections_df


def _load_corr_workbook(
    path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Sabersim projections and correlation matrix from a correlation workbook.

    Returns:
        sabersim_proj_df: flex-style Sabersim projections (non-CPT slots).
        corr_df: square correlation matrix with player names as index/columns.
    """
    xls = pd.ExcelFile(path)

    try:
        sabersim_proj_df = pd.read_excel(
            xls, sheet_name=CORR_PROJECTIONS_SHEET_NAME
        )
    except ValueError as exc:
        raise KeyError(
            f"Correlation workbook missing '{CORR_PROJECTIONS_SHEET_NAME}' "
            f"sheet: {path}"
        ) from exc

    try:
        corr_df = pd.read_excel(
            xls,
            sheet_name=CORR_MATRIX_SHEET_NAME,
            index_col=0,
        )
    except ValueError as exc:
        raise KeyError(
            f"Correlation workbook missing '{CORR_MATRIX_SHEET_NAME}' sheet: {path}"
        ) from exc

    # Ensure correlation matrix is numeric and square.
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")
    if corr_df.shape[0] != corr_df.shape[1]:
        raise ValueError(
            f"Correlation matrix is not square: shape={corr_df.shape} in {path}"
        )

    return sabersim_proj_df, corr_df


def _resolve_non_cpt_slot_columns(lineups_df: pd.DataFrame) -> List[str]:
    """
    Resolve non-CPT slot column names for a Showdown lineup DataFrame.

    Supports both legacy FLEX-style naming (flex1–flex5) and UTIL-style naming
    (util1–util5, used by NBA going forward).
    """
    flex_cols = [f"flex{j}" for j in range(1, 6)]
    util_cols = [f"util{j}" for j in range(1, 6)]

    flex_missing = [c for c in flex_cols if c not in lineups_df.columns]
    util_missing = [c for c in util_cols if c not in lineups_df.columns]

    if not flex_missing:
        return flex_cols
    if not util_missing:
        return util_cols

    raise KeyError(
        "Lineups sheet missing expected non-CPT slot columns. "
        f"Expected either FLEX-style {flex_cols} or UTIL-style {util_cols}."
    )


def _build_player_universe(
    lineups_df: pd.DataFrame,
    ownership_df: pd.DataFrame,
    sabersim_proj_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    lineups_proj_df: pd.DataFrame,
) -> Tuple[
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build unified player universe and aligned arrays.

    Returns:
        player_names: list of player names in model order.
        mu: DK point means (shape P).
        sigma: DK point standard deviations (shape P).
        cpt_own: CPT ownership fractions (shape P).
        flex_own: flex-style ownership fractions (shape P).
        positions: string position labels (shape P).
    """
    # Players from correlation matrix.
    corr_players = [str(x) for x in corr_df.index.tolist()]

    # Players from Sabersim projections in correlation workbook.
    if "Name" not in sabersim_proj_df.columns:
        raise KeyError(
            "Correlation Sabersim projections sheet missing 'Name' column."
        )
    sabersim_names = sabersim_proj_df["Name"].astype(str).tolist()

    # Players from lineups sheet.
    lineup_player_names: List[str] = []
    non_cpt_cols = _resolve_non_cpt_slot_columns(lineups_df)
    lineup_cols = ["cpt"] + non_cpt_cols

    for _, row in lineups_df.iterrows():
        for col in lineup_cols:
            name = _parse_player_name(row[col])
            lineup_player_names.append(name)

    # Union of all relevant players (preserving correlation matrix order first).
    universe_names: List[str] = []
    seen: set[str] = set()

    def _add_name(n: str) -> None:
        if n not in seen:
            seen.add(n)
            universe_names.append(n)

    for n in corr_players:
        _add_name(n)
    for n in sabersim_names:
        _add_name(n)
    for n in lineup_player_names:
        _add_name(n)

    # ------------------------------------------------------------------
    # Build mean, std dev, and position from projections sources.
    # ------------------------------------------------------------------
    sabersim_proj_df = sabersim_proj_df.copy()
    if "My Proj" not in sabersim_proj_df.columns:
        raise KeyError(
            "Correlation Sabersim projections sheet missing 'My Proj' column."
        )
    sabersim_proj_df["Name"] = sabersim_proj_df["Name"].astype(str)

    # Handle possible duplicate Name entries (e.g. CPT vs non-CPT rows from
    # a raw Sabersim export). For our purposes we want a single flex-style
    # row per player, so when duplicates exist we keep the *lower* projection
    # row and drop the higher projection rows (which correspond to CPT rows
    # in typical Sabersim Showdown outputs).
    if sabersim_proj_df["Name"].duplicated().any():
        before = len(sabersim_proj_df)
        sabersim_proj_df = (
            sabersim_proj_df.sort_values(
                by=["Name", "My Proj"], ascending=[True, True]
            )
            .groupby("Name", as_index=False)
            .first()
        )
        after = len(sabersim_proj_df)
        dropped = before - after
        if dropped > 0:
            print(
                "Warning: Sabersim_Projections sheet contained duplicate player "
                f"names; dropped {dropped} higher-projection CPT rows, keeping "
                "one flex-style row per player."
            )

    corr_proj_indexed = sabersim_proj_df.set_index("Name")

    # Optional dk_std / Pos columns from correlation projections.
    if "dk_std" in corr_proj_indexed.columns:
        dk_std_corr = corr_proj_indexed["dk_std"].astype(float)
    else:
        dk_std_corr = pd.Series(index=corr_proj_indexed.index, dtype=float)
    if "Pos" in corr_proj_indexed.columns:
        pos_corr = corr_proj_indexed["Pos"].astype(str)
    else:
        pos_corr = pd.Series(index=corr_proj_indexed.index, dtype=str)

    # Secondary projections source from the lineup optimizer workbook.
    lineups_proj_df = lineups_proj_df.copy()
    required_lp = ["Name", "Team", "Salary", "My Proj"]
    missing_lp = [c for c in required_lp if c not in lineups_proj_df.columns]
    if missing_lp:
        raise KeyError(
            "Lineups projections sheet missing required columns "
            f"{missing_lp}."
        )
    lineups_proj_df["Name"] = lineups_proj_df["Name"].astype(str)
    lineups_proj_df["Team"] = lineups_proj_df["Team"].astype(str)
    # Keep lowest-salary row per (Name, Team) as flex-style (non-CPT) equivalent.
    lineups_proj_df = (
        lineups_proj_df.sort_values(
            by=["Name", "Team", "Salary"], ascending=[True, True, True]
        )
        .groupby(["Name", "Team"], as_index=False)
        .first()
    )
    lp_indexed = lineups_proj_df.set_index("Name")

    mu_corr = corr_proj_indexed["My Proj"].astype(float)
    mu_lp = lp_indexed["My Proj"].astype(float)

    # Optional dk_std / Pos from lineup projections.
    if "dk_std" in lp_indexed.columns:
        dk_std_lp = lp_indexed["dk_std"].astype(float)
    else:
        dk_std_lp = pd.Series(index=lp_indexed.index, dtype=float)
    if "Pos" in lp_indexed.columns:
        pos_lp = lp_indexed["Pos"].astype(str)
    else:
        pos_lp = pd.Series(index=lp_indexed.index, dtype=str)

    # Combine dk_std from correlation projections, then lineup projections.
    dk_std_series = dk_std_corr.combine_first(dk_std_lp)

    # Combine positions with preference for correlation projections, then
    # lineup projections.
    pos_series = pos_corr.combine_first(pos_lp)

    mu_list: List[float] = []
    sigma_list: List[float] = []
    pos_list: List[str] = []
    salary_list: List[float] = []
    team_list: List[str] = []

    for name in universe_names:
        if name in mu_corr.index:
            mu_val = float(mu_corr.at[name])
        elif name in mu_lp.index:
            mu_val = float(mu_lp.at[name])
        else:
            # If the player is completely unknown to projections, fall back to 0.
            mu_val = 0.0

        if name in dk_std_series.index and pd.notna(dk_std_series.at[name]):
            sigma_val = float(dk_std_series.at[name])
        else:
            # Heuristic fallback: standard deviation proportional to mean.
            # Ensure strictly positive to avoid degenerate covariance.
            sigma_val = max(1.0, 0.7 * max(mu_val, 0.0))

        if name in pos_series.index and pd.notna(pos_series.at[name]):
            pos_val = str(pos_series.at[name])
        else:
            pos_val = ""

        mu_list.append(mu_val)
        sigma_list.append(sigma_val)
        pos_list.append(pos_val)
        # Flex-style salary and team from lineup projections when available.
        if name in lp_indexed.index:
            salary_val = float(lp_indexed.at[name, "Salary"])
            team_val = str(lp_indexed.at[name, "Team"])
        else:
            salary_val = 0.0
            team_val = ""
        salary_list.append(salary_val)
        team_list.append(team_val)

    mu = np.array(mu_list, dtype=float)
    sigma = np.array(sigma_list, dtype=float)
    positions = np.array(pos_list, dtype=object)
    salaries = np.array(salary_list, dtype=float)
    teams = np.array(team_list, dtype=object)

    # Ownership: convert percentage columns to fractions.
    expected_own_cols = {"player", "cpt_ownership", "flex_ownership"}
    missing_own_cols = expected_own_cols.difference(ownership_df.columns)
    if missing_own_cols:
        raise KeyError(
            "Ownership sheet missing required columns "
            f"{sorted(missing_own_cols)}."
        )

    own_df = ownership_df.copy()
    own_df["player"] = own_df["player"].astype(str)
    own_df["cpt_own"] = own_df["cpt_ownership"].astype(float) / 100.0
    own_df["flex_own"] = own_df["flex_ownership"].astype(float) / 100.0

    # Aggregate ownership by player name.
    own_by_name = (
        own_df.groupby("player")[["cpt_own", "flex_own"]]
        .sum()
        .reset_index()
    )
    own_by_name.set_index("player", inplace=True)

    cpt_own_list: List[float] = []
    flex_own_list: List[float] = []

    for name in universe_names:
        if name in own_by_name.index:
            row = own_by_name.loc[name]
            cpt_own_list.append(float(row["cpt_own"]))
            flex_own_list.append(float(row["flex_own"]))
        else:
            cpt_own_list.append(0.0)
            flex_own_list.append(0.0)

    cpt_own = np.array(cpt_own_list, dtype=float)
    flex_own = np.array(flex_own_list, dtype=float)

    return universe_names, mu, sigma, cpt_own, flex_own, positions, salaries, teams


def _build_full_correlation(
    universe_names: List[str],
    corr_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build a full correlation matrix over the unified player universe.

    Players that are not present in the input matrix are treated as
    independent (corr=0 with others, corr=1 with self).
    """
    base_names = [str(x) for x in corr_df.index.tolist()]
    base_index: Dict[str, int] = {name: i for i, name in enumerate(base_names)}

    n_total = len(universe_names)
    corr_full = np.eye(n_total, dtype=float)
    base_corr = corr_df.to_numpy(dtype=float)

    # Copy over the known correlation block for players present in corr_df.
    for i_univ, name_i in enumerate(universe_names):
        i_base = base_index.get(name_i)
        if i_base is None:
            continue
        for j_univ, name_j in enumerate(universe_names):
            j_base = base_index.get(name_j)
            if j_base is None:
                continue
            corr_full[i_univ, j_univ] = float(base_corr[i_base, j_base])

    # Ensure symmetry and unit diagonal.
    corr_full = 0.5 * (corr_full + corr_full.T)
    np.fill_diagonal(corr_full, 1.0)
    return corr_full


def _make_psd(cov: np.ndarray) -> np.ndarray:
    """
    Ensure covariance matrix is positive semi-definite by adjusting the diagonal.
    """
    # Try a Cholesky factorization; if it fails, bump the diagonal.
    for _ in range(5):
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = float(eigvals.min())
            if min_eig >= 0.0:
                # Numerical issue only; add a tiny jitter.
                jitter = 1e-8
            else:
                jitter = -min_eig + 1e-8
            cov = cov + np.eye(cov.shape[0], dtype=float) * jitter
    # Final attempt; may still raise, which is preferable to silently failing.
    np.linalg.cholesky(cov)
    return cov


def _simulate_player_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_full: np.ndarray,
    num_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw correlated player DK scores from a multivariate normal.

    Returns:
        X: array of shape (num_sims, P) with DK scores.
    """
    # Build covariance from std devs and correlation.
    outer = np.outer(sigma, sigma)
    cov = corr_full * outer
    cov = _make_psd(cov)

    X = rng.multivariate_normal(mean=mu, cov=cov, size=num_sims)
    return X


def _compute_field_thresholds(
    X: np.ndarray,
    cpt_own: np.ndarray,
    flex_own: np.ndarray,
    *,
    field_var_shrink: float,
    field_z_score: float,
    flex_var_factor: float,
) -> np.ndarray:
    """
    Compute approximate 99th percentile thresholds of the field score distribution.

    This models the field as a mixture of CPT and flex-style roster slots. The
    flex component is scaled by ``flex_var_factor`` to calibrate its variance,
    regardless of whether those slots are labelled FLEX (NFL) or UTIL (NBA)
    in downstream DraftKings artifacts.

    Args:
        X: player DK scores, shape (S, P).
        cpt_own: CPT ownership fractions, shape (P,).
        flex_own: flex-style ownership fractions aggregated over non-CPT slots.

    Returns:
        thresholds: array of shape (S,) with per-simulation 99th percentile scores.
    """
    S, P = X.shape
    if cpt_own.shape[0] != P or flex_own.shape[0] != P:
        raise ValueError("Ownership arrays must match player dimension.")

    # CPT mixture: scores are 1.5 * X[:, i] with probabilities cpt_own[i].
    cpt_scores = 1.5 * X  # shape (S, P)
    mu_C = cpt_scores @ cpt_own  # shape (S,)
    mC2 = (cpt_scores ** 2) @ cpt_own
    var_C = mC2 - mu_C**2
    var_C = np.clip(var_C, a_min=0.0, a_max=None)

    # Flex-style mixture: per-slot distribution, nominally 5 slots but with an
    # optional variance calibration factor to soften the tail.
    pi = flex_own / 5.0
    flex_scores = X  # DK points at non-CPT flex-style slots are just X.
    mu_F = flex_scores @ pi
    mF2 = (flex_scores ** 2) @ pi
    var_F = mF2 - mu_F**2
    var_F = np.clip(var_F, a_min=0.0, a_max=None)

    mu_F_total = 5.0 * mu_F
    # Calibrated flex-style variance factor k (<= 5.0 ideally).
    var_F_total = flex_var_factor * var_F

    mu_field = mu_C + mu_F_total
    var_field = var_C + var_F_total
    var_field = np.clip(var_field, a_min=0.0, a_max=None)

    # Calibrated field variance shrinkage and z-score for the tail.
    var_field_eff = field_var_shrink * var_field
    var_field_eff = np.clip(var_field_eff, a_min=0.0, a_max=None)

    std_field = np.sqrt(var_field_eff)
    thresholds = mu_field + field_z_score * std_field
    return thresholds


def _build_lineup_weights(
    lineups_df: pd.DataFrame,
    player_index: Dict[str, int],
) -> np.ndarray:
    """
    Build weight matrix W of shape (K, P) from lineups DataFrame.

    Each row corresponds to a lineup:
      - CPT player has weight 1.5.
      - Each flex-style player has weight 1.0 (weights add if duplicated).
    """
    non_cpt_cols = _resolve_non_cpt_slot_columns(lineups_df)
    lineup_cols = ["cpt"] + non_cpt_cols

    K = len(lineups_df)
    P = len(player_index)
    W = np.zeros((K, P), dtype=float)

    unknown_players: set[str] = set()

    for k, (_, row) in enumerate(lineups_df.iterrows()):
        # CPT
        cpt_name = _parse_player_name(row["cpt"])
        idx = player_index.get(cpt_name)
        if idx is None:
            unknown_players.add(cpt_name)
        else:
            W[k, idx] += 1.5

        # Flex/UTIL-style slots
        for col in non_cpt_cols:
            flex_name = _parse_player_name(row[col])
            idx = player_index.get(flex_name)
            if idx is None:
                unknown_players.add(flex_name)
            else:
                W[k, idx] += 1.0

    if unknown_players:
        names_sorted = ", ".join(sorted(unknown_players))
        raise KeyError(
            "Some lineup players were not found in the player universe built "
            f"from correlations and projections: {names_sorted}"
        )

    return W


def _annotate_lineups_with_meta(
    lineups_df: pd.DataFrame,
    player_names: List[str],
    mu: np.ndarray,
    salaries: np.ndarray,
    teams: np.ndarray,
    *,
    sport: str | None = None,
) -> pd.DataFrame:
    """
    Annotate a CPT + flex/UTIL lineups DataFrame with projection/salary/stack metadata.

    Returns a DataFrame with:
      - lineup_projection: 1.5× CPT projection + 1× each non-CPT slot projection.
      - lineup_salary: same weighting pattern using flex-style salaries.
      - stack: team-count pattern like "4|2" inferred from team counts.
      - stack_pattern: reserved label column (empty for field-generated lineups).
      - CPT and slot columns preserved from the input, with NBA slots exposed as
        util1–util5 for convenience when sport == "nba".
    """
    if len(player_names) != mu.shape[0] or len(player_names) != salaries.shape[0]:
        raise ValueError("Player arrays must have matching length.")

    name_to_mu = {name: float(mu[i]) for i, name in enumerate(player_names)}
    name_to_salary = {name: float(salaries[i]) for i, name in enumerate(player_names)}
    name_to_team = {name: str(teams[i]) for i, name in enumerate(player_names)}

    non_cpt_cols = _resolve_non_cpt_slot_columns(lineups_df)

    records: List[Dict[str, object]] = []
    for _, row in lineups_df.iterrows():
        cpt_name = _parse_player_name(row["cpt"])
        slot_names: List[str] = []
        for col in non_cpt_cols:
            slot_names.append(_parse_player_name(row[col]))

        proj_cpt = 1.5 * name_to_mu.get(cpt_name, 0.0)
        proj_flex = sum(name_to_mu.get(n, 0.0) for n in slot_names)
        lineup_proj = proj_cpt + proj_flex

        sal_cpt = 1.5 * name_to_salary.get(cpt_name, 0.0)
        sal_flex = sum(name_to_salary.get(n, 0.0) for n in slot_names)
        lineup_salary = sal_cpt + sal_flex

        teams_in_lineup = [
            name_to_team.get(cpt_name, ""),
            *[name_to_team.get(n, "") for n in slot_names],
        ]
        team_counts: Dict[str, int] = {}
        for t in teams_in_lineup:
            if not t:
                continue
            team_counts[t] = team_counts.get(t, 0) + 1
        counts_sorted = sorted(team_counts.values(), reverse=True)
        stack_str = "|".join(str(c) for c in counts_sorted) if counts_sorted else ""

        record: Dict[str, object] = {
            "lineup_projection": lineup_proj,
            "lineup_salary": lineup_salary,
            "stack": stack_str,
            "stack_pattern": "",
            "cpt": row["cpt"],
        }
        for col in non_cpt_cols:
            record[col] = row[col]

        records.append(record)

    annotated = pd.DataFrame(records)

    # For NBA, prefer UTIL-style naming in outputs even if the internal columns
    # used FLEX-style names.
    if sport == "nba":
        rename_map: Dict[str, str] = {}
        for j in range(1, 6):
            flex_col = f"flex{j}"
            util_col = f"util{j}"
            if flex_col in annotated.columns and util_col not in annotated.columns:
                rename_map[flex_col] = util_col
        if rename_map:
            annotated = annotated.rename(columns=rename_map)

    meta_cols = ["lineup_projection", "lineup_salary", "stack", "stack_pattern"]
    slot_cols: List[str] = ["cpt"] + [
        c for c in annotated.columns if c not in set(meta_cols + ["cpt"])
    ]
    annotated = annotated[meta_cols + slot_cols]
    return annotated


def run_top1pct(
    field_size: int | None,
    outputs_dir: Path,
    *,
    lineups_excel: str | None = None,
    corr_excel: str | None = None,
    num_sims: int = 20_000,
    random_seed: int | None = None,
    field_var_shrink: float = 0.7,
    field_z_score: float = 2.0,
    flex_var_factor: float = 3.5,
    field_model: str = "mixture",
    run_dir: Path | None = None,
    # Optional contest-based EV ROI computation
    contest_id: str | None = None,
    payouts_json: str | None = None,
    data_dir: Path | None = None,
    sim_batch_size: int = 200,
) -> Path:
    """
    Execute the top 1% finish rate estimation pipeline for a given sport.

    Args:
        field_size: Total number of lineups in the contest field. When contest_id
            is provided, field_size may be omitted (None) and will be inferred
            from DraftKings contest metadata.
        outputs_dir: Root outputs directory for the sport
                     (e.g. outputs/nfl or outputs/nba).
        lineups_excel: Optional explicit path to a lineups workbook.
        corr_excel: Optional explicit path to a correlation workbook.
        num_sims: Number of Monte Carlo simulations.
        random_seed: Optional RNG seed; if None, use nondeterministic seed.
        field_var_shrink: Variance shrinkage factor for the modeled field when
            using the analytic ownership-mixture field model.
        field_z_score: Z-score for the upper tail of the analytic field
            distribution.
        flex_var_factor: Effective variance factor for the aggregate flex-style
            component in the analytic field model.
        field_model: Strategy for modeling the contest field:
            - \"mixture\": current analytic ownership-mixture approximation.
            - \"explicit\": simulate a quota-balanced field of lineups and take
              empirical 99th-percentile thresholds from those scores.
        run_dir: Optional explicit directory for the output workbook. When
            provided, the top1pct workbook is written directly into this
            directory instead of outputs_dir / \"top1pct\".
        contest_id: Optional DraftKings contest id. When provided, we will fetch
            payout structure and compute per-lineup EV payout + EV ROI using an
            explicit field model.
        payouts_json: Optional path to a cached DK contest JSON file. When
            provided, bypasses the download and reads this file directly.
        data_dir: Data directory for the sport (e.g. data/nfl or data/nba). This
            is required when contest_id is provided so we can cache contest JSON.
        sim_batch_size: Batch size for simulation streaming. Smaller values use
            less memory but may be slower.

    Returns:
        Path to the written output Excel workbook.
    """
    lineups_dir = outputs_dir / "lineups"
    corr_dir = outputs_dir / "correlations"

    lineups_path = _resolve_latest_excel(lineups_dir, lineups_excel)
    corr_path = _resolve_latest_excel(corr_dir, corr_excel)

    print(f"Using lineups workbook: {lineups_path}")
    print(f"Using correlation workbook: {corr_path}")

    lineups_df, ownership_df, lineups_proj_df = _load_lineups_workbook(lineups_path)
    sabersim_proj_df, corr_df = _load_corr_workbook(corr_path)

    (
        player_names,
        mu,
        sigma,
        cpt_own,
        flex_own,
        positions,
        salaries,
        teams,
    ) = _build_player_universe(
        lineups_df=lineups_df,
        ownership_df=ownership_df,
        sabersim_proj_df=sabersim_proj_df,
        corr_df=corr_df,
        lineups_proj_df=lineups_proj_df,
    )

    corr_full = _build_full_correlation(player_names, corr_df)
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Optional contest metadata for EV payout / EV ROI.
    # ------------------------------------------------------------------
    entry_fee: float | None = None
    payouts_full: np.ndarray | None = None
    payout_prefix: np.ndarray | None = None
    paid_places: int = 0

    inferred_field_size: int | None = None
    if contest_id is not None:
        if data_dir is None:
            raise ValueError("data_dir must be provided when contest_id is set.")

        dk_field_size, dk_entry_fee, dk_payouts = load_contest_payouts_and_size(
            str(contest_id),
            data_dir=data_dir,
            payouts_json=payouts_json,
        )
        inferred_field_size = int(dk_field_size)
        entry_fee = float(dk_entry_fee)

        if field_size is None:
            field_size = inferred_field_size
        else:
            # Allow explicit override but warn if it differs from DK metadata.
            if int(field_size) != inferred_field_size:
                print(
                    "Warning: --field-size does not match DraftKings contest metadata "
                    f"({field_size} != {inferred_field_size}). Proceeding with "
                    f"field_size={field_size} for simulation."
                )

        # Normalize payouts to the simulation field size.
        payouts_full = np.zeros(int(field_size), dtype=float)
        dk_arr = np.asarray(dk_payouts, dtype=float).reshape(-1)
        n_copy = min(payouts_full.size, dk_arr.size)
        if n_copy > 0:
            payouts_full[:n_copy] = dk_arr[:n_copy]
        paid_places = int(np.sum(payouts_full > 0.0))
        payout_prefix = np.concatenate([[0.0], np.cumsum(payouts_full)])

        # ROI needs an explicit field to define rank→payout; auto-upgrade.
        if field_model == "mixture":
            print(
                "contest_id provided; switching field_model from 'mixture' to "
                "'explicit' to enable EV ROI computation."
            )
            field_model = "explicit"

    if field_size is None:
        raise ValueError(
            "field_size is required unless contest_id is provided (to infer it)."
        )

    if sim_batch_size <= 0:
        raise ValueError("sim_batch_size must be positive.")

    print(
        f"Simulating correlated DK scores for {len(player_names)} players across "
        f"{num_sims} simulations..."
    )
    X = _simulate_player_scores(
        mu=mu,
        sigma=sigma,
        corr_full=corr_full,
        num_sims=num_sims,
        rng=rng,
    )

    # Floor simulated DK points at 0 for all non-DST, non-K positions.
    non_dst_k_mask = ~np.isin(positions, ["DST", "K"])
    X[:, non_dst_k_mask] = np.maximum(X[:, non_dst_k_mask], 0.0)

    player_index = {name: i for i, name in enumerate(player_names)}
    print("Building lineup weight matrices...")
    W = _build_lineup_weights(lineups_df=lineups_df, player_index=player_index)

    if field_model not in {"mixture", "explicit"}:
        raise ValueError(
            f"Unsupported field_model '{field_model}'. Expected 'mixture' or 'explicit'."
        )

    field_lineups_with_meta: pd.DataFrame | None = None
    field_lineups_df: pd.DataFrame | None = None
    field_meta_payload: Dict[str, pd.DataFrame] | None = None

    thresholds: np.ndarray | None = None
    W_field: np.ndarray | None = None
    K_field: int = 0

    if field_model == "mixture":
        print("Computing field 99th percentile thresholds from ownership (mixture model)...")
        thresholds = _compute_field_thresholds(
            X=X,
            cpt_own=cpt_own,
            flex_own=flex_own,
            field_var_shrink=field_var_shrink,
            field_z_score=field_z_score,
            flex_var_factor=flex_var_factor,
        )
    else:
        print("Building explicit quota-balanced field lineups...")

        # For contest-id ROI, treat the explicit field as *opponents* so that
        # (opponents + your lineup) approximates the DK contest field size.
        effective_field_size = int(field_size)
        if contest_id is not None:
            effective_field_size = max(1, int(field_size) - 1)

        field_lineups_df = build_quota_balanced_field(
            field_size=effective_field_size,
            ownership_df=ownership_df,
            sabersim_proj_df=sabersim_proj_df,
            lineups_proj_df=lineups_proj_df,
            corr_df=corr_df,
            random_seed=random_seed,
        )
        W_field = _build_lineup_weights(
            lineups_df=field_lineups_df,
            player_index=player_index,
        )
        K_field = int(W_field.shape[0])
        if K_field == 0:
            raise ValueError("Explicit field builder produced zero field lineups.")

        # Build a Field Lineups sheet with per-lineup metadata for the explicit
        # field model. Use a simple sport hint based on the outputs_dir.
        sport_hint = outputs_dir.name.lower() if outputs_dir.name else None
        field_lineups_with_meta = _annotate_lineups_with_meta(
            lineups_df=field_lineups_df,
            player_names=player_names,
            mu=mu,
            salaries=salaries,
            teams=teams,
            sport=sport_hint,
        )

        # Build Field Meta payload (ownership + averages + stack distribution).
        non_cpt_cols = _resolve_non_cpt_slot_columns(field_lineups_df)
        cpt_series = field_lineups_df["cpt"].apply(_parse_player_name)
        cpt_counts = (
            cpt_series.value_counts()
            .rename_axis("player")
            .reset_index(name="count")
        )
        cpt_counts["pct_lineups"] = 100.0 * cpt_counts["count"] / float(len(field_lineups_df))

        flex_series = pd.concat(
            [field_lineups_df[c].apply(_parse_player_name) for c in non_cpt_cols],
            ignore_index=True,
        )
        denom_flex = float(len(field_lineups_df) * len(non_cpt_cols)) if non_cpt_cols else 1.0
        flex_counts = (
            flex_series.value_counts()
            .rename_axis("player")
            .reset_index(name="count")
        )
        flex_counts["pct_slots"] = 100.0 * flex_counts["count"] / denom_flex

        avg_proj = float(field_lineups_with_meta["lineup_projection"].mean())
        avg_salary = float(field_lineups_with_meta["lineup_salary"].mean())
        summary_df = pd.DataFrame(
            [
                {"key": "field_size", "value": int(len(field_lineups_df))},
                {"key": "avg_lineup_projection", "value": avg_proj},
                {"key": "avg_lineup_salary", "value": avg_salary},
            ]
        )

        stack_series = field_lineups_with_meta["stack"].fillna("").astype(str)
        stack_counts = (
            stack_series.value_counts()
            .rename_axis("stack")
            .reset_index(name="count")
        )
        stack_counts["pct_lineups"] = 100.0 * stack_counts["count"] / float(len(field_lineups_df))

        field_meta_payload = {
            "summary": summary_df,
            "cpt_ownership": cpt_counts,
            "flex_ownership": flex_counts,
            "stack_distribution": stack_counts,
        }
    # ------------------------------------------------------------------
    # Streaming scoring: compute top1%, mean/std, and optional EV payout/ROI
    # without materializing (num_sims x num_lineups) matrices.
    # ------------------------------------------------------------------
    num_lineups = int(W.shape[0])
    top1_hits = np.zeros(num_lineups, dtype=np.int64)
    mean_scores = np.zeros(num_lineups, dtype=float)
    m2_scores = np.zeros(num_lineups, dtype=float)
    payout_sums = np.zeros(num_lineups, dtype=float) if contest_id is not None else None

    if field_model == "mixture" and thresholds is None:
        raise RuntimeError("Internal error: thresholds not computed for mixture model.")
    if field_model == "explicit" and W_field is None:
        raise RuntimeError("Internal error: W_field not built for explicit model.")

    print("Scoring lineups across simulations (streaming)...")
    n_seen = 0
    for start in range(0, num_sims, sim_batch_size):
        end = min(num_sims, start + sim_batch_size)
        X_batch = X[start:end]  # (B, P)
        B = int(X_batch.shape[0])

        scores_cand = X_batch @ W.T  # (B, K)

        # Welford updates for mean/std.
        for i in range(B):
            n_seen += 1
            row = scores_cand[i]
            delta = row - mean_scores
            mean_scores += delta / float(n_seen)
            delta2 = row - mean_scores
            m2_scores += delta * delta2

        if field_model == "mixture":
            thr = thresholds[start:end]  # (B,)
            top1_hits += (scores_cand >= thr[:, None]).sum(axis=0)
        else:
            # Explicit field: compute per-sim 99th percentile threshold from
            # simulated field lineups for this batch.
            scores_field = X_batch @ W_field.T  # (B, K_field)

            q_index = max(0, min(K_field - 1, int(np.floor(0.99 * K_field))))
            thr = np.partition(scores_field, q_index, axis=1)[:, q_index]
            top1_hits += (scores_cand >= thr[:, None]).sum(axis=0)

            # Optional: EV payout / EV ROI against DK contest payout structure.
            if contest_id is not None and payouts_full is not None and payout_prefix is not None:
                if paid_places > 0 and entry_fee is not None and entry_fee > 0.0:
                    # For each sim-row, use the top paid_places field scores to
                    # efficiently compute ranks/payouts for candidate lineups.
                    for i in range(B):
                        field_row = scores_field[i]
                        cand_row = scores_cand[i]

                        # Handle small fields gracefully.
                        M = min(int(paid_places), int(field_row.shape[0]))
                        if M <= 0:
                            continue

                        # Largest M field scores (opponents) for this sim.
                        kth = int(field_row.shape[0]) - M
                        topM = np.partition(field_row, kth)[kth:]
                        topM_sorted = np.sort(topM)  # ascending
                        cutoff = float(topM_sorted[0])

                        paid_mask = cand_row >= cutoff
                        if not np.any(paid_mask):
                            continue

                        # For paid candidates, compute tie groups within topM.
                        # Note: ties across the full field are extremely rare with
                        # continuous scoring; we treat ties within topM exactly.
                        cand_paid = cand_row[paid_mask]
                        left = np.searchsorted(topM_sorted, cand_paid, side="left")
                        right = np.searchsorted(topM_sorted, cand_paid, side="right")

                        # Elements > cand are those strictly above the insertion
                        # point on the right in an ascending array.
                        higher = M - right  # 0-indexed rank start
                        eq_count = right - left
                        group_len = eq_count + 1
                        lo = higher
                        hi = lo + group_len

                        # Prefix-sum range query for payout sums.
                        # If a tie group (pathologically) extends beyond the
                        # contest field size, treat ranks beyond the field as
                        # paying $0 by clamping hi but still dividing by the
                        # full group length.
                        hi_clamped = np.minimum(hi, int(payouts_full.size))
                        payout_sum = payout_prefix[hi_clamped] - payout_prefix[lo]
                        payout_vals = payout_sum / group_len.astype(float)

                        if payout_sums is None:
                            raise RuntimeError("Internal error: payout_sums missing.")
                        payout_sums[paid_mask] += payout_vals

    if n_seen != num_sims:
        raise RuntimeError("Internal error: did not process expected number of simulations.")

    p_top1 = top1_hits.astype(float) / float(num_sims)
    std_scores = np.sqrt(m2_scores / float(num_sims))
    lineup_summary_df = pd.DataFrame(
        {
            "rank": lineups_df.get("rank", pd.Series(range(1, num_lineups + 1))),
            "mean_score": mean_scores,
            "std_score": std_scores,
            "top1_pct_finish_rate": 100.0 * p_top1,
        }
    )

    # Augment lineups DataFrame with top 1% probabilities.
    lineups_with_top1 = lineups_df.copy()
    lineups_with_top1["top1_pct_finish_rate"] = 100.0 * p_top1
    lineups_with_top1["mean_score"] = mean_scores
    lineups_with_top1["std_score"] = std_scores

    if contest_id is not None and payout_sums is not None and entry_fee is not None:
        avg_payout = payout_sums / float(num_sims)
        ev_roi = (avg_payout - float(entry_fee)) / float(entry_fee) if float(entry_fee) > 0 else 0.0
        lineups_with_top1["avg_payout"] = avg_payout
        lineups_with_top1["ev_roi"] = ev_roi
        lineups_with_top1["entry_fee"] = float(entry_fee)
        lineups_with_top1["contest_id"] = str(contest_id)

    # Prepare output path.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir is not None:
        output_path = run_dir / f"top1pct_lineups_{timestamp}.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        top1pct_dir = outputs_dir / "top1pct"
        top1pct_dir.mkdir(parents=True, exist_ok=True)
        output_path = top1pct_dir / f"top1pct_lineups_{timestamp}.xlsx"

    meta_rows = [
        {"key": "field_size", "value": field_size},
        {"key": "num_sims", "value": num_sims},
        {"key": "field_var_shrink", "value": field_var_shrink},
        {"key": "field_z_score", "value": field_z_score},
        {"key": "flex_var_factor", "value": flex_var_factor},
        {"key": "field_model", "value": field_model},
        {"key": "lineups_workbook", "value": str(lineups_path)},
        {"key": "correlation_workbook", "value": str(corr_path)},
        {"key": "n_players", "value": len(player_names)},
        {"key": "n_lineups", "value": len(lineups_df)},
    ]
    if contest_id is not None:
        meta_rows.append({"key": "contest_id", "value": str(contest_id)})
        if inferred_field_size is not None:
            meta_rows.append({"key": "dk_field_size", "value": int(inferred_field_size)})
        if entry_fee is not None:
            meta_rows.append({"key": "entry_fee", "value": float(entry_fee)})
        meta_rows.append({"key": "paid_places", "value": int(paid_places)})
        if field_model == "explicit":
            meta_rows.append({"key": "explicit_field_size_used", "value": int(K_field)})
    meta_df = pd.DataFrame(meta_rows)

    print(f"Writing top 1% estimates to {output_path}...")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        lineups_with_top1.to_excel(
            writer, sheet_name="Lineups_Top1Pct", index=False
        )
        meta_df.to_excel(writer, sheet_name="Meta", index=False)
        if field_lineups_with_meta is not None:
            field_lineups_with_meta.to_excel(
                writer, sheet_name="Field_Lineups", index=False
            )
        if field_meta_payload is not None:
            sheet = "Field Meta"
            # Create the sheet by writing the summary first.
            field_meta_payload["summary"].to_excel(
                writer, sheet_name=sheet, index=False, startrow=0
            )
            ws = writer.sheets[sheet]
            row = len(field_meta_payload["summary"]) + 2

            ws.write(row, 0, "CPT Ownership")
            field_meta_payload["cpt_ownership"].to_excel(
                writer, sheet_name=sheet, index=False, startrow=row + 1
            )
            row = row + 1 + len(field_meta_payload["cpt_ownership"]) + 2

            ws.write(row, 0, "FLEX Ownership")
            field_meta_payload["flex_ownership"].to_excel(
                writer, sheet_name=sheet, index=False, startrow=row + 1
            )
            row = row + 1 + len(field_meta_payload["flex_ownership"]) + 2

            ws.write(row, 0, "Stack Distribution")
            field_meta_payload["stack_distribution"].to_excel(
                writer, sheet_name=sheet, index=False, startrow=row + 1
            )

    print("Done.")
    return output_path


__all__ = [
    "run_top1pct",
]


