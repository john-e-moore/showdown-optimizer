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
        sabersim_proj_df: FLEX-only Sabersim projections.
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


def _build_player_universe(
    lineups_df: pd.DataFrame,
    ownership_df: pd.DataFrame,
    sabersim_proj_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    lineups_proj_df: pd.DataFrame,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build unified player universe and aligned arrays.

    Returns:
        player_names: list of player names in model order.
        mu: DK point means (shape P).
        sigma: DK point standard deviations (shape P).
        cpt_own: CPT ownership fractions (shape P).
        flex_own: FLEX ownership fractions (shape P).
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
    lineup_cols = ["cpt"] + [f"flex{j}" for j in range(1, 6)]
    missing_cols = [c for c in lineup_cols if c not in lineups_df.columns]
    if missing_cols:
        raise KeyError(
            f"Lineups sheet missing expected columns {missing_cols}."
        )

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
    # Keep lowest-salary row per (Name, Team) as FLEX-equivalent.
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

    mu = np.array(mu_list, dtype=float)
    sigma = np.array(sigma_list, dtype=float)
    positions = np.array(pos_list, dtype=object)

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

    return universe_names, mu, sigma, cpt_own, flex_own, positions


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

    Args:
        X: player DK scores, shape (S, P).
        cpt_own: CPT ownership fractions, shape (P,).
        flex_own: FLEX ownership fractions, shape (P,).

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

    # FLEX mixture: per-slot distribution, nominally 5 slots but with an
    # optional variance calibration factor to soften the tail.
    pi = flex_own / 5.0
    flex_scores = X  # DK points at FLEX are just X.
    mu_F = flex_scores @ pi
    mF2 = (flex_scores ** 2) @ pi
    var_F = mF2 - mu_F**2
    var_F = np.clip(var_F, a_min=0.0, a_max=None)

    mu_F_total = 5.0 * mu_F
    # Calibrated FLEX variance factor k (<= 5.0 ideally).
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
      - Each FLEX player has weight 1.0 (weights add if duplicated).
    """
    lineup_cols = ["cpt"] + [f"flex{j}" for j in range(1, 6)]
    missing_cols = [c for c in lineup_cols if c not in lineups_df.columns]
    if missing_cols:
        raise KeyError(
            f"Lineups sheet missing expected columns {missing_cols}."
        )

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

        # FLEX slots
        for col in [f"flex{j}" for j in range(1, 6)]:
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


def run_top1pct(
    field_size: int,
    outputs_dir: Path,
    *,
    lineups_excel: str | None = None,
    corr_excel: str | None = None,
    num_sims: int = 20_000,
    random_seed: int | None = None,
    field_var_shrink: float = 0.7,
    field_z_score: float = 2.0,
    flex_var_factor: float = 3.5,
) -> Path:
    """
    Execute the top 1% finish rate estimation pipeline for a given sport.

    Args:
        field_size: Total number of lineups in the contest field.
        outputs_dir: Root outputs directory for the sport
                     (e.g. outputs/nfl or outputs/nba).
        lineups_excel: Optional explicit path to a lineups workbook.
        corr_excel: Optional explicit path to a correlation workbook.
        num_sims: Number of Monte Carlo simulations.
        random_seed: Optional RNG seed; if None, use nondeterministic seed.
        field_var_shrink: Variance shrinkage factor for the modeled field.
        field_z_score: Z-score for the upper tail of the field distribution.
        flex_var_factor: Effective variance factor for FLEX component.

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
    ) = _build_player_universe(
        lineups_df=lineups_df,
        ownership_df=ownership_df,
        sabersim_proj_df=sabersim_proj_df,
        corr_df=corr_df,
        lineups_proj_df=lineups_proj_df,
    )

    corr_full = _build_full_correlation(player_names, corr_df)
    rng = np.random.default_rng(random_seed)

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

    print("Computing field 99th percentile thresholds from ownership...")
    thresholds = _compute_field_thresholds(
        X=X,
        cpt_own=cpt_own,
        flex_own=flex_own,
        field_var_shrink=field_var_shrink,
        field_z_score=field_z_score,
        flex_var_factor=flex_var_factor,
    )

    player_index = {name: i for i, name in enumerate(player_names)}
    print("Scoring lineups across simulations...")
    W = _build_lineup_weights(lineups_df=lineups_df, player_index=player_index)

    # scores[s, k] = sum_i w_k[i] * X[s, i]
    scores = X @ W.T  # shape (S, K)
    if thresholds.shape[0] != scores.shape[0]:
        raise ValueError("Thresholds array must have length equal to num_sims.")

    indicators = scores >= thresholds[:, None]
    p_top1 = indicators.mean(axis=0)

    # Per-lineup mean/std plus top 1% rate.
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0, ddof=0)
    lineup_summary_df = pd.DataFrame(
        {
            "rank": lineups_df.get("rank", pd.Series(range(1, scores.shape[1] + 1))),
            "mean_score": mean_scores,
            "std_score": std_scores,
            "top1_pct_finish_rate": 100.0 * p_top1,
        }
    )

    # Augment lineups DataFrame with top 1% probabilities.
    lineups_with_top1 = lineups_df.copy()
    lineups_with_top1["top1_pct_finish_rate"] = 100.0 * p_top1

    # Prepare output path.
    top1pct_dir = outputs_dir / "top1pct"
    top1pct_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = top1pct_dir / f"top1pct_lineups_{timestamp}.xlsx"

    meta_rows = [
        {"key": "field_size", "value": field_size},
        {"key": "num_sims", "value": num_sims},
        {"key": "field_var_shrink", "value": field_var_shrink},
        {"key": "field_z_score", "value": field_z_score},
        {"key": "flex_var_factor", "value": flex_var_factor},
        {"key": "lineups_workbook", "value": str(lineups_path)},
        {"key": "correlation_workbook", "value": str(corr_path)},
        {"key": "n_players", "value": len(player_names)},
        {"key": "n_lineups", "value": len(lineups_df)},
    ]
    meta_df = pd.DataFrame(meta_rows)

    print(f"Writing top 1% estimates to {output_path}...")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        lineups_with_top1.to_excel(
            writer, sheet_name="Lineups_Top1Pct", index=False
        )
        meta_df.to_excel(writer, sheet_name="Meta", index=False)

    print("Done.")
    return output_path


__all__ = [
    "run_top1pct",
]


