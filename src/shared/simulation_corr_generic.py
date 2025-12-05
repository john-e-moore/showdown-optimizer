from __future__ import annotations

"""
Generic Monte Carlo simulator for Showdown player correlations.

This module implements a simple, sport-agnostic simulator that:
  1. Treats each team's total DK projection as fixed (sum of per-player
     projections).
  2. Samples per-team player shares from a Dirichlet distribution centered on
     the projected shares.
  3. Allocates the team total DK points to players according to the sampled
     shares.
  4. Repeats across many simulations and computes an empirical correlation
     matrix of DK points across players.

It is intentionally lightweight and free of NFL- or NBA-specific box-score
assumptions. Concrete sport pipelines are expected to wrap this helper with
appropriate column names and hyperparameters.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def _dirichlet_shares(
    rng: np.random.Generator,
    weights: np.ndarray,
    k: float,
    eps: float,
) -> np.ndarray:
    """
    Sample a probability vector centered on `weights` using a Dirichlet prior.

    Weights are normalized to sum to 1 over positive entries; zero-weight
    entries receive zero probability mass.
    """
    weights = np.asarray(weights, dtype=float)
    mask = weights > 0
    if not np.any(mask):
        return np.zeros_like(weights, dtype=float)

    w = weights[mask]
    total = float(w.sum())
    if total <= 0:
        return np.zeros_like(weights, dtype=float)

    w = w / total
    alpha = np.maximum(k * w, eps)
    p_sub = rng.dirichlet(alpha)

    p = np.zeros_like(weights, dtype=float)
    p[mask] = p_sub
    return p


def simulate_corr_matrix_from_projections_generic(
    sabersim_df: pd.DataFrame,
    *,
    name_col: str,
    team_col: str,
    dk_proj_col: str,
    n_sims: int,
    dirichlet_k: float = 50.0,
    eps: float = 1e-9,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate many joint game outcomes and compute a DK points correlation matrix.

    Parameters
    ----------
    sabersim_df:
        Projections dataframe for a single Showdown slate (FLEX rows only).
    name_col, team_col, dk_proj_col:
        Column names for player display name, team, and DK projection.
    n_sims:
        Number of Monte Carlo simulations (games) to run.
    dirichlet_k:
        Dirichlet concentration parameter controlling how tightly per-team DK
        point shares cluster around projected shares.
    eps:
        Small epsilon to guard against degenerate Dirichlet parameters.
    random_seed:
        Optional random seed.
    """
    required_cols = [name_col, team_col, dk_proj_col]
    missing = [c for c in required_cols if c not in sabersim_df.columns]
    if missing:
        raise KeyError(
            f"Projections dataframe missing required columns {missing}. "
            "Please check the Sabersim CSV schema."
        )

    df = sabersim_df.reset_index(drop=True).copy()
    df["player_name"] = df[name_col].astype(str)
    df["team"] = df[team_col].astype(str)
    df["dk_proj"] = df[dk_proj_col].astype(float)

    n_players = len(df)
    if n_players == 0:
        raise ValueError("Projections dataframe has no players after preprocessing.")

    # Precompute per-team structures.
    teams: Dict[str, Dict[str, np.ndarray]] = {}
    for team_name, team_df in df.groupby("team", sort=False):
        idx = team_df.index.to_numpy()
        dk_proj = team_df["dk_proj"].to_numpy(dtype=float)
        total = float(dk_proj.sum())
        if total <= 0:
            # Skip teams with non-positive total projection; they will receive
            # zero DK points in all simulations.
            continue
        w = dk_proj / total
        teams[team_name] = {
            "indices": idx,
            "dk_proj": dk_proj,
            "total_dk": np.array(total, dtype=float),
            "weights": w,
        }

    if not teams:
        raise ValueError(
            "All teams have non-positive projected DK totals; cannot simulate."
        )

    if n_sims <= 1:
        raise ValueError("n_sims must be greater than 1 to compute correlations.")

    rng = np.random.default_rng(random_seed)

    # Collect DK fantasy points for each player across simulations.
    dk_points = np.zeros((n_players, n_sims), dtype=float)
    team_names: List[str] = list(teams.keys())

    for sim_idx in range(n_sims):
        sim_dk = np.zeros(n_players, dtype=float)
        for team_name in team_names:
            info = teams[team_name]
            idx = info["indices"]
            w = info["weights"]
            total_dk = float(info["total_dk"])

            p = _dirichlet_shares(rng, w, dirichlet_k, eps)
            sim_dk[idx] = p * total_dk

        dk_points[:, sim_idx] = sim_dk

    # Drop players whose simulated DK points are identically zero across all sims.
    has_activity = (dk_points != 0.0).any(axis=1)
    active_idx = np.nonzero(has_activity)[0]
    if active_idx.size == 0:
        # Fallback: no active players; return an identity matrix over all players.
        player_names = df["player_name"].tolist()
        corr_df = pd.DataFrame(
            np.eye(len(player_names)), index=player_names, columns=player_names
        )
        return corr_df

    dk_points_active = dk_points[has_activity, :]

    # Compute empirical correlation matrix across active players only.
    corr = np.corrcoef(dk_points_active, rowvar=True)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    active_names = df.loc[has_activity, "player_name"].tolist()
    corr_df = pd.DataFrame(corr, index=active_names, columns=active_names, dtype=float)
    return corr_df



