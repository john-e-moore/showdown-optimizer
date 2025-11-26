from __future__ import annotations

"""
Monte Carlo simulator for Showdown player correlation.

Given a Sabersim projections dataframe for a single Showdown slate, this module
simulates many joint game outcomes consistent with team-level projection
totals and computes an empirical correlation matrix of DraftKings fantasy
points between players.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .build_corr_matrix_from_projections import (
    SABERSIM_DK_PROJ_COL,
    SABERSIM_NAME_COL,
    SABERSIM_POS_COL,
    SABERSIM_TEAM_COL,
)


SimTeamInfo = Dict[str, object]


def _prepare_simulation_players(
    sabersim_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, SimTeamInfo]]:
    """
    Map Sabersim projections to canonical stat inputs and per-team metadata.

    The returned dataframe includes, for each player:
      - player_name
      - team
      - position
      - dk_proj (Sabersim projected DK points)
      - canonical projected stats:
          pass_yards, pass_tds, rush_yards, rush_tds,
          rec_yards, rec_tds, receptions
      - is_qb (bool flag)

    The team_infos dict contains, for each team:
      - indices: np.ndarray of player row indices (into players_df)
      - is_qb_mask: np.ndarray[bool] of length n_team_players
      - primary_qb_local_idx: int | None
      - proj_* arrays and team totals for each stat type
    """
    required_cols = [SABERSIM_NAME_COL, SABERSIM_TEAM_COL, SABERSIM_POS_COL]
    missing = [c for c in required_cols if c not in sabersim_df.columns]
    if missing:
        raise KeyError(
            f"Sabersim dataframe missing required columns {missing}. "
            "Did load_sabersim_projections run correctly?"
        )

    if SABERSIM_DK_PROJ_COL not in sabersim_df.columns:
        raise KeyError(
            f"Sabersim dataframe missing DK projection column '{SABERSIM_DK_PROJ_COL}'."
        )

    # Work with a clean, 0-based index so that team-specific indices can be
    # used directly to index into per-simulation arrays of length n_players.
    df = sabersim_df.reset_index(drop=True).copy()
    df["player_name"] = df[SABERSIM_NAME_COL].astype(str)
    df["team"] = df[SABERSIM_TEAM_COL].astype(str)
    df["position"] = df[SABERSIM_POS_COL].astype(str)
    df["dk_proj"] = df[SABERSIM_DK_PROJ_COL].astype(float)

    # Map Sabersim stat columns to canonical names used by the scoring rules.
    stat_map = {
        "Pass Yds": config.COL_PASS_YARDS,
        "Pass TD": config.COL_PASS_TDS,
        "Rush Yds": config.COL_RUSH_YARDS,
        "Rush TD": config.COL_RUSH_TDS,
        "Rec Yds": config.COL_REC_YARDS,
        "Rec TD": config.COL_REC_TDS,
        "Rec": config.COL_RECEPTIONS,
    }
    for raw, canonical in stat_map.items():
        if raw in df.columns:
            df[canonical] = df[raw].astype(float)
        else:
            df[canonical] = 0.0

    # Simple QB heuristic: position == QB or positive passing yards projection.
    df["is_qb"] = (df["position"] == "QB") | (df[config.COL_PASS_YARDS] > 0)

    # Build per-team metadata needed for the simulator.
    team_infos: Dict[str, SimTeamInfo] = {}
    grouped = df.groupby("team", sort=False)

    for team_name, team_df in grouped:
        idx = team_df.index.to_numpy()

        proj_pass_yards = team_df[config.COL_PASS_YARDS].to_numpy(dtype=float)
        proj_pass_tds = team_df[config.COL_PASS_TDS].to_numpy(dtype=float)
        proj_rush_yards = team_df[config.COL_RUSH_YARDS].to_numpy(dtype=float)
        proj_rush_tds = team_df[config.COL_RUSH_TDS].to_numpy(dtype=float)
        proj_rec_yards = team_df[config.COL_REC_YARDS].to_numpy(dtype=float)
        proj_rec_tds = team_df[config.COL_REC_TDS].to_numpy(dtype=float)
        proj_receptions = team_df[config.COL_RECEPTIONS].to_numpy(dtype=float)
        is_qb_mask = team_df["is_qb"].to_numpy(dtype=bool)

        # For passing/receiving consistency we treat team passing yards/TDs as
        # equal to the sum of receiving yards/TDs projections.
        team_tot_rec_yards = float(proj_rec_yards.sum())
        team_tot_rec_tds = float(proj_rec_tds.sum())

        team_infos[team_name] = {
            "indices": idx,
            "is_qb_mask": is_qb_mask,
            "proj_pass_yards": proj_pass_yards,
            "proj_pass_tds": proj_pass_tds,
            "proj_rush_yards": proj_rush_yards,
            "proj_rush_tds": proj_rush_tds,
            "proj_rec_yards": proj_rec_yards,
            "proj_rec_tds": proj_rec_tds,
            "proj_receptions": proj_receptions,
            # Team-level projected totals
            "team_tot_rush_yards": float(proj_rush_yards.sum()),
            "team_tot_rush_tds": float(proj_rush_tds.sum()),
            "team_tot_rec_yards": team_tot_rec_yards,
            "team_tot_rec_tds": team_tot_rec_tds,
            "team_tot_receptions": float(proj_receptions.sum()),
        }

        # Identify a primary QB index within this team for assigning pass stats.
        qb_local_indices = np.nonzero(is_qb_mask)[0]
        primary_qb_local_idx: int | None
        if qb_local_indices.size == 0:
            primary_qb_local_idx = None
        elif qb_local_indices.size == 1:
            primary_qb_local_idx = int(qb_local_indices[0])
        else:
            # If multiple QBs, choose the one with the highest passing yards proj.
            qb_pass_yards = proj_pass_yards[qb_local_indices]
            best_rel = int(np.argmax(qb_pass_yards))
            primary_qb_local_idx = int(qb_local_indices[best_rel])

        team_infos[team_name]["primary_qb_local_idx"] = primary_qb_local_idx

    return df, team_infos


def _dirichlet_shares(
    rng: np.random.Generator, weights: np.ndarray, k: float
) -> np.ndarray:
    """
    Sample a probability vector centered on `weights` using a Dirichlet prior.

    Weights are normalized to sum to 1 over positive entries; zero-weight
    entries receive zero probability mass.
    """
    weights = np.asarray(weights, dtype=float)
    mask = weights > 0
    if not np.any(mask):
        # No positive weights: return all zeros.
        return np.zeros_like(weights, dtype=float)

    w = weights[mask]
    total = float(w.sum())
    if total <= 0:
        return np.zeros_like(weights, dtype=float)

    w = w / total
    alpha = np.maximum(k * w, config.SIM_EPS)
    p_sub = rng.dirichlet(alpha)

    p = np.zeros_like(weights, dtype=float)
    p[mask] = p_sub
    return p


def _simulate_team_offense(
    rng: np.random.Generator, team_info: SimTeamInfo
) -> Dict[str, np.ndarray]:
    """
    Simulate one game's offensive box-score stats for a single team.

    Returns a dict of arrays (all length n_team_players) for:
      - pass_yards, pass_tds
      - rush_yards, rush_tds
      - rec_yards, rec_tds, receptions
    """
    proj_rush_yards = team_info["proj_rush_yards"]
    proj_rush_tds = team_info["proj_rush_tds"]
    proj_rec_yards = team_info["proj_rec_yards"]
    proj_rec_tds = team_info["proj_rec_tds"]
    proj_receptions = team_info["proj_receptions"]
    is_qb_mask = team_info["is_qb_mask"]
    primary_qb_local_idx = team_info["primary_qb_local_idx"]

    n_players = proj_rush_yards.shape[0]

    team_tot_rush_yards = float(team_info["team_tot_rush_yards"])
    team_tot_rush_tds = float(team_info["team_tot_rush_tds"])
    team_tot_rec_yards = float(team_info["team_tot_rec_yards"])
    team_tot_rec_tds = float(team_info["team_tot_rec_tds"])
    team_tot_receptions = float(team_info["team_tot_receptions"])

    # Receiving yards
    if team_tot_rec_yards > 0:
        p_rec_yards = _dirichlet_shares(
            rng, proj_rec_yards, config.SIM_DIRICHLET_K_YARDS
        )
        rec_yards = p_rec_yards * team_tot_rec_yards
    else:
        rec_yards = np.zeros(n_players, dtype=float)

    # Receptions
    if team_tot_receptions > 0:
        p_receptions = _dirichlet_shares(
            rng, proj_receptions, config.SIM_DIRICHLET_K_RECEPTIONS
        )
        receptions = p_receptions * team_tot_receptions
    else:
        receptions = np.zeros(n_players, dtype=float)

    # Receiving TDs: Poisson team total, then multinomial allocation.
    if team_tot_rec_tds > 0:
        lam_rec_tds = max(team_tot_rec_tds, 0.0)
        team_rec_tds = int(rng.poisson(lam_rec_tds))
        if team_rec_tds > 0:
            p_rec_tds = _dirichlet_shares(
                rng, proj_rec_tds, config.SIM_DIRICHLET_K_TDS
            )
            if p_rec_tds.sum() > 0:
                rec_tds = rng.multinomial(team_rec_tds, p_rec_tds).astype(float)
            else:
                rec_tds = np.zeros(n_players, dtype=float)
        else:
            rec_tds = np.zeros(n_players, dtype=float)
    else:
        rec_tds = np.zeros(n_players, dtype=float)

    # Rushing yards
    if team_tot_rush_yards > 0:
        p_rush_yards = _dirichlet_shares(
            rng, proj_rush_yards, config.SIM_DIRICHLET_K_YARDS
        )
        rush_yards = p_rush_yards * team_tot_rush_yards
    else:
        rush_yards = np.zeros(n_players, dtype=float)

    # Rushing TDs
    if team_tot_rush_tds > 0:
        lam_rush_tds = max(team_tot_rush_tds, 0.0)
        team_rush_tds = int(rng.poisson(lam_rush_tds))
        if team_rush_tds > 0:
            p_rush_tds = _dirichlet_shares(
                rng, proj_rush_tds, config.SIM_DIRICHLET_K_TDS
            )
            if p_rush_tds.sum() > 0:
                rush_tds = rng.multinomial(team_rush_tds, p_rush_tds).astype(float)
            else:
                rush_tds = np.zeros(n_players, dtype=float)
        else:
            rush_tds = np.zeros(n_players, dtype=float)
    else:
        rush_tds = np.zeros(n_players, dtype=float)

    # Passing stats are tied to receiving stats:
    #   - Team passing yards == sum(rec_yards)
    #   - Team passing TDs == team receiving TDs
    pass_yards = np.zeros(n_players, dtype=float)
    pass_tds = np.zeros(n_players, dtype=float)

    team_pass_yards = float(rec_yards.sum())
    team_pass_tds = float(rec_tds.sum())

    if primary_qb_local_idx is not None:
        pass_yards[primary_qb_local_idx] = team_pass_yards
        pass_tds[primary_qb_local_idx] = team_pass_tds

    return {
        config.COL_PASS_YARDS: pass_yards,
        config.COL_PASS_TDS: pass_tds,
        config.COL_RUSH_YARDS: rush_yards,
        config.COL_RUSH_TDS: rush_tds,
        config.COL_REC_YARDS: rec_yards,
        config.COL_REC_TDS: rec_tds,
        config.COL_RECEPTIONS: receptions,
    }


def _compute_dk_points_from_stats(
    pass_yards: np.ndarray,
    pass_tds: np.ndarray,
    interceptions: np.ndarray,
    rush_yards: np.ndarray,
    rush_tds: np.ndarray,
    rec_yards: np.ndarray,
    rec_tds: np.ndarray,
    receptions: np.ndarray,
) -> np.ndarray:
    """
    Vectorized DK offensive scoring, mirroring fantasy_scoring.compute_dk_points_offense
    but without diagnostics or dataframe overhead.
    """
    dk_pass = (
        0.04 * pass_yards
        + 4.0 * pass_tds
        - 1.0 * interceptions
        + 3.0 * (pass_yards >= 300).astype(float)
    )

    dk_rush = (
        0.1 * rush_yards
        + 6.0 * rush_tds
        + 3.0 * (rush_yards >= 100).astype(float)
    )

    dk_rec = (
        receptions
        + 0.1 * rec_yards
        + 6.0 * rec_tds
        + 3.0 * (rec_yards >= 100).astype(float)
    )

    return dk_pass + dk_rush + dk_rec


def simulate_corr_matrix_from_projections(
    sabersim_df: pd.DataFrame,
    n_sims: int | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate many joint game outcomes and compute a DK points correlation matrix.

    Parameters
    ----------
    sabersim_df:
        Sabersim projections dataframe for a single Showdown slate, after
        load_sabersim_projections (FLEX rows only).
    n_sims:
        Number of Monte Carlo simulations (games) to run. If None, defaults to
        config.SIM_N_GAMES.
    random_seed:
        Optional random seed. If None, uses config.SIM_RANDOM_SEED.
    """
    players_df, team_infos = _prepare_simulation_players(sabersim_df)

    n_players = len(players_df)
    if n_players == 0:
        raise ValueError("Sabersim dataframe has no players after preprocessing.")

    if n_sims is None:
        n_sims = config.SIM_N_GAMES
    if n_sims <= 1:
        raise ValueError("n_sims must be greater than 1 to compute correlations.")

    seed = random_seed if random_seed is not None else config.SIM_RANDOM_SEED
    rng = np.random.default_rng(seed)

    # Collect DK fantasy points for each player across simulations.
    dk_points = np.zeros((n_players, n_sims), dtype=float)

    teams: List[str] = list(team_infos.keys())

    for sim_idx in range(n_sims):
        sim_dk = np.zeros(n_players, dtype=float)

        for team_name in teams:
            info = team_infos[team_name]
            team_stats = _simulate_team_offense(rng, info)

            idx = info["indices"]
            pass_yards = team_stats[config.COL_PASS_YARDS]
            pass_tds = team_stats[config.COL_PASS_TDS]
            rush_yards = team_stats[config.COL_RUSH_YARDS]
            rush_tds = team_stats[config.COL_RUSH_TDS]
            rec_yards = team_stats[config.COL_REC_YARDS]
            rec_tds = team_stats[config.COL_REC_TDS]
            receptions = team_stats[config.COL_RECEPTIONS]

            interceptions = np.zeros_like(pass_yards, dtype=float)

            dk = _compute_dk_points_from_stats(
                pass_yards=pass_yards,
                pass_tds=pass_tds,
                interceptions=interceptions,
                rush_yards=rush_yards,
                rush_tds=rush_tds,
                rec_yards=rec_yards,
                rec_tds=rec_tds,
                receptions=receptions,
            )

            sim_dk[idx] = dk

        dk_points[:, sim_idx] = sim_dk

    # Compute empirical correlation matrix across players.
    corr = np.corrcoef(dk_points, rowvar=True)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure perfect self-correlation on the diagonal.
    np.fill_diagonal(corr, 1.0)

    player_names = players_df["player_name"].tolist()
    corr_df = pd.DataFrame(corr, index=player_names, columns=player_names, dtype=float)
    return corr_df


def summarize_simulation_vs_projections(
    sabersim_df: pd.DataFrame,
    n_sims: int | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Run a short simulation and compare simulated DK means to Sabersim projections.

    This is a lightweight diagnostic helper; it returns a dataframe with:
      - player_name
      - dk_proj (Sabersim projection)
      - dk_sim_mean (mean DK from simulations)
      - dk_sim_std (std DK from simulations)
    """
    players_df, _ = _prepare_simulation_players(sabersim_df)

    if n_sims is None:
        n_sims = max(500, min(config.SIM_N_GAMES, 2000))

    seed = random_seed if random_seed is not None else config.SIM_RANDOM_SEED
    rng = np.random.default_rng(seed)

    # Reuse main simulator logic but keep per-simulation DK points.
    _, team_infos = _prepare_simulation_players(sabersim_df)
    n_players = len(players_df)
    dk_points = np.zeros((n_players, n_sims), dtype=float)
    teams: List[str] = list(team_infos.keys())

    for sim_idx in range(n_sims):
        sim_dk = np.zeros(n_players, dtype=float)
        for team_name in teams:
            info = team_infos[team_name]
            team_stats = _simulate_team_offense(rng, info)

            idx = info["indices"]
            pass_yards = team_stats[config.COL_PASS_YARDS]
            pass_tds = team_stats[config.COL_PASS_TDS]
            rush_yards = team_stats[config.COL_RUSH_YARDS]
            rush_tds = team_stats[config.COL_RUSH_TDS]
            rec_yards = team_stats[config.COL_REC_YARDS]
            rec_tds = team_stats[config.COL_REC_TDS]
            receptions = team_stats[config.COL_RECEPTIONS]

            interceptions = np.zeros_like(pass_yards, dtype=float)

            dk = _compute_dk_points_from_stats(
                pass_yards=pass_yards,
                pass_tds=pass_tds,
                interceptions=interceptions,
                rush_yards=rush_yards,
                rush_tds=rush_tds,
                rec_yards=rec_yards,
                rec_tds=rec_tds,
                receptions=receptions,
            )
            sim_dk[idx] = dk

        dk_points[:, sim_idx] = sim_dk

    dk_mean = dk_points.mean(axis=1)
    dk_std = dk_points.std(axis=1, ddof=0)

    summary = pd.DataFrame(
        {
            "player_name": players_df["player_name"].tolist(),
            "dk_proj": players_df["dk_proj"].to_numpy(dtype=float),
            "dk_sim_mean": dk_mean,
            "dk_sim_std": dk_std,
        }
    )
    return summary




