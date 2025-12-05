from __future__ import annotations

"""
NBA box-score simulator and DK fantasy scoring for Showdown correlations.

This module mirrors the NFL flow in `src/simulation_corr.py` but for NBA:

  1. Start from Sabersim projections that include per-player expected stats
     such as PTS, RB, AST, STL, BLK, TO, and 3PT.
  2. Sample per-player stat lines across many simulations.
  3. Convert those stat lines to DraftKings fantasy points using the standard
     NBA scoring rules (including 3PM, double-double, and triple-double
     bonuses).
  4. Provide helpers to build a correlation matrix over simulated DK scores.
"""

from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from . import config
from . import sabersim_parser


# Columns in the Sabersim CSV used for NBA stats
COL_PTS = "PTS"
COL_REB = "RB"
COL_AST = "AST"
COL_STL = "STL"
COL_BLK = "BLK"
COL_TOV = "TO"
COL_FG3M = "3PT"


def _get_projection_values(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Safely extract a non-negative float projection column; fall back to zeros
    when the column is missing.
    """
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(
            dtype=float
        )
    else:
        vals = np.zeros(len(df), dtype=float)
    # Guard against tiny negative projections from numeric noise.
    return np.maximum(vals, 0.0)


def _build_game_and_team_indices(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build integer game and team indices for each player row.

    Games are identified by unordered {Team, Opp} pairs so that both sides of
    a matchup share the same game id. Teams are identified by the `Team`
    column alone.
    """
    if sabersim_parser.SABERSIM_TEAM_COL not in df.columns:
        raise KeyError(
            f"Sabersim dataframe missing Team column "
            f"'{sabersim_parser.SABERSIM_TEAM_COL}'."
        )

    teams = df[sabersim_parser.SABERSIM_TEAM_COL].astype(str).to_numpy()
    if "Opp" in df.columns:
        opps = df["Opp"].astype(str).to_numpy()
    else:
        # Fallback: treat each team as its own \"game\" when Opp is unavailable.
        opps = teams.copy()

    n_players = len(df)
    game_ids = np.empty(n_players, dtype=int)
    team_ids = np.empty(n_players, dtype=int)

    game_key_to_id: Dict[Tuple[str, str], int] = {}
    team_to_id: Dict[str, int] = {}

    for i in range(n_players):
        team = teams[i]
        opp = opps[i]
        # Unordered key so (OKC, GSW) and (GSW, OKC) map to the same game.
        key = tuple(sorted({team, opp}))
        if key not in game_key_to_id:
            game_key_to_id[key] = len(game_key_to_id)
        game_ids[i] = game_key_to_id[key]

        if team not in team_to_id:
            team_to_id[team] = len(team_to_id)
        team_ids[i] = team_to_id[team]

    n_games = len(game_key_to_id)
    n_teams = len(team_to_id)
    return game_ids, team_ids, n_games, n_teams


def simulate_nba_stats_from_projections(
    sabersim_df: pd.DataFrame,
    n_sims: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Sample per-player NBA box-score stats from Sabersim projections.

    For a first pass, each stat is modeled as an independent Poisson random
    variable with mean equal to the Sabersim-projected value.

    Returns:
        player_names: list of player display names in model order.
        stats: mapping stat_name -> array of shape (n_players, n_sims).
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")

    df = sabersim_df.reset_index(drop=True).copy()
    if sabersim_parser.SABERSIM_NAME_COL not in df.columns:
        raise KeyError(
            f"Sabersim dataframe missing Name column "
            f"'{sabersim_parser.SABERSIM_NAME_COL}'."
        )

    player_names = df[sabersim_parser.SABERSIM_NAME_COL].astype(str).tolist()
    n_players = len(df)
    if n_players == 0:
        raise ValueError("Sabersim dataframe has no players after preprocessing.")

    if rng is None:
        seed = config.SIM_RANDOM_SEED
        rng = np.random.default_rng(seed)

    # Expected values for each counting stat.
    mu_pts = _get_projection_values(df, COL_PTS)
    mu_reb = _get_projection_values(df, COL_REB)
    mu_ast = _get_projection_values(df, COL_AST)
    mu_stl = _get_projection_values(df, COL_STL)
    mu_blk = _get_projection_values(df, COL_BLK)
    mu_tov = _get_projection_values(df, COL_TOV)
    mu_fg3 = _get_projection_values(df, COL_FG3M)

    # Optionally apply game/team coupling if configured; otherwise fall back
    # to independent per-player Poisson draws.
    game_var = getattr(config, "NBA_SIM_GAME_VAR", 0.0)
    team_var = getattr(config, "NBA_SIM_TEAM_VAR", 0.0)

    def _independent_poisson(mu: np.ndarray) -> np.ndarray:
        sims = rng.poisson(lam=mu[None, :], size=(n_sims, n_players))
        return sims.T.astype(float)

    if game_var <= 0.0 and team_var <= 0.0:
        stats: Dict[str, np.ndarray] = {
            "pts": _independent_poisson(mu_pts),
            "reb": _independent_poisson(mu_reb),
            "ast": _independent_poisson(mu_ast),
            "stl": _independent_poisson(mu_stl),
            "blk": _independent_poisson(mu_blk),
            "tov": _independent_poisson(mu_tov),
            "fg3m": _independent_poisson(mu_fg3),
        }
        return player_names, stats

    # Build game/team indices so we can apply shared multipliers.
    game_ids, team_ids, n_games, n_teams = _build_game_and_team_indices(df)

    # Lognormal game-level factors with mean 1.0.
    sigma_game = float(game_var) ** 0.5
    if sigma_game > 0.0:
        mu_log_game = -0.5 * sigma_game * sigma_game
        z_game = rng.normal(
            loc=mu_log_game, scale=sigma_game, size=(n_games, n_sims)
        )
        game_factors = np.exp(z_game)
    else:
        game_factors = np.ones((n_games, n_sims), dtype=float)

    # Lognormal team-level factors with mean 1.0.
    sigma_team = float(team_var) ** 0.5
    if sigma_team > 0.0:
        mu_log_team = -0.5 * sigma_team * sigma_team
        z_team = rng.normal(
            loc=mu_log_team, scale=sigma_team, size=(n_teams, n_sims)
        )
        team_factors = np.exp(z_team)
    else:
        team_factors = np.ones((n_teams, n_sims), dtype=float)

    # Combined multiplier per player and simulation.
    # game_factors[game_ids] -> (n_players, n_sims)
    # team_factors[team_ids] -> (n_players, n_sims)
    m_player = game_factors[game_ids] * team_factors[team_ids]

    def _coupled_poisson(mu: np.ndarray) -> np.ndarray:
        # Scale each player's mean by the game/team multiplier.
        lam = mu[:, None] * m_player
        sims = rng.poisson(lam=lam)
        return sims.astype(float)

    stats: Dict[str, np.ndarray] = {
        "pts": _coupled_poisson(mu_pts),
        "reb": _coupled_poisson(mu_reb),
        "ast": _coupled_poisson(mu_ast),
        "stl": _coupled_poisson(mu_stl),
        "blk": _coupled_poisson(mu_blk),
        "tov": _coupled_poisson(mu_tov),
        "fg3m": _coupled_poisson(mu_fg3),
    }

    return player_names, stats


def compute_dk_points_nba(
    pts: np.ndarray,
    reb: np.ndarray,
    ast: np.ndarray,
    stl: np.ndarray,
    blk: np.ndarray,
    tov: np.ndarray,
    fg3m: np.ndarray,
) -> np.ndarray:
    """
    Vectorized DraftKings NBA scoring with 3PM, double-double, and triple-double
    bonuses applied.

    DK NBA (Showdown uses the same underlying scoring):
      - 1 point per real point
      - 1.25 per rebound
      - 1.5 per assist
      - 2 per steal
      - 2 per block
      - -0.5 per turnover
      - 0.5 per made three-pointer
      - +1.5 for a double-double (>=10 in exactly two of {PTS, REB, AST, STL, BLK})
      - +3.0 for a triple-double (>=10 in three or more of {PTS, REB, AST, STL, BLK})
    """
    base = (
        1.0 * pts
        + 1.25 * reb
        + 1.5 * ast
        + 2.0 * stl
        + 2.0 * blk
        - 0.5 * tov
        + 0.5 * fg3m
    )

    # Double-double / triple-double bonuses.
    cat_ge_10 = np.stack(
        [
            pts >= 10.0,
            reb >= 10.0,
            ast >= 10.0,
            stl >= 10.0,
            blk >= 10.0,
        ],
        axis=0,
    )  # shape (5, n_players, n_sims)
    cat_count = cat_ge_10.sum(axis=0)  # shape (n_players, n_sims)

    dd_mask = (cat_count >= 2) & (cat_count < 3)
    td_mask = cat_count >= 3

    bonuses = 1.5 * dd_mask.astype(float) + 3.0 * td_mask.astype(float)
    return base + bonuses


def simulate_nba_dk_points(
    sabersim_df: pd.DataFrame,
    n_sims: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Convenience wrapper: simulate per-player NBA stats and convert them to DK
    fantasy points.

    Returns:
        player_names: list of player display names in model order.
        dk_points: array of shape (n_players, n_sims) with DK fantasy scores.
    """
    player_names, stats = simulate_nba_stats_from_projections(
        sabersim_df, n_sims=n_sims, rng=rng
    )

    dk_points = compute_dk_points_nba(
        pts=stats["pts"],
        reb=stats["reb"],
        ast=stats["ast"],
        stl=stats["stl"],
        blk=stats["blk"],
        tov=stats["tov"],
        fg3m=stats["fg3m"],
    )
    return player_names, dk_points


def build_corr_from_dk_points(
    player_names: List[str],
    dk_points: np.ndarray,
) -> pd.DataFrame:
    """
    Build an empirical correlation matrix over simulated DK scores.

    Players with zero variance across simulations are dropped from the
    correlation matrix (they would otherwise cause degenerate correlations).
    """
    if dk_points.ndim != 2:
        raise ValueError("dk_points must have shape (n_players, n_sims).")

    n_players, n_sims = dk_points.shape
    if n_players != len(player_names):
        raise ValueError(
            "player_names length must match dk_points.shape[0] (n_players)."
        )
    if n_sims <= 1:
        raise ValueError("n_sims must be greater than 1 to compute correlations.")

    # Drop players with identically zero scores.
    has_activity = (dk_points != 0.0).any(axis=1)
    if not np.any(has_activity):
        # Fallback: identity matrix over all players.
        corr_df = pd.DataFrame(
            np.eye(len(player_names)), index=player_names, columns=player_names
        )
        return corr_df

    dk_active = dk_points[has_activity, :]
    names_active = [name for name, keep in zip(player_names, has_activity) if keep]

    corr = np.corrcoef(dk_active, rowvar=True)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    corr_df = pd.DataFrame(corr, index=names_active, columns=names_active, dtype=float)
    return corr_df



