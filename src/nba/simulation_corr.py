from __future__ import annotations

"""
NBA Showdown correlation simulator that mirrors the NFL flow:

  - Simulate per-player box-score stats from Sabersim projections.
  - Convert those stats to DraftKings fantasy points.
  - Compute empirical correlations between players' DK scores.
"""

from typing import Optional

import numpy as np
import pandas as pd

from . import config, sabersim_parser, stat_sim


def simulate_corr_matrix_from_projections(
    sabersim_df: pd.DataFrame,
    n_sims: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate many joint game outcomes and compute a DK points correlation matrix
    for an NBA Showdown slate.
    """
    if n_sims is None:
        n_sims = config.SIM_N_GAMES
    if n_sims <= 1:
        raise ValueError("n_sims must be greater than 1 to compute correlations.")

    seed = random_seed if random_seed is not None else config.SIM_RANDOM_SEED
    rng = np.random.default_rng(seed)

    player_names, dk_points = stat_sim.simulate_nba_dk_points(
        sabersim_df, n_sims=n_sims, rng=rng
    )
    corr_df = stat_sim.build_corr_from_dk_points(player_names, dk_points)
    return corr_df


