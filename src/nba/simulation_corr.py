from __future__ import annotations

"""
NBA Showdown correlation simulator built on the shared generic engine.
"""

from typing import Optional

import pandas as pd

from . import config, sabersim_parser
from ..shared.simulation_corr_generic import (
    simulate_corr_matrix_from_projections_generic,
)


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

    seed = random_seed if random_seed is not None else config.SIM_RANDOM_SEED

    corr_df = simulate_corr_matrix_from_projections_generic(
        sabersim_df,
        name_col=sabersim_parser.SABERSIM_NAME_COL,
        team_col=sabersim_parser.SABERSIM_TEAM_COL,
        dk_proj_col=sabersim_parser.SABERSIM_DK_PROJ_COL,
        n_sims=n_sims,
        dirichlet_k=config.SIM_DIRICHLET_K_DK_POINTS,
        eps=config.SIM_EPS,
        random_seed=seed,
    )
    return corr_df



