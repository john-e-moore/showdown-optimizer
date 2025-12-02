from __future__ import annotations

"""
NBA-specific configuration for the Showdown pipelines.

This module mirrors `src.config` (NFL) but routes all data/outputs/diagnostics
through sport-aware subdirectories:

  - data/nba/...
  - outputs/nba/...
"""

from pathlib import Path
from typing import Final, List

from ..shared import config_base


# -----------------------------------------------------------------------------
# Base paths
# -----------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = config_base.PROJECT_ROOT

DATA_DIR: Final[Path] = config_base.get_data_dir_for_sport("nba")
SABERSIM_DIR: Final[Path] = DATA_DIR / "sabersim"
MODELS_DIR: Final[Path] = config_base.MODELS_ROOT

OUTPUTS_DIR: Final[Path] = config_base.get_outputs_dir_for_sport("nba")
CORR_OUTPUTS_DIR: Final[Path] = OUTPUTS_DIR / "correlations"
DIAGNOSTICS_DIR: Final[Path] = config_base.get_diagnostics_dir_for_sport("nba")


# -----------------------------------------------------------------------------
# Data file locations (defaults)
# -----------------------------------------------------------------------------

SABERSIM_CSV: Final[str] = str(
    SABERSIM_DIR / "NBA_showdown_example.csv"
)

OUTPUT_CORR_EXCEL: Final[str] = str(
    CORR_OUTPUTS_DIR / "showdown_corr_matrix.xlsx"
)


# -----------------------------------------------------------------------------
# Modeling / filtering constants (NBA)
# -----------------------------------------------------------------------------

# Enable or disable writing diagnostics snapshots for NBA-specific code that
# chooses to use this flag.
ENABLE_DIAGNOSTICS: Final[bool] = True

# NBA positions as reported by Sabersim (this is only used for optional
# filtering; the optimizer is already robust to arbitrary positions).
OFFENSIVE_POSITIONS: Final[List[str]] = ["PG", "SG", "SF", "PF", "C", "G", "F"]


# -----------------------------------------------------------------------------
# Simulation configuration for correlation-from-projections
# -----------------------------------------------------------------------------

# Number of Monte Carlo simulations (games) to run per Showdown slate.
SIM_N_GAMES: Final[int] = 5000

# Random seed for the simulator (set to None to use nondeterministic seed).
SIM_RANDOM_SEED: Final[int | None] = 42

# Dirichlet concentration parameter controlling how tightly per-team DK point
# shares cluster around their projected shares. Higher values => lower
# variance around projections.
SIM_DIRICHLET_K_DK_POINTS: Final[float] = 50.0

# Small epsilon used when guarding against division by zero in share
# computations.
SIM_EPS: Final[float] = 1e-9


def ensure_directories() -> None:
    """
    Ensure that key directories used by the NBA pipelines exist.
    """

    CORR_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)



