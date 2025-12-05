from __future__ import annotations

"""
Sport-agnostic configuration primitives and path helpers.

Concrete sport configs (e.g. `src.nfl.config`, `src.nba.config`) should import
from this module and then define sport-specific constants such as default
Sabersim paths, position sets, and modeling knobs.
"""

from pathlib import Path
from typing import Final


# -----------------------------------------------------------------------------
# Project roots
# -----------------------------------------------------------------------------

# `src/shared/config_base.py` → src/shared → src → project root
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
OUTPUTS_ROOT: Final[Path] = PROJECT_ROOT / "outputs"
DIAGNOSTICS_ROOT: Final[Path] = PROJECT_ROOT / "diagnostics"
MODELS_ROOT: Final[Path] = PROJECT_ROOT / "models"


def get_data_dir_for_sport(sport: str) -> Path:
    """
    Root data directory for a given sport, e.g. data/nfl, data/nba.
    """

    return DATA_ROOT / sport


def get_outputs_dir_for_sport(sport: str) -> Path:
    """
    Root outputs directory for a given sport, e.g. outputs/nfl, outputs/nba.
    """

    return OUTPUTS_ROOT / sport


def get_diagnostics_dir_for_sport(sport: str) -> Path:
    """
    Root diagnostics directory for a given sport, e.g. diagnostics/nfl.
    """

    return DIAGNOSTICS_ROOT / sport



