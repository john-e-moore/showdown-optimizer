from __future__ import annotations

"""
NFL-specific namespace.

This package currently provides thin wrappers around the existing NFL
correlation pipeline modules in the top-level `src` package. It exists so that
NBA-specific code can live under `src/nba` while NFL logic is grouped under
`src/nfl` without breaking existing entrypoints like `python -m src.main`.
"""

from .. import config as config  # Re-export NFL config
from .. import build_corr_matrix_from_projections  # Sabersim helpers (NFL)
from .. import simulation_corr  # NFL Monte Carlo correlation engine
from .. import main as corr_main  # CLI entrypoint for NFL correlation


__all__ = [
    "config",
    "build_corr_matrix_from_projections",
    "simulation_corr",
    "corr_main",
]


