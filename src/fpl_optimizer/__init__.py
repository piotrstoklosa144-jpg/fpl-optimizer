"""FPL Optimizer: Poisson expected points, MILP transfer optimization and Monte Carlo risk analysis."""

from .data import FPLData, load_fpl_data
from .model import add_expected_points
from .optimizer import lineup_points, optimize_transfers, pick_lineup
from .simulation import lineup_distribution, simulate_points, summarize

__version__ = "0.1.0"

__all__ = [
    "FPLData",
    "add_expected_points",
    "lineup_distribution",
    "lineup_points",
    "load_fpl_data",
    "optimize_transfers",
    "pick_lineup",
    "simulate_points",
    "summarize",
]
