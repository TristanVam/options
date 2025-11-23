"""Monte Carlo pricing for barrier options."""
from __future__ import annotations

from math import exp
from typing import Optional

import numpy as np

from .mc_paths import generate_gbm_paths


def price_barrier_up_and_out_call_mc(
    S0: float,
    K: float,
    H: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> float:
    """Price an up-and-out call via Monte Carlo simulation.

    The option is knocked out if the underlying crosses the barrier ``H`` at any
    monitoring date. Otherwise it pays the discounted European call payoff.
    """
    paths = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    barrier_breached = (paths[:, 1:] >= H).any(axis=1)
    terminal_prices = paths[:, -1]
    payoffs = np.where(barrier_breached, 0.0, np.maximum(terminal_prices - K, 0.0))
    return exp(-r * T) * payoffs.mean()
