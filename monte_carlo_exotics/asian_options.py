"""Monte Carlo pricing routines for Asian options."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Literal, Optional

import numpy as np

from .mc_paths import generate_gbm_paths

OptionType = Literal["call", "put"]


@dataclass
class AsianResult:
    """Container for Asian option pricing results."""

    price: float
    payoffs: np.ndarray


def _asian_payoff(averages: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(averages - K, 0.0)
    if option_type == "put":
        return np.maximum(K - averages, 0.0)
    raise ValueError("option_type must be either 'call' or 'put'.")


def price_asian_arithmetic_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: OptionType = "call",
    antithetic: bool = False,
    seed: Optional[int] = None,
) -> AsianResult:
    """Price an arithmetic-average Asian option via Monte Carlo.

    The arithmetic average is taken over the simulated prices excluding the
    initial spot ``S0`` (i.e., average over ``n_steps`` monitoring dates).

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    K : float
        Strike price of the option.
    r : float
        Risk-free annual interest rate (continuous compounding).
    sigma : float
        Annualized volatility of the underlying asset.
    T : float
        Maturity in years.
    n_steps : int
        Number of monitoring steps for the average.
    n_paths : int
        Number of Monte Carlo paths.
    option_type : {"call", "put"}, optional
        Option payoff type.
    antithetic : bool, optional
        Whether to employ antithetic variates for variance reduction.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    AsianResult
        Estimated price and raw payoff samples.
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        pair_count = n_paths // 2
        remainder = n_paths % 2
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = S0

        for t in range(1, n_steps + 1):
            z = rng.standard_normal(pair_count + remainder)
            if pair_count:
                pos = np.exp(drift + diffusion * z[:pair_count])
                neg = np.exp(drift - diffusion * z[:pair_count])
                paths[:pair_count, t] = paths[:pair_count, t - 1] * pos
                paths[pair_count : 2 * pair_count, t] = (
                    paths[pair_count : 2 * pair_count, t - 1] * neg
                )
            if remainder:
                idx = 2 * pair_count
                step_factor = np.exp(drift + diffusion * z[-1])
                paths[idx, t] = paths[idx, t - 1] * step_factor
    else:
        paths = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)

    arithmetic_means = paths[:, 1:].mean(axis=1)
    payoffs = _asian_payoff(arithmetic_means, K, option_type)
    discount_factor = exp(-r * T)
    price = discount_factor * payoffs.mean()
    return AsianResult(price=price, payoffs=payoffs)


def price_asian_geometric_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: OptionType = "call",
    seed: Optional[int] = None,
) -> AsianResult:
    """Price a geometric-average Asian option via Monte Carlo.

    The geometric average offers faster convergence and can be compared against
    the known closed-form solution for additional validation.
    """
    paths = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    log_prices = np.log(paths[:, 1:])
    geometric_means = np.exp(log_prices.mean(axis=1))
    payoffs = _asian_payoff(geometric_means, K, option_type)
    price = exp(-r * T) * payoffs.mean()
    return AsianResult(price=price, payoffs=payoffs)
