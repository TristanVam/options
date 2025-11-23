"""Black-Scholes pricing utilities for European options.

This module provides the classic closed-form Black-Scholes valuation for
European call and put options under the assumption of a geometric Brownian
motion for the underlying asset. The risk-free rate ``r`` is interpreted as a
continuously compounded annual rate.
"""
from __future__ import annotations

from math import log, sqrt
from typing import Tuple

import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    """
    Compute the Black-Scholes ``d1`` and ``d2`` terms.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Continuously compounded risk-free rate (annualized).
    sigma : float
        Annualized volatility of the underlying.
    T : float
        Time to maturity in years.

    Returns
    -------
    tuple[float, float]
        The ``d1`` and ``d2`` parameters used in Black-Scholes pricing.
    """
    if sigma <= 0 or T <= 0:
        raise ValueError("Sigma and T must be positive to compute d1 and d2.")

    denom = sigma * sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / denom
    d2 = d1 - denom
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
) -> float:
    """
    Compute the Black-Scholes price of a European option.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Continuously compounded risk-free rate (annualized).
    sigma : float
        Annualized volatility of the underlying.
    T : float
        Time to maturity in years.
    option_type : str
        Type of the option: either ``"call"`` or ``"put"`` (case-insensitive).

    Returns
    -------
    float
        The theoretical Black-Scholes option price.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be strictly positive.")

    d1, d2 = _d1_d2(S, K, r, sigma, T)
    discounted_strike = K * np.exp(-r * T)

    if option_type == "call":
        price = S * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    else:
        price = discounted_strike * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)
