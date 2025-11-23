"""Black--Scholes closed-form pricing utilities.

This module provides analytical pricing formulas for European call and put
options under the assumptions of the Black--Scholes model. Functions include
pricing and delta computation with detailed type hints for clarity.
"""
from __future__ import annotations

from math import exp, log, sqrt
from typing import Literal, Tuple

from scipy.stats import norm

OptionType = Literal["call", "put"]


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    """Compute the d1 and d2 terms used in Black--Scholes formulas.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free annual interest rate with continuous compounding.
    sigma : float
        Annualized volatility of the underlying asset.
    T : float
        Time to maturity in years.

    Returns
    -------
    Tuple[float, float]
        The tuple (d1, d2) used in pricing and Greeks.
    """
    if T <= 0:
        raise ValueError("Time to maturity T must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")

    numerator = log(S / K) + (r + 0.5 * sigma**2) * T
    denominator = sigma * sqrt(T)
    d1 = numerator / denominator
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float, K: float, r: float, sigma: float, T: float, option_type: OptionType
) -> float:
    """Compute the Black--Scholes price of a European option.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free annual interest rate with continuous compounding.
    sigma : float
        Annualized volatility of the underlying asset.
    T : float
        Time to maturity in years.
    option_type : {"call", "put"}
        Type of the European option.

    Returns
    -------
    float
        Black--Scholes option price.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)

    discount = exp(-r * T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * norm.cdf(d2) * discount
    elif option_type == "put":
        price = K * norm.cdf(-d2) * discount - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")
    return price


def black_scholes_delta(
    S: float, K: float, r: float, sigma: float, T: float, option_type: OptionType
) -> float:
    """Compute the delta of a European option under Black--Scholes.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free annual interest rate with continuous compounding.
    sigma : float
        Annualized volatility of the underlying asset.
    T : float
        Time to maturity in years.
    option_type : {"call", "put"}
        Type of the European option.

    Returns
    -------
    float
        Delta of the option.
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)

    if option_type == "call":
        return norm.cdf(d1)
    if option_type == "put":
        return norm.cdf(d1) - 1
    raise ValueError("option_type must be either 'call' or 'put'.")
