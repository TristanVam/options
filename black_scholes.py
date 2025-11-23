"""Black-Scholes option pricing and Greeks.

This module implements Black-Scholes pricing for European-style options.
Although the user prompt mentions American options with exercise only at maturity,
this aligns with European exercise, so the closed-form Black-Scholes formulas apply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from scipy.stats import norm


@dataclass
class BlackScholesParams:
    """Container for Black-Scholes inputs."""

    spot: float
    strike: float
    risk_free_rate: float
    volatility: float
    maturity: float
    option_type: str


def _validate_option_type(option_type: str) -> None:
    """Ensure the option type is supported.

    Raises:
        ValueError: If the option type is not "call" or "put".
    """

    if option_type not in {"call", "put"}:
        raise ValueError('option_type must be "call" or "put"')


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    """Return the d1 and d2 terms used in the Black-Scholes formula."""

    if T <= 0:
        raise ValueError("Maturity T must be positive")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black_scholes_price(
    S: float, K: float, r: float, sigma: float, T: float, option_type: str
) -> float:
    """Compute the Black-Scholes price for a European option.

    Args:
        S: Current price of the underlying asset.
        K: Strike price of the option.
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility of the underlying asset.
        T: Time to maturity in years.
        option_type: "call" or "put".

    Returns:
        Theoretical option price.
    """

    _validate_option_type(option_type)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    discount_factor = math.exp(-r * T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)
    else:  # put
        price = K * discount_factor * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def black_scholes_greeks(
    S: float, K: float, r: float, sigma: float, T: float, option_type: str
) -> Dict[str, float]:
    """Compute primary Greeks for a European option under Black-Scholes.

    Conventions:
        - Delta: change in price per unit change in underlying.
        - Gamma: change in delta per unit change in underlying.
        - Vega: change in price per 1 point (i.e., 1.0 = 100%) change in volatility.
        - Theta: change in price per year (continuously compounded rates).
        - Rho: change in price per 1 point change in the risk-free rate.

    Args:
        S: Current price of the underlying asset.
        K: Strike price of the option.
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility of the underlying asset.
        T: Time to maturity in years.
        option_type: "call" or "put".

    Returns:
        Dictionary with keys "delta", "gamma", "vega", "theta", and "rho".
    """

    _validate_option_type(option_type)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    discount_factor = math.exp(-r * T)
    sqrt_T = math.sqrt(T)

    if option_type == "call":
        delta = cdf_d1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            - r * K * discount_factor * cdf_d2
        )
        rho = K * T * discount_factor * cdf_d2
    else:
        delta = cdf_d1 - 1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            + r * K * discount_factor * norm.cdf(-d2)
        )
        rho = -K * T * discount_factor * norm.cdf(-d2)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


__all__ = [
    "BlackScholesParams",
    "black_scholes_price",
    "black_scholes_greeks",
    "_d1_d2",
]
