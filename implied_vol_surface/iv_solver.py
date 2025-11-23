"""Implied volatility solvers using Black-Scholes pricing.

This module provides Newton-Raphson and bisection routines to back out the
implied volatility from observed option prices. Newton is attempted first for
speed; in case of non-convergence the algorithm falls back to a robust
bisection method.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .black_scholes import black_scholes_price, _d1_d2


def _vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Compute Black-Scholes vega (derivative of price with respect to sigma)."""
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return float(S * np.sqrt(T) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi))


def _newton(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str,
    initial_sigma: float,
    tol: float,
    max_iterations: int,
) -> float:
    sigma = initial_sigma
    for _ in range(max_iterations):
        bs_price = black_scholes_price(S, K, r, sigma, T, option_type)
        diff = bs_price - price
        if abs(diff) < tol:
            return sigma
        vega = _vega(S, K, r, sigma, T)
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if sigma <= 1e-4 or sigma > 5:
            break
    raise RuntimeError("Newton-Raphson did not converge to a valid volatility.")


def implied_volatility_newton(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str,
    initial_sigma: float = 0.2,
    tol: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """
    Estimate implied volatility with the Newton-Raphson algorithm.

    Parameters follow the Black-Scholes conventions; the risk-free rate ``r``
    uses continuous compounding. The method iteratively adjusts ``sigma`` using
    the local derivative (vega). If the iteration exits the bounds ``[1e-4, 5]``
    or fails to reach ``tol`` within ``max_iterations``, a ``RuntimeError`` is
    raised.
    """
    return _newton(price, S, K, r, T, option_type, initial_sigma, tol, max_iterations)


def implied_volatility_bisection(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str,
    sigma_low: float = 1e-4,
    sigma_high: float = 5.0,
    tol: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """
    Estimate implied volatility using a robust bisection method.

    The function searches for a volatility ``sigma`` such that the pricing error
    changes sign between ``sigma_low`` and ``sigma_high``. If no sign change is
    detected, the function returns ``np.nan`` to signal failure.
    """
    def price_error(sig: float) -> float:
        return black_scholes_price(S, K, r, sig, T, option_type) - price

    low, high = sigma_low, sigma_high
    f_low, f_high = price_error(low), price_error(high)

    if np.sign(f_low) == np.sign(f_high):
        return np.nan

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        f_mid = price_error(mid)
        if abs(f_mid) < tol or (high - low) < tol:
            return mid
        if np.sign(f_mid) == np.sign(f_low):
            low, f_low = mid, f_mid
        else:
            high, f_high = mid, f_mid

    return np.nan


def implied_volatility(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str,
    initial_sigma: float = 0.2,
) -> float:
    """
    High-level implied volatility solver.

    Newton-Raphson is attempted first for speed; upon failure the function
    falls back to bisection. If both methods fail, ``np.nan`` is returned.
    """
    try:
        return implied_volatility_newton(
            price,
            S,
            K,
            r,
            T,
            option_type,
            initial_sigma=initial_sigma,
        )
    except Exception:
        return implied_volatility_bisection(price, S, K, r, T, option_type)
