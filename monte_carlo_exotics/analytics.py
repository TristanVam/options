"""Analytical helpers for Monte Carlo diagnostics and validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_price
from .mc_paths import generate_gbm_paths

OptionType = Literal["call", "put"]


@dataclass
class ConvergenceResult:
    n_paths: int
    price: float


def price_european_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: OptionType = "call",
    seed: int | None = None,
) -> float:
    """Monte Carlo pricer for a European option used for validation."""
    paths = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    terminal = paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(terminal - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - terminal, 0.0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")
    return float(np.exp(-r * T) * payoff.mean())


def estimate_mc_convergence(
    pricing_func: Callable[..., float], n_paths_list: Sequence[int], **kwargs
) -> pd.DataFrame:
    """Estimate convergence of a Monte Carlo pricer over multiple path counts."""
    results = []
    for n_paths in n_paths_list:
        price = pricing_func(n_paths=n_paths, **kwargs)
        results.append(ConvergenceResult(n_paths=n_paths, price=price))
    return pd.DataFrame(results)


def compare_mc_vs_black_scholes(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: OptionType = "call",
    seed: int | None = None,
) -> dict[str, float]:
    """Compare Monte Carlo pricing against the Black--Scholes benchmark."""
    mc_price = price_european_mc(S0, K, r, sigma, T, n_steps, n_paths, option_type, seed)
    bs_price = black_scholes_price(S0, K, r, sigma, T, option_type)
    absolute_error = abs(mc_price - bs_price)
    relative_error = absolute_error / bs_price if bs_price != 0 else np.nan
    return {
        "mc_price": mc_price,
        "bs_price": bs_price,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }
