"""Monte Carlo path generation for geometric Brownian motion (GBM).

Functions here simulate risk-neutral GBM trajectories used for pricing options
via Monte Carlo. Both single-process and optional CPU-parallelized versions are
provided to scale across many simulated paths.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Optional

import numpy as np


def _simulate_chunk(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate a chunk of GBM paths for internal parallel use."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * z)
    return paths


def generate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate GBM paths using vectorized simulation.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    r : float
        Risk-free annual interest rate (continuous compounding).
    sigma : float
        Annualized volatility of the underlying asset.
    T : float
        Maturity in years.
    n_steps : int
        Number of time steps for the discretization.
    n_paths : int
        Number of simulated Monte Carlo paths.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1) containing simulated paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * z)
    return paths


def generate_gbm_paths_parallel(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    n_workers: int | None = None,
    chunk_size: int = 10_000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate GBM paths using CPU parallelization.

    The function splits the total number of paths into chunks that are
    simulated across multiple worker processes. Each chunk receives a derived
    seed to keep reproducibility when desired.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    r : float
        Risk-free annual interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    T : float
        Maturity in years.
    n_steps : int
        Number of time steps for the discretization.
    n_paths : int
        Total number of simulated paths.
    n_workers : int | None, optional
        Number of worker processes. Defaults to the number of CPU cores.
    chunk_size : int, optional
        Number of paths simulated per worker chunk.
    seed : int | None, optional
        Base random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1) with simulated paths.
    """
    if n_paths <= chunk_size:
        return generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)

    rng = np.random.default_rng(seed)
    seeds: Iterable[int] = rng.integers(0, 1_000_000_000, size=(n_paths + chunk_size - 1) // chunk_size)

    chunk_sizes = [chunk_size] * (n_paths // chunk_size)
    remainder = n_paths % chunk_size
    if remainder:
        chunk_sizes.append(remainder)

    args = [
        (S0, r, sigma, T, n_steps, size, int(seed_val))
        for size, seed_val in zip(chunk_sizes, seeds)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(_simulate_chunk, *zip(*args))

    collected = [chunk for chunk in results]
    return np.vstack(collected)
