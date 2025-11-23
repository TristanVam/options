"""Demonstration script for pricing barrier options via Monte Carlo."""
from __future__ import annotations

from monte_carlo_exotics.barrier_options import price_barrier_up_and_out_call_mc


if __name__ == "__main__":
    S0 = 100.0
    K = 100.0
    H = 130.0
    r = 0.05
    sigma = 0.25
    T = 1.0
    n_steps = 252

    price = price_barrier_up_and_out_call_mc(
        S0=S0,
        K=K,
        H=H,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=100_000,
        seed=7,
    )
    print(f"Up-and-out barrier call price: {price:.4f}")
