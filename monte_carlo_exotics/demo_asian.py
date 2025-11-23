"""Demonstration script for pricing Asian options via Monte Carlo."""
from __future__ import annotations

from functools import partial

from monte_carlo_exotics.asian_options import price_asian_arithmetic_mc
from monte_carlo_exotics.analytics import (
    compare_mc_vs_black_scholes,
    estimate_mc_convergence,
)
from monte_carlo_exotics.visuals import plot_mc_convergence


if __name__ == "__main__":
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 252

    # Price an arithmetic Asian call with and without antithetic variates
    base_result = price_asian_arithmetic_mc(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=20_000,
        option_type="call",
        antithetic=False,
        seed=42,
    )
    anti_result = price_asian_arithmetic_mc(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=20_000,
        option_type="call",
        antithetic=True,
        seed=42,
    )
    print(f"Arithmetic Asian call (plain MC): {base_result.price:.4f}")
    print(f"Arithmetic Asian call (antithetic): {anti_result.price:.4f}")

    # Convergence study for the Asian option
    pricing_partial = partial(
        price_asian_arithmetic_mc,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        option_type="call",
        antithetic=True,
        seed=123,
    )
    df_conv = estimate_mc_convergence(pricing_func=pricing_partial, n_paths_list=[1_000, 5_000, 10_000, 20_000])
    print("\nConvergence results:")
    print(df_conv)

    try:
        import matplotlib.pyplot as plt

        plot_mc_convergence(df_conv)
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Plotting skipped: {exc}")

    # Validate Monte Carlo on a European option against Black--Scholes
    validation = compare_mc_vs_black_scholes(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=50_000,
        option_type="call",
        seed=99,
    )
    print("\nEuropean call validation (MC vs Black--Scholes):")
    for key, value in validation.items():
        print(f"  {key}: {value:.6f}")
