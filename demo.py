"""Demonstration script for Black-Scholes pricing and Greeks."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from black_scholes import black_scholes_greeks, black_scholes_price


DEFAULT_PARAMS = {
    "S": 100.0,
    "K": 100.0,
    "r": 0.02,
    "sigma": 0.2,
    "T": 1.0,
}


def print_examples() -> None:
    """Compute and print example prices and Greeks."""

    print("=== Black-Scholes Examples ===")
    call_price = black_scholes_price(**DEFAULT_PARAMS, option_type="call")
    put_price = black_scholes_price(**DEFAULT_PARAMS, option_type="put")
    call_greeks = black_scholes_greeks(**DEFAULT_PARAMS, option_type="call")

    print(f"Call price: {call_price:.4f}")
    print(f"Put price:  {put_price:.4f}")
    print("Call Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize():<5}: {value:.4f}")


def plot_prices_and_greeks() -> None:
    """Generate plots for prices, delta, and vega as functions of spot price."""

    spot_values = np.linspace(50, 150, 200)

    call_prices = np.array(
        [
            black_scholes_price(S=s, K=DEFAULT_PARAMS["K"], r=DEFAULT_PARAMS["r"], sigma=DEFAULT_PARAMS["sigma"], T=DEFAULT_PARAMS["T"], option_type="call")
            for s in spot_values
        ]
    )
    put_prices = np.array(
        [
            black_scholes_price(S=s, K=DEFAULT_PARAMS["K"], r=DEFAULT_PARAMS["r"], sigma=DEFAULT_PARAMS["sigma"], T=DEFAULT_PARAMS["T"], option_type="put")
            for s in spot_values
        ]
    )

    call_greeks = [
        black_scholes_greeks(
            S=s,
            K=DEFAULT_PARAMS["K"],
            r=DEFAULT_PARAMS["r"],
            sigma=DEFAULT_PARAMS["sigma"],
            T=DEFAULT_PARAMS["T"],
            option_type="call",
        )
        for s in spot_values
    ]

    put_greeks = [
        black_scholes_greeks(
            S=s,
            K=DEFAULT_PARAMS["K"],
            r=DEFAULT_PARAMS["r"],
            sigma=DEFAULT_PARAMS["sigma"],
            T=DEFAULT_PARAMS["T"],
            option_type="put",
        )
        for s in spot_values
    ]

    call_deltas = np.array([g["delta"] for g in call_greeks])
    put_deltas = np.array([g["delta"] for g in put_greeks])
    call_vegas = np.array([g["vega"] for g in call_greeks])

    plt.figure(figsize=(10, 8))

    # Price plot
    plt.subplot(3, 1, 1)
    plt.plot(spot_values, call_prices, label="Call price", color="tab:blue")
    plt.plot(spot_values, put_prices, label="Put price", color="tab:orange")
    plt.title("Option price vs Spot price")
    plt.xlabel("Spot price (S)")
    plt.ylabel("Option price")
    plt.legend()
    plt.grid(True)

    # Delta plot
    plt.subplot(3, 1, 2)
    plt.plot(spot_values, call_deltas, label="Call delta", color="tab:green")
    plt.plot(spot_values, put_deltas, label="Put delta", color="tab:red")
    plt.title("Delta vs Spot price")
    plt.xlabel("Spot price (S)")
    plt.ylabel("Delta")
    plt.legend()
    plt.grid(True)

    # Vega plot
    plt.subplot(3, 1, 3)
    plt.plot(spot_values, call_vegas, label="Call vega", color="tab:purple")
    plt.title("Vega vs Spot price")
    plt.xlabel("Spot price (S)")
    plt.ylabel("Vega")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print_examples()
    plot_prices_and_greeks()
