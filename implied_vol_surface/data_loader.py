"""Utilities for loading option chain data from CSV files."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_option_data(filepath: str) -> pd.DataFrame:
    """
    Load option data from a CSV file into a cleaned DataFrame.

    Expected columns include:
    - underlying_price: spot of the underlying
    - strike: option strike
    - maturity: time to maturity in years (float). If provided as dates, convert
      to year fractions prior to loading.
    - rate: annual risk-free rate (continuous compounding assumed elsewhere)
    - option_price: observed option market price
    - option_type: "call" or "put"
    """
    df = pd.read_csv(filepath)
    expected = {"underlying_price", "strike", "maturity", "rate", "option_price", "option_type"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["underlying_price"] = df["underlying_price"].astype(float)
    df["strike"] = df["strike"].astype(float)
    df["maturity"] = df["maturity"].astype(float)
    df["rate"] = df["rate"].astype(float)
    df["option_price"] = df["option_price"].astype(float)
    df["option_type"] = df["option_type"].str.lower()

    valid_types = {"call", "put"}
    if not df["option_type"].isin(valid_types).all():
        invalid = df.loc[~df["option_type"].isin(valid_types), "option_type"].unique()
        raise ValueError(f"Invalid option_type values: {invalid}")

    return df


def generate_mock_data(
    n_strikes: int = 7,
    maturities: list[float] | None = None,
    spot: float = 100.0,
    rate: float = 0.01,
    base_vol: float = 0.2,
    noise: float = 0.01,
) -> pd.DataFrame:
    """
    Generate a simple synthetic option chain for demonstration purposes.

    The generated prices use a quadratic smile around the spot and a mild term
    structure. Results are meant only for demo/testing, not for calibration.
    """
    from .black_scholes import black_scholes_price

    maturities = maturities or [0.1, 0.25, 0.5, 1.0]
    strikes = np.linspace(0.7 * spot, 1.3 * spot, n_strikes)

    records = []
    for T in maturities:
        for K in strikes:
            skew = 0.15 * ((K / spot - 1) ** 2)
            sigma = base_vol + skew + 0.05 * np.log1p(T)
            sigma = max(sigma, 1e-4)
            call_price = black_scholes_price(spot, K, rate, sigma, T, "call")
            put_price = black_scholes_price(spot, K, rate, sigma, T, "put")
            records.append(
                {
                    "underlying_price": spot,
                    "strike": K,
                    "maturity": T,
                    "rate": rate,
                    "option_price": call_price + np.random.normal(0, noise),
                    "option_type": "call",
                }
            )
            records.append(
                {
                    "underlying_price": spot,
                    "strike": K,
                    "maturity": T,
                    "rate": rate,
                    "option_price": put_price + np.random.normal(0, noise),
                    "option_type": "put",
                }
            )

    return pd.DataFrame(records)
