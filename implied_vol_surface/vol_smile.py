"""Construct implied volatility smiles for given maturities."""
from __future__ import annotations

import pandas as pd

from .iv_solver import implied_volatility


def compute_iv_for_maturity(df: pd.DataFrame, maturity: float) -> pd.DataFrame:
    """
    Compute implied volatilities for all strikes at a specific maturity.

    Parameters
    ----------
    df : pd.DataFrame
        Option chain containing columns: underlying_price, strike, maturity,
        rate, option_price, option_type.
    maturity : float
        Target maturity in years.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with an additional ``implied_vol`` column.
    """
    subset = df[df["maturity"].astype(float) == maturity].copy()
    if subset.empty:
        return subset.assign(implied_vol=pd.NA)

    subset["implied_vol"] = subset.apply(
        lambda row: implied_volatility(
            row.option_price,
            row.underlying_price,
            row.strike,
            row.rate,
            row.maturity,
            row.option_type,
        ),
        axis=1,
    )
    return subset


def compute_all_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute implied volatilities across all strikes and maturities.
    """
    maturities = df["maturity"].unique()
    frames = [compute_iv_for_maturity(df, float(T)) for T in maturities]
    return pd.concat(frames, ignore_index=True)
