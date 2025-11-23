"""Build an implied volatility surface by interpolating smile data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def build_vol_surface(df_iv: pd.DataFrame, grid_size: int = 50) -> dict:
    """
    Interpolate an implied volatility surface from scattered (strike, maturity)
    observations.

    Parameters
    ----------
    df_iv : pd.DataFrame
        DataFrame containing ``strike``, ``maturity`` and ``implied_vol`` columns.
    grid_size : int
        Number of points along each axis for the output grid.

    Returns
    -------
    dict
        A dictionary containing ``strikes_grid``, ``maturities_grid`` and
        ``iv_grid`` suitable for plotting.
    """
    if df_iv.empty:
        raise ValueError("Input DataFrame is empty; cannot build vol surface.")

    strikes = df_iv["strike"].values
    maturities = df_iv["maturity"].values
    ivs = df_iv["implied_vol"].values

    strike_grid = np.linspace(strikes.min(), strikes.max(), grid_size)
    maturity_grid = np.linspace(maturities.min(), maturities.max(), grid_size)
    strikes_grid, maturities_grid = np.meshgrid(strike_grid, maturity_grid)

    iv_grid = griddata(
        points=(strikes, maturities),
        values=ivs,
        xi=(strikes_grid, maturities_grid),
        method="linear",
    )

    # Fill potential gaps with nearest-neighbor interpolation
    if np.isnan(iv_grid).any():
        iv_grid_nn = griddata(
            points=(strikes, maturities),
            values=ivs,
            xi=(strikes_grid, maturities_grid),
            method="nearest",
        )
        nan_mask = np.isnan(iv_grid)
        iv_grid[nan_mask] = iv_grid_nn[nan_mask]

    return {
        "strikes_grid": strikes_grid,
        "maturities_grid": maturities_grid,
        "iv_grid": iv_grid,
    }
