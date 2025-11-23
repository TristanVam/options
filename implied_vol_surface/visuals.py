"""Plotting utilities for volatility smiles and surfaces."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D plotting


def plot_vol_smile(strikes: np.ndarray, implied_vols: np.ndarray, maturity: float) -> None:
    """Plot a single volatility smile for a given maturity."""
    plt.figure(figsize=(8, 5))
    plt.plot(strikes, implied_vols, marker="o", label=f"T={maturity:.2f}y")
    plt.title(f"Implied Volatility Smile (T={maturity:.2f}y)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.legend()


def plot_multiple_smiles(smiles_data: dict[float, tuple[np.ndarray, np.ndarray]]) -> None:
    """Plot multiple volatility smiles on the same axes."""
    plt.figure(figsize=(10, 6))
    for maturity, (strikes, vols) in smiles_data.items():
        plt.plot(strikes, vols, marker="o", label=f"T={maturity:.2f}y")
    plt.title("Implied Volatility Smiles")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.legend()


def plot_vol_surface_3d(
    strikes_grid: np.ndarray, maturities_grid: np.ndarray, iv_grid: np.ndarray
) -> None:
    """Plot a 3D surface of implied volatilities."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(strikes_grid, maturities_grid, iv_grid, cmap="viridis")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Implied Volatility Surface")
    fig.colorbar(surf, shrink=0.5, aspect=10)


def plot_vol_surface_heatmap(
    strikes_grid: np.ndarray, maturities_grid: np.ndarray, iv_grid: np.ndarray
) -> None:
    """Plot a heatmap of the implied volatility surface."""
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    mesh = plt.pcolormesh(strikes_grid, maturities_grid, iv_grid, shading="auto", cmap=cmap)
    plt.xlabel("Strike")
    plt.ylabel("Maturity")
    plt.title("Implied Volatility Surface (Heatmap)")
    plt.colorbar(mesh, label="Implied Volatility")
    plt.grid(True)
