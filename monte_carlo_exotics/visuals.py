"""Visualization helpers for Monte Carlo simulations."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mc_convergence(df_convergence: pd.DataFrame, true_value: float | None = None):
    """Plot Monte Carlo price convergence as a function of path count."""
    fig, ax = plt.subplots()
    ax.plot(df_convergence["n_paths"], df_convergence["price"], marker="o", label="MC price")
    if true_value is not None:
        ax.axhline(true_value, color="red", linestyle="--", label="Benchmark")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Option price")
    ax.set_title("Monte Carlo convergence")
    ax.grid(True)
    ax.legend()
    return fig, ax


def plot_payoff_distribution(payoffs: np.ndarray, title: str = ""):
    """Plot histogram for payoff samples."""
    fig, ax = plt.subplots()
    ax.hist(payoffs, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    mean_payoff = payoffs.mean()
    ax.axvline(mean_payoff, color="red", linestyle="--", label=f"Mean: {mean_payoff:.4f}")
    ax.set_title(title or "Payoff distribution")
    ax.set_xlabel("Payoff")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    return fig, ax


def plot_paths_subset(paths: np.ndarray, n_to_plot: int = 10, title: str = ""):
    """Plot a subset of simulated paths."""
    fig, ax = plt.subplots()
    for path in paths[:n_to_plot]:
        ax.plot(path, alpha=0.7)
    ax.set_title(title or f"Sample of {n_to_plot} simulated paths")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Underlying price")
    ax.grid(True)
    return fig, ax
