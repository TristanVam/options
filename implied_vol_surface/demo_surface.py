"""Demo script to build and visualize an implied volatility surface."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .data_loader import generate_mock_data, load_option_data
from .vol_smile import compute_all_smiles
from .vol_surface import build_vol_surface
from .visuals import plot_vol_surface_3d, plot_vol_surface_heatmap



def main() -> None:
    """Load data, compute implied vols, interpolate a surface, and plot it."""
    try:
        df = load_option_data("option_chain.csv")
    except FileNotFoundError:
        df = generate_mock_data()

    df_iv = compute_all_smiles(df)
    surface = build_vol_surface(df_iv)

    plot_vol_surface_3d(
        surface["strikes_grid"], surface["maturities_grid"], surface["iv_grid"]
    )
    plot_vol_surface_heatmap(
        surface["strikes_grid"], surface["maturities_grid"], surface["iv_grid"]
    )
    plt.show()


if __name__ == "__main__":
    main()
