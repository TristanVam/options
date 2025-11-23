"""Demo script to compute and plot volatility smiles."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .data_loader import generate_mock_data, load_option_data
from .vol_smile import compute_all_smiles
from .visuals import plot_multiple_smiles



def main() -> None:
    """Load option data, compute implied vols, and plot smiles for sample maturities."""
    try:
        df = load_option_data("option_chain.csv")
    except FileNotFoundError:
        df = generate_mock_data()

    df_iv = compute_all_smiles(df)

    maturities = sorted(df_iv["maturity"].unique())
    selected = maturities[:3] if len(maturities) > 3 else maturities

    smiles = {}
    for T in selected:
        subset = df_iv[df_iv["maturity"] == T]
        smiles[T] = (subset["strike"].to_numpy(), subset["implied_vol"].to_numpy())

    plot_multiple_smiles(smiles)
    plt.show()


if __name__ == "__main__":
    main()
