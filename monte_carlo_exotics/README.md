# Monte Carlo Pricing for Exotic Options

This mini-project implements CPU-based Monte Carlo pricing utilities for exotic
options (with emphasis on arithmetic Asian options) along with analytical
Black--Scholes benchmarks for European options.

## Project layout

- `black_scholes.py` – closed-form Black--Scholes pricing and delta.
- `mc_paths.py` – GBM path generation (single process + optional CPU parallel).
- `asian_options.py` – Monte Carlo pricing for arithmetic and geometric Asian options.
- `barrier_options.py` – Up-and-out barrier call Monte Carlo pricer.
- `analytics.py` – Convergence study helpers and MC vs Black--Scholes comparison.
- `visuals.py` – Matplotlib-based plotting helpers for convergence, payoffs, and paths.
- `demo_asian.py` – Example script showing Asian pricing, convergence, and validation.
- `demo_barrier.py` – Example script for pricing a barrier option.

Run scripts as modules to keep imports relative to the package:

```bash
python -m monte_carlo_exotics.demo_asian
python -m monte_carlo_exotics.demo_barrier
```

## Dependencies

The examples rely on common scientific Python libraries:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`

Install them with:

```bash
pip install numpy pandas scipy matplotlib
```

## Notes

- Simulations are run under the risk-neutral measure using a log-normal Euler
  discretization for GBM.
- Optional antithetic variates are available for the arithmetic Asian pricer.
- CPU-only parallelization is supported in `generate_gbm_paths_parallel` for
  handling very large path counts.
