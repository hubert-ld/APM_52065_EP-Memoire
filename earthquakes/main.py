"""
main.py — Entry point for the Hawkes earthquake study.

Pipeline:
    1. Load and preprocess the USGS earthquake catalogue
    2. Fit all five models by MLE
    3. Print parameter estimates and information criteria
    4. Compute compensators and run residual tests
    5. Save figures
"""

import os

import numpy as np
import pandas as pd

from src.estimation import fit_all_models
from src.diagnostics import compute_all_compensators, run_residual_tests
from src.metrics import print_information_criteria
from src.plotting import save_all_figures
from src.plotting import plot_earthquake_location_map


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/earthquake.csv"
FIG_DIR   = "figures"


# ============================================================
# 1. Data loading
# ============================================================

def load_data(path):
    """Load the USGS catalogue and return (t, M, M0)."""
    df = pd.read_csv(path)
    df = df[df["type"] == "earthquake"].copy()
    df = df.sort_values("time").reset_index(drop=True)

    time_raw = df["time"].to_numpy().astype("datetime64[s]").astype(float) / 86400.0
    t = time_raw - time_raw[0]
    t = t.astype(float)

    M  = df["mag"].to_numpy().astype(float)
    M0 = M.min()

    return df, t, M, M0


# ============================================================
# 2. Estimation summary
# ============================================================

def print_estimates(results):
    param_labels = {
        "poisson":      ("lambda",),
        "exp":          ("mu", "alpha", "beta"),
        "power":        ("mu", "K", "c", "p"),
        "exp_marked":   ("mu", "alpha", "beta", "gamma"),
        "power_marked": ("mu", "K", "c", "p", "gamma"),
    }
    model_names = {
        "poisson":      "Homogeneous Poisson",
        "exp":          "Exponential Hawkes (unmarked)",
        "power":        "Power-law Hawkes (unmarked)",
        "exp_marked":   "Exponential Hawkes (marked)",
        "power_marked": "Power-law Hawkes (marked)",
    }
    print("\n=== MLE parameter estimates ===")
    for key, res in results.items():
        print(f"\n{model_names[key]}")
        print(f"  success : {res.success} | {res.message}")
        labels = param_labels[key]
        params = ", ".join(f"{l}={v:.5f}" for l, v in zip(labels, res.x))
        print(f"  theta   : {params}")
        print(f"  NLL     : {res.fun:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    df, t, M, M0 = load_data(DATA_PATH)
    n = len(t)
    print(f"Events: {n}  |  Window: {t[0]:.1f} → {t[-1]:.1f} days")
    print(f"Magnitude range: {M.min():.1f} – {M.max():.1f}  (M0 = {M0:.1f})")
    
    # Plot map of the events
    plot_earthquake_location_map(df, os.path.join(FIG_DIR, "earthquake_location_map.png"), 
                                  title="Earthquake Locations") # Save the map figure as well

    # Fit
    results = fit_all_models(t, M, M0)
    print_estimates(results)
    print_information_criteria(results, n)

    # Diagnostics
    compensators = compute_all_compensators(results, t, M, M0)
    run_residual_tests(compensators)

    # Figures
    save_all_figures(t, compensators, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
