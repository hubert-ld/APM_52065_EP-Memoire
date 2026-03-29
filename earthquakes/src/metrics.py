"""
Information criteria (AIC / BIC) for model comparison.
"""

import numpy as np


def aic(nll, k):
    """Akaike Information Criterion.  nll: negative log-likelihood, k: nb params."""
    return 2 * k + 2 * nll


def bic(nll, k, n):
    """Bayesian Information Criterion.  n: number of observations."""
    return k * np.log(n) + 2 * nll


# Number of free parameters per model
N_PARAMS = {
    "poisson":      1,
    "exp":          3,
    "power":        4,
    "exp_marked":   4,
    "power_marked": 5,
}


def print_information_criteria(results, n):
    """
    Print AIC and BIC for all models.

    Parameters
    ----------
    results : dict of OptimizeResult (output of estimation.fit_all_models)
    n       : number of observations
    """
    labels = {
        "poisson":      "Poisson          ",
        "exp":          "Hawkes exp       ",
        "power":        "Hawkes power     ",
        "exp_marked":   "Hawkes exp marked",
        "power_marked": "Hawkes pow marked",
    }
    print("\n=== Information criteria ===")
    for key, res in results.items():
        k = N_PARAMS[key]
        a = aic(res.fun, k)
        b = bic(res.fun, k, n)
        print(f"{labels[key]}  AIC = {a:.3f}  BIC = {b:.3f}")
