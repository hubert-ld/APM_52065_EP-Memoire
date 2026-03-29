"""
Maximum likelihood estimation for all point process models.

Provides a multi-start L-BFGS-B optimiser and a convenience function
that fits all five models and returns their results in a dict.
"""

import numpy as np
from scipy import optimize

from .models import (
    nll_poisson,
    nll_hawkes_exp,
    nll_hawkes_power,
    nll_hawkes_exp_marked,
    nll_hawkes_power_marked,
)


def multistart_minimize(fun, x0_list, bounds, args=()):
    """Run L-BFGS-B from several starting points; return the best result."""
    best = None
    for x0 in x0_list:
        if not np.isfinite(fun(x0, *args)):
            continue
        res = optimize.minimize(
            fun,
            x0=np.array(x0, dtype=float),
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("No feasible starting point found.")
    return best


def fit_all_models(t, M, M0):
    """
    Fit all five models by MLE and return results as a dict.

    Parameters
    ----------
    t  : 1-D array of event times (days, starting at 0)
    M  : 1-D array of magnitudes (same length as t)
    M0 : minimum magnitude (completeness threshold)

    Returns
    -------
    dict with keys: 'poisson', 'exp', 'power', 'exp_marked', 'power_marked'
    Each value is a scipy OptimizeResult.
    """
    n = len(t)
    rate = n / t[-1]

    # ------------------------------------------------------------------
    # Poisson (closed-form, but kept consistent through minimize)
    # ------------------------------------------------------------------
    res_pois = optimize.minimize(
        nll_poisson,
        x0=[rate],
        args=(t,),
        method="L-BFGS-B",
        bounds=[(1e-9, None)],
    )

    # ------------------------------------------------------------------
    # Exponential Hawkes (unmarked)
    # ------------------------------------------------------------------
    bounds_exp = [(1e-9, None), (1e-9, None), (1e-3, 250.0)]
    x0s_exp = [
        [0.5 * rate, 0.3,  0.6],
        [0.3 * rate, 0.5,  1.0],
        [0.2 * rate, 0.1,  0.5],
        [0.1 * rate, 1.0,  5.0],
        [0.5 * rate, 2.0, 10.0],
    ]
    res_exp = multistart_minimize(nll_hawkes_exp, x0s_exp, bounds_exp, args=(t,))

    # ------------------------------------------------------------------
    # Power-law Hawkes (unmarked)
    # ------------------------------------------------------------------
    bounds_pow = [(1e-9, None), (1e-9, None), (1e-4, None), (1.05, 5.0)]
    x0s_pow = [
        [0.5 * rate, 0.05, 0.1, 1.5],
        [0.3 * rate, 0.01, 0.5, 1.5],
        [0.2 * rate, 0.10, 1.0, 2.0],
        [0.1 * rate, 0.01, 0.1, 2.0],
        [0.5 * rate, 0.30, 5.0, 2.0],
    ]
    res_pow = multistart_minimize(nll_hawkes_power, x0s_pow, bounds_pow, args=(t,))

    # ------------------------------------------------------------------
    # Exponential marked Hawkes
    # ------------------------------------------------------------------
    bounds_exp_m = [(1e-9, None), (1e-9, None), (1e-3, 250.0), (-5.0, 5.0)]
    x0_exp_m = [0.1 * rate, 0.5 * rate, 1.0, 0.0]
    res_exp_m = optimize.minimize(
        nll_hawkes_exp_marked,
        x0=x0_exp_m,
        args=(t, M, M0),
        method="L-BFGS-B",
        bounds=bounds_exp_m,
    )

    # ------------------------------------------------------------------
    # Power-law marked Hawkes
    # ------------------------------------------------------------------
    bounds_pow_m = [(1e-9, None), (1e-9, None), (1e-4, None), (1.05, 5.0), (-5.0, 5.0)]
    x0_pow_m = [0.1 * rate, 0.5 * rate, 0.01, 1.5, 0.0]
    res_pow_m = optimize.minimize(
        nll_hawkes_power_marked,
        x0=x0_pow_m,
        args=(t, M, M0),
        method="L-BFGS-B",
        bounds=bounds_pow_m,
    )

    return {
        "poisson":     res_pois,
        "exp":         res_exp,
        "power":       res_pow,
        "exp_marked":  res_exp_m,
        "power_marked": res_pow_m,
    }
