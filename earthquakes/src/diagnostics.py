"""
Residual diagnostics via Ogata's time-rescaling theorem.

Under correct specification, the rescaled inter-arrivals Δτ_i should be
i.i.d. Exp(1). We provide compensator functions and goodness-of-fit tests.
"""

import numpy as np
from scipy import stats


# ============================================================
# Compensators (integrated intensities at event times)
# ============================================================

def compensator_poisson(theta, t):
    """Λ(t_i) for the homogeneous Poisson model."""
    (lam,) = theta
    return lam * t


def compensator_exp(theta, t):
    """Λ(t_i) for the exponential Hawkes model."""
    mu, alpha, beta = theta
    n = len(t)
    tau = np.empty(n)
    for j in range(n):
        tj = t[j]
        mask_j = t < tj
        if np.any(mask_j):
            dt = tj - t[mask_j]
            arg = np.clip(-beta * dt, -700.0, 700.0)
            contrib = (alpha / beta) * np.sum(1.0 - np.exp(arg))
        else:
            contrib = 0.0
        tau[j] = mu * tj + contrib
    return tau


def compensator_power(theta, t):
    """Λ(t_i) for the power-law Hawkes model."""
    mu, K_param, c, p = theta
    n = len(t)
    tau = np.empty(n)
    for j in range(n):
        tj = t[j]
        mask_j = t < tj
        if np.any(mask_j):
            dt = tj - t[mask_j]
            term = (1.0 / c ** (p - 1)) - 1.0 / (c + dt) ** (p - 1)
            contrib = (K_param / (p - 1)) * np.sum(term)
        else:
            contrib = 0.0
        tau[j] = mu * tj + contrib
    return tau


def compensator_exp_marked(theta, t, M, M0):
    """Λ(t_i) for the exponential marked Hawkes model."""
    mu, alpha, beta, gamma = theta
    n = len(t)
    w = np.exp(gamma * (M - M0))
    tau = np.empty(n)
    for j in range(n):
        tj = t[j]
        mask_j = t < tj
        if np.any(mask_j):
            dt = tj - t[mask_j]
            wj = w[mask_j]
            arg = np.clip(-beta * dt, -700.0, 700.0)
            contrib = (alpha / beta) * np.sum(wj * (1.0 - np.exp(arg)))
        else:
            contrib = 0.0
        tau[j] = mu * tj + contrib
    return tau


def compensator_power_marked(theta, t, M, M0):
    """Λ(t_i) for the power-law marked Hawkes model."""
    mu, K_param, c, p, gamma = theta
    n = len(t)
    w = np.exp(gamma * (M - M0))
    tau = np.empty(n)
    for j in range(n):
        tj = t[j]
        mask_j = t < tj
        if np.any(mask_j):
            dt = tj - t[mask_j]
            wj = w[mask_j]
            term = (1.0 / c ** (p - 1)) - 1.0 / (c + dt) ** (p - 1)
            contrib = (K_param / (p - 1)) * np.sum(wj * term)
        else:
            contrib = 0.0
        tau[j] = mu * tj + contrib
    return tau


# ============================================================
# Goodness-of-fit tests on rescaled inter-arrivals
# ============================================================

def ks_exp_test(delta_tau):
    """Kolmogorov–Smirnov test against Exp(1)."""
    stat, pval = stats.kstest(delta_tau, "expon")
    return stat, pval


def chi2_exp_test(delta_tau, B=10):
    """Chi-squared goodness-of-fit test for Exp(1) using B equiprobable bins."""
    n = len(delta_tau)
    quantiles = np.linspace(0, 1, B + 1)
    edges = -np.log(1.0 - quantiles[:-1])
    edges = np.append(edges, np.inf)
    O = np.histogram(delta_tau, bins=edges)[0]
    E = np.full(B, n / B)
    chi2_stat = ((O - E) ** 2 / E).sum()
    p_val = 1.0 - stats.chi2.cdf(chi2_stat, B - 1)
    return chi2_stat, p_val


# ============================================================
# High-level runner
# ============================================================

def compute_all_compensators(results, t, M, M0):
    """
    Compute compensators for all fitted models.

    Parameters
    ----------
    results : dict of OptimizeResult, keyed as in estimation.fit_all_models
    t, M, M0 : data arrays

    Returns
    -------
    dict of 1-D arrays (compensator values at event times)
    """
    return {
        "poisson":     compensator_poisson(results["poisson"].x, t),
        "exp":         compensator_exp(results["exp"].x, t),
        "power":       compensator_power(results["power"].x, t),
        "exp_marked":  compensator_exp_marked(results["exp_marked"].x, t, M, M0),
        "power_marked": compensator_power_marked(results["power_marked"].x, t, M, M0),
    }


def run_residual_tests(compensators, B=10):
    """
    Run KS and chi-squared tests for all models and print a summary table.

    Parameters
    ----------
    compensators : dict of compensator arrays (output of compute_all_compensators)
    B            : number of equiprobable bins for chi-squared test
    """
    labels = {
        "poisson":      "Poisson      ",
        "exp":          "Hawkes exp   ",
        "power":        "Hawkes power ",
        "exp_marked":   "Hawkes exp M ",
        "power_marked": "Hawkes pow M ",
    }
    print("\n=== Residual tests (time-rescaling) ===")
    for key, tau in compensators.items():
        dt = np.diff(tau)
        ks_stat, ks_p = ks_exp_test(dt)
        chi2_stat, chi2_p = chi2_exp_test(dt, B=B)
        print(
            f"{labels[key]}  "
            f"KS: stat={ks_stat:.4f}  p={ks_p:.3e}  |  "
            f"Chi2: stat={chi2_stat:.2f}  p={chi2_p:.3e}"
        )
