"""
Log-likelihood functions for all point process models.

Each function takes a parameter vector theta and returns the negative
log-likelihood (NLL). No I/O, no plotting — pure numerical functions.
"""

import numpy as np


# ============================================================
# Homogeneous Poisson process
# ============================================================

def nll_poisson(theta, t):
    """Negative log-likelihood for a homogeneous Poisson process."""
    (lam,) = theta
    if lam <= 0:
        return np.inf
    T = t[-1]
    n = len(t)
    loglik = n * np.log(lam) - lam * T
    return -loglik if np.isfinite(loglik) else np.inf


# ============================================================
# Exponential Hawkes process (unmarked)
# λ(t) = μ + Σ_{t_i < t} α exp(−β(t − t_i)),  with α/β < 1
# ============================================================

def nll_hawkes_exp(theta, t):
    """Negative log-likelihood for an exponential Hawkes process."""
    mu, alpha, beta = theta
    if mu <= 0 or alpha <= 0 or beta <= 0:
        return np.inf
    if alpha / beta >= 1.0:
        return np.inf

    T = t[-1]
    diff = t[:, None] - t[None, :]
    mask = diff > 0.0

    arg = np.where(mask, -beta * diff, 0.0)
    arg = np.clip(arg, -700.0, 700.0)
    K = np.where(mask, alpha * np.exp(arg), 0.0)
    lam = mu + K.sum(axis=1)

    if not np.isfinite(lam).all() or np.any(lam <= 1e-12):
        return np.inf

    loglik_events = np.log(lam).sum()
    arg_integ = np.clip(-beta * (T - t), -700.0, 700.0)
    integ = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(arg_integ))

    loglik = loglik_events - integ
    return -loglik if np.isfinite(loglik) else np.inf


# ============================================================
# Power-law Hawkes process (unmarked)
# λ(t) = μ + Σ_{t_i < t} K / (c + t − t_i)^p
# with K c^{1−p} / (p−1) < 1
# ============================================================

def nll_hawkes_power(theta, t):
    """Negative log-likelihood for a power-law Hawkes process."""
    mu, K_param, c, p = theta
    if mu <= 0 or K_param <= 0 or c <= 0 or p <= 1:
        return np.inf
    if K_param * c ** (1.0 - p) / (p - 1.0) >= 1.0:
        return np.inf

    T = t[-1]
    diff = t[:, None] - t[None, :]
    mask = diff > 0.0

    with np.errstate(invalid="ignore"):
        G = np.where(mask, K_param / (c + diff) ** p, 0.0)
    lam = mu + G.sum(axis=1)

    if not np.isfinite(lam).all() or np.any(lam <= 1e-12):
        return np.inf

    loglik_events = np.log(lam).sum()
    term = (1.0 / c ** (p - 1)) - 1.0 / (c + (T - t)) ** (p - 1)
    integ = mu * T + (K_param / (p - 1)) * np.sum(term)

    loglik = loglik_events - integ
    return -loglik if np.isfinite(loglik) else np.inf


# ============================================================
# Exponential marked Hawkes process
# λ(t) = μ + Σ_{t_i < t} exp(γ(M_i − M0)) α exp(−β(t − t_i))
# ============================================================

def nll_hawkes_exp_marked(theta, t, M, M0):
    """Negative log-likelihood for an exponential marked Hawkes process."""
    mu, alpha, beta, gamma = theta
    if mu <= 0 or alpha <= 0 or beta <= 0:
        return np.inf

    T = t[-1]
    w = np.exp(gamma * (M - M0))

    diff = t[:, None] - t[None, :]
    mask = diff > 0.0

    arg = np.where(mask, -beta * diff, 0.0)
    arg = np.clip(arg, -700.0, 700.0)
    K = np.where(mask, alpha * np.exp(arg), 0.0)
    lam = mu + (K * w[None, :]).sum(axis=1)

    if not np.isfinite(lam).all() or np.any(lam <= 1e-12):
        return np.inf

    loglik_events = np.log(lam).sum()
    arg_integ = np.clip(-beta * (T - t), -700.0, 700.0)
    integ = mu * T + (alpha / beta) * np.sum(w * (1.0 - np.exp(arg_integ)))

    loglik = loglik_events - integ
    return -loglik if np.isfinite(loglik) else np.inf


# ============================================================
# Power-law marked Hawkes process
# λ(t) = μ + Σ_{t_i < t} exp(γ(M_i − M0)) K / (c + t − t_i)^p
# ============================================================

def nll_hawkes_power_marked(theta, t, M, M0):
    """Negative log-likelihood for a power-law marked Hawkes process."""
    mu, K_param, c, p, gamma = theta
    if mu <= 0 or K_param <= 0 or c <= 0 or p <= 1:
        return np.inf

    T = t[-1]
    w = np.exp(gamma * (M - M0))

    diff = t[:, None] - t[None, :]
    mask = diff > 0.0

    with np.errstate(invalid="ignore"):
        G = np.where(mask, K_param / (c + diff) ** p, 0.0)
    lam = mu + (G * w[None, :]).sum(axis=1)

    if not np.isfinite(lam).all() or np.any(lam <= 1e-12):
        return np.inf

    loglik_events = np.log(lam).sum()
    term = (1.0 / c ** (p - 1)) - 1.0 / (c + (T - t)) ** (p - 1)
    integ = mu * T + (K_param / (p - 1)) * np.sum(w * term)

    loglik = loglik_events - integ
    return -loglik if np.isfinite(loglik) else np.inf
