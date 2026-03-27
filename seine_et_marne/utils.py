# =======================================================================================
# 2026 - Modélisation aléatoire, statistiques et processus
# Projet : modélisation de la Seine et de la Marne
# Utils : fonctions de chargement et de prétraitement des données
# =======================================================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from typing import Optional

def load_data_dahti(filename: str):
    """
    Load the data from the given filename and return a pandas DataFrame with the following columns:
    - datetime: the date and time of the measurement, as a datetime object
    - wse: the water surface elevation in meters, as a float
    - wse_u: the uncertainty of the water surface elevation, as a float
    - elevation: the elevation of the water surface compared to the minimum altitude, as a float
    """
    import pandas as pd

    df = pd.read_csv(f"./data/{filename}", sep=";")
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    df["wse"] = df["wse"].astype(float)
    df["wse_u"] = df["wse_u"].astype(float)
    # Add a column for the elevation compared to the minimum altitude
    df["elevation"] = df["wse"] - df["wse"].min()
    return df

def load_data_eaufrance(filename: str):
    """
    Load the data from the given filename and return a pandas DataFrame with the following columns:
    - Date: the date and time of the measurement, as a datetime object
    - Hauteur: the water height in meters, as a float
    """
    import pandas as pd

    df = pd.read_csv(f"./data/{filename}", sep=";")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Hauteur"] = df["Hauteur"].astype(float)
    return df



"""
Modèle semi-paramétrique : KDE adaptatif + GPD

Modèle :
    Pour x <= u :  f(x) = w_body * kde(x) / CDF_kde(u)
    Pour x >  u :  f(x) = w_tail * gpd(x - u; xi, beta)

    avec w_body = F_empirique(u)  et  w_tail = 1 - w_body

La densité est raccordée continûment au seuil u.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from typing import Optional


# ---------------------------------------------------------------------------
# 1. KDE adaptatif (bande passante variable)
# ---------------------------------------------------------------------------

def fit_adaptive_kde(data: np.ndarray, u: float, bandwidth: Optional[float] = None) -> KernelDensity:
    """
    Fit un KDE sur les données sous le seuil u.
    La bande passante est estimée par règle de Silverman si non fournie.
    """
    body = data[data <= u].reshape(-1, 1)
    if bandwidth is None:
        # Règle de Silverman
        n, d = len(body), 1
        bandwidth = 1.06 * np.std(body) * n ** (-1 / (d + 4))
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(body)
    return kde


def kde_pdf(kde: KernelDensity, x: np.ndarray) -> np.ndarray:
    """Évalue la PDF du KDE en x."""
    return np.exp(kde.score_samples(x.reshape(-1, 1)))


def kde_cdf_at_u(kde: KernelDensity, u: float, x_min: float, n: int = 5000) -> float:
    """CDF numérique du KDE jusqu'au seuil u."""
    xs = np.linspace(x_min, u, n)
    pdf_vals = kde_pdf(kde, xs)
    return float(np.trapezoid(pdf_vals, xs))


# ---------------------------------------------------------------------------
# 2. GPD
# ---------------------------------------------------------------------------

def gpd_pdf(y: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """PDF GPD pour les excès y = x - u > 0."""
    out = np.zeros_like(y, dtype=float)
    mask = y > 0
    if beta <= 0:
        return out
    if abs(xi) < 1e-8:
        out[mask] = (1.0 / beta) * np.exp(-y[mask] / beta)
    else:
        z = 1.0 + xi * y[mask] / beta
        valid = z > 0
        tmp = np.zeros(mask.sum())
        tmp[valid] = (1.0 / beta) * z[valid] ** (-1.0 / xi - 1.0)
        out[mask] = tmp
    return out


def gpd_neg_loglik(params: np.ndarray, excesses: np.ndarray) -> float:
    xi, beta = params
    if beta <= 0 or xi < -0.5:
        return 1e12
    pdf_vals = gpd_pdf(excesses, xi, beta)
    pdf_vals = np.clip(pdf_vals, 1e-300, None)
    return -np.sum(np.log(pdf_vals))


# ---------------------------------------------------------------------------
# 3. Fit du modèle complet
# ---------------------------------------------------------------------------

def fit_mixture_kde_gpd(
    data: np.ndarray,
    u: Optional[float] = None,
    u_quantile: float = 0.85,
    bandwidth: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """
    Fit le modèle semi-paramétrique KDE + GPD.

    Paramètres
    ----------
    data        : array 1D des hauteurs (valeurs > 0)
    u           : seuil de transition. Si None, calculé via u_quantile.
    u_quantile  : quantile pour le seuil automatique (défaut 0.85)
    bandwidth   : bande passante du KDE. Si None, règle de Silverman.
    verbose     : affiche les résultats

    Retourne
    --------
    dict : kde, xi, beta, u, w_body, w_tail, bandwidth, aic, bic
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    n = len(data)

    # Seuil
    if u is None:
        u = float(np.quantile(data, u_quantile))
    if verbose:
        print(f"Seuil u = {u:.3f} m  ({(data <= u).mean():.1%} des données sous le seuil)")

    # Poids empiriques
    w_body = float((data <= u).mean())
    w_tail = 1.0 - w_body

    # --- KDE sur le corps ---
    kde = fit_adaptive_kde(data, u, bandwidth=bandwidth)
    bw = kde.bandwidth
    if verbose:
        print(f"Bande passante KDE = {bw:.4f} m")

    # --- GPD sur les excès ---
    excesses = data[data > u] - u
    xi0, beta0 = 0.2, float(np.mean(excesses))
    res = minimize(
        gpd_neg_loglik,
        [xi0, beta0],
        args=(excesses,),
        method="L-BFGS-B",
        bounds=[(-0.49, 2.0), (0.01, 20.0)],
        options={"maxiter": 2000},
    )
    xi, beta = res.x

    # --- AIC / BIC (paramètres GPD seulement, KDE est non-param) ---
    nll_gpd  = res.fun
    nll_kde  = -np.sum(kde.score_samples(data[data <= u].reshape(-1, 1)))
    nll_total = nll_kde + nll_gpd
    k = 2  # xi, beta (paramètres de la GPD)
    aic = 2 * k + 2 * nll_gpd
    bic = k * np.log(len(excesses)) + 2 * nll_gpd

    params = {
        "kde": kde,
        "xi": xi, "beta": beta, "u": u,
        "w_body": w_body, "w_tail": w_tail,
        "bandwidth": bw,
        "x_min": float(data.min()),
        "aic": aic, "bic": bic,
        "gpd_success": res.success,
    }

    if verbose:
        print("\n=== Résultats KDE + GPD ===")
        print(f"  Poids corps (KDE)   w_body = {w_body:.4f}")
        print(f"  Poids queue (GPD)   w_tail = {w_tail:.4f}")
        print(f"  xi   (forme GPD)           = {xi:.4f}  {'→ queue lourde' if xi > 0 else '→ exponentielle'}")
        print(f"  beta (échelle GPD)         = {beta:.4f}")
        print(f"  AIC (GPD) = {aic:.1f}   BIC (GPD) = {bic:.1f}")
        print(f"  Convergence GPD : {'OK' if res.success else 'WARNING'}")

    return params


# ---------------------------------------------------------------------------
# 4. Évaluation de la densité du modèle complet
# ---------------------------------------------------------------------------

def mixture_kde_gpd_pdf(x: np.ndarray, params: dict) -> np.ndarray:
    """
    Évalue la PDF du modèle complet en x.
    Corps  (x <= u) : w_body * kde(x) / cdf_kde(u)
    Queue  (x >  u) : w_tail * gpd(x - u; xi, beta)
    """
    kde     = params["kde"]
    xi      = params["xi"]
    beta    = params["beta"]
    u       = params["u"]
    w_body  = params["w_body"]
    w_tail  = params["w_tail"]
    x_min   = params["x_min"]

    # Normalisation du KDE sur [x_min, u]
    cdf_u = kde_cdf_at_u(kde, u, x_min)

    out = np.zeros_like(x, dtype=float)

    mask_body = x <= u
    if mask_body.any():
        out[mask_body] = w_body * kde_pdf(kde, x[mask_body]) / (cdf_u + 1e-12)

    mask_tail = x > u
    if mask_tail.any():
        out[mask_tail] = w_tail * gpd_pdf(x[mask_tail] - u, xi, beta)

    return out


# ---------------------------------------------------------------------------
# 6. Quantiles / périodes de retour
# ---------------------------------------------------------------------------

def mixture_kde_gpd_quantile(p: float, params: dict, n: int = 10000) -> float:
    """Quantile d'ordre p par intégration numérique de la PDF."""
    x_max = params["u"] + params["beta"] * 20
    x = np.linspace(params["x_min"] + 1e-6, x_max, n)
    pdf_vals = mixture_kde_gpd_pdf(x, params)
    cdf = np.cumsum(pdf_vals) * (x[1] - x[0])
    cdf /= cdf[-1]
    idx = np.searchsorted(cdf, p)
    return float(x[min(idx, n - 1)])


# ---------------------------------------------------------------------------
# 7. Exemple d'utilisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Données synthétiques proches du profil Seine
    rng = np.random.default_rng(42)
    n = 50000
    body = rng.lognormal(mean=0.0, sigma=0.25, size=int(0.82 * n))
    tail = stats.genpareto.rvs(c=0.3, loc=2.0, scale=1.2,
                                size=int(0.18 * n), random_state=rng)
    data = np.concatenate([body, tail])
    data = data[data > 0]

    # Fit
    params = fit_mixture_kde_gpd(data, u_quantile=0.85)

    # Périodes de retour
    print("\n=== Périodes de retour ===")
    for T in [2, 5, 10, 50, 100]:
        q = mixture_kde_gpd_quantile(1 - 1 / T, params)
        print(f"  T = {T:>3} ans  →  hauteur = {q:.2f} m")