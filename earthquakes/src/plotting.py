"""
Publication-quality figures for the Hawkes earthquake study.

All matplotlib logic lives here. Functions take pre-computed arrays
(event times, compensators, fitted parameters) and save figures to disk.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Style
# ============================================================

COLOURS = {
    "poisson":   "#5E6472",   # slate
    "exp":       "#274C9B",   # royal blue
    "exp_m":     "#7A1F3D",   # burgundy
    "pow":       "#1E4D3A",   # fir green
    "pow_m":     "#C69214",   # mustard
    "reference": "#C69214",   # mustard  (Exp(1) benchmark)
    "counting":  "#274C9B",
    "poisson_fit": "#C69214",
    "grid":      "#D9D9D9",
}


def set_publication_style():
    plt.rcParams.update({
        "figure.figsize":   (7.0, 4.0),
        "figure.dpi":       160,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
        "axes.edgecolor":   "black",
        "axes.linewidth":   1.0,
        "axes.grid":        True,
        "grid.color":       COLOURS["grid"],
        "grid.linestyle":   "--",
        "grid.linewidth":   0.6,
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
    })


def _finish(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False)


# ============================================================
# Figure 1 — Raw inter-event times vs Poisson CDF
# ============================================================

def plot_ecdf_raw(t, out_path):
    """ECDF of raw inter-event times against Exp(λ̂) benchmark."""
    dt = np.diff(t)
    lam_hat = len(t) / (t[-1] - t[0])
    x = np.sort(dt)
    ecdf = np.arange(1, len(x) + 1) / len(x)

    x_grid = np.linspace(0, np.quantile(dt, 0.99), 500)
    cdf_pois = 1 - np.exp(-lam_hat * x_grid)

    fig, ax = plt.subplots()
    ax.plot(x_grid, cdf_pois, "--", color=COLOURS["reference"],
            label=r"Poisson $\mathrm{Exp}(\hat\lambda)$ CDF")
    ax.plot(x, ecdf, color=COLOURS["poisson"],
            label=r"Empirical CDF of $\Delta t$")
    _finish(ax, "ECDF of inter-event times vs Poisson theory",
            "Inter-event time (days)", "CDF")
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Figure 2 — Counting process N(t) vs Poisson trend
# ============================================================

def plot_counting_process(t, out_path):
    """Empirical counting process against the fitted Poisson straight line."""
    lam_hat = len(t) / t[-1]
    N = np.arange(1, len(t) + 1)
    t_line = np.linspace(0, t[-1], 500)

    fig, ax = plt.subplots()
    ax.step(t, N, where="post", color=COLOURS["counting"],
            linewidth=1.6, label=r"$N(t)$")
    ax.plot(t_line, lam_hat * t_line, "--", color=COLOURS["poisson_fit"],
            linewidth=1.6, label=r"$\hat\lambda\,(t - t_0)$")
    _finish(ax, "Counting process vs homogeneous Poisson fit",
            "Time (days)", "Cumulative number of events")
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Figure 3 — KS-style ECDFs of time-rescaled residuals
# ============================================================

def plot_ecdf_residuals(compensators, out_path):
    """
    Overlay ECDFs of rescaled inter-arrivals for three main models
    against the Exp(1) reference.

    Parameters
    ----------
    compensators : dict with keys 'poisson', 'exp', 'exp_marked'
    """
    model_styles = [
        ("poisson",    COLOURS["poisson"], "Poisson"),
        ("exp",        COLOURS["exp"],     "Exponential Hawkes"),
        ("exp_marked", COLOURS["exp_m"],   "Marked exponential Hawkes"),
    ]

    all_vals = np.concatenate([np.diff(compensators[k]) for k, _, _ in model_styles])
    xmax = np.quantile(all_vals, 0.99)
    xgrid = np.linspace(0, xmax, 600)

    fig, ax = plt.subplots()
    ax.plot(xgrid, 1.0 - np.exp(-xgrid), "--",
            color=COLOURS["reference"], lw=2.2, label="Exp(1) CDF")

    for key, colour, label in model_styles:
        xs = np.sort(np.diff(compensators[key]))
        ecdf = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ecdf, color=colour, lw=1.8, label=label)

    _finish(ax, "KS-style ECDFs of time-rescaled residuals",
            "Rescaled inter-arrival", "CDF")
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Figure 4 — Residual densities vs Exp(1)
# ============================================================

def plot_density_residuals(compensators, out_path, bins=35):
    """
    Histogram + per-model density curves vs Exp(1).

    Parameters
    ----------
    compensators : dict with keys 'poisson', 'exp', 'exp_marked'
    """
    model_styles = [
        ("poisson",    COLOURS["poisson"], "Poisson"),
        ("exp",        COLOURS["exp"],     "Exponential Hawkes"),
        ("exp_marked", COLOURS["exp_m"],   "Marked exponential Hawkes"),
    ]

    all_vals = np.concatenate([np.diff(compensators[k]) for k, _, _ in model_styles])
    xmax = np.quantile(all_vals, 0.99)
    xgrid = np.linspace(0, xmax, 600)

    fig, ax = plt.subplots()
    ax.hist(all_vals, bins=bins, range=(0, xmax), density=True,
            color="#CCCCCC", alpha=0.6, edgecolor="white",
            linewidth=0.4, label="Pooled empirical density")
    ax.plot(xgrid, np.exp(-xgrid), "--",
            color=COLOURS["reference"], lw=2.2, label="Exp(1) density")

    for key, colour, label in model_styles:
        samples = np.diff(compensators[key])
        hist_y, hist_x = np.histogram(samples, bins=bins,
                                      range=(0, xmax), density=True)
        centres = 0.5 * (hist_x[1:] + hist_x[:-1])
        ax.plot(centres, hist_y, color=colour, lw=1.8, label=label)

    _finish(ax, "Residual densities vs Exp(1)",
            "Rescaled inter-arrival", "Density")
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Convenience wrapper
# ============================================================

def save_all_figures(t, compensators, fig_dir="figures"):
    """Generate and save all four figures."""
    os.makedirs(fig_dir, exist_ok=True)
    set_publication_style()

    plot_ecdf_raw(t,          os.path.join(fig_dir, "ecdf_dt_vs_poisson.png"))
    plot_counting_process(t,  os.path.join(fig_dir, "counting_process_vs_poisson.png"))
    plot_ecdf_residuals(compensators,
                        os.path.join(fig_dir, "ks_ecdf_residuals.png"))
    plot_density_residuals(compensators,
                           os.path.join(fig_dir, "density_residuals.png"))
    print(f"Figures saved to {fig_dir}/")
