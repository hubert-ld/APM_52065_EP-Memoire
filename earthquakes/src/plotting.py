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


# ============================================================
# Map — Earthquake locations (lon/lat)
# ============================================================

def plot_earthquake_location_map(
    df,
    out_path,
    *,
    lon_col="longitude",
    lat_col="latitude",
    mag_col="mag",
    depth_col="depth",
    title="Earthquake locations",
    extent=None,
    pad_deg=1.0,
    size_range=(12.0, 180.0),
    cmap="viridis_r",
    alpha=0.85,
    publication_style=True,
):
    """Plot earthquake epicentres on a lon/lat map.

    Points are sized by magnitude and coloured by depth.

    This function intentionally keeps dependencies minimal: if Cartopy is
    available it will add coastlines, country borders and US state boundaries;
    otherwise it falls back to a plain lon/lat scatter plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalogue with at least lon/lat/mag/depth columns.
    out_path : str
        Output path for the saved figure (e.g. "figures/map.png").
    lon_col, lat_col, mag_col, depth_col : str
        Column names in df.
    title : str
        Plot title.
    extent : tuple[float, float, float, float] | None
        (lon_min, lon_max, lat_min, lat_max). If None, inferred from data
        with padding.
    pad_deg : float
        Degrees to pad inferred extent.
    size_range : tuple[float, float]
        Minimum and maximum marker areas passed to matplotlib.scatter (points^2).
    cmap : str
        Matplotlib colormap.
    alpha : float
        Marker alpha.
    publication_style : bool
        If True, apply set_publication_style() before plotting.
    """
    if publication_style:
        set_publication_style()

    # Extract arrays (works for pandas without importing it here).
    lon = np.asarray(df[lon_col], dtype=float)
    lat = np.asarray(df[lat_col], dtype=float)
    mag = np.asarray(df[mag_col], dtype=float)
    depth = np.asarray(df[depth_col], dtype=float)

    mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(mag) & np.isfinite(depth)
    lon, lat, mag, depth = lon[mask], lat[mask], mag[mask], depth[mask]

    if lon.size == 0:
        raise ValueError("No finite lon/lat/mag/depth rows to plot.")

    # Size scaling (scatter 's' is marker area in points^2).
    mag_min, mag_max = float(np.min(mag)), float(np.max(mag))
    s_min, s_max = map(float, size_range)
    if np.isclose(mag_min, mag_max):
        sizes = np.full_like(mag, 0.5 * (s_min + s_max), dtype=float)
    else:
        sizes = s_min + (mag - mag_min) * (s_max - s_min) / (mag_max - mag_min)

    # Extent
    if extent is None:
        lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
        lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
        extent = (lon_min - pad_deg, lon_max + pad_deg, lat_min - pad_deg, lat_max + pad_deg)
    lon_min, lon_max, lat_min, lat_max = map(float, extent)

    # Try for a real map with Cartopy; otherwise, plain lon/lat axes.
    use_cartopy = False
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore

        use_cartopy = True
    except Exception:
        use_cartopy = False

    if use_cartopy:
        proj = ccrs.PlateCarree()
        fig = plt.figure()
        ax = plt.axes(projection=proj)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax.add_feature(cfeature.LAND, facecolor="#F2F2F2", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#FFFFFF", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
        # Country borders (USA / neighbouring countries)
        ax.add_feature(cfeature.BORDERS, linewidth=0.7, zorder=1)
        # US states / provinces lines (Cartopy Natural Earth admin-1)
        try:
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.45, edgecolor="#444444", zorder=1)
        except Exception:
            # Older Cartopy versions may not expose STATES; ignore gracefully.
            pass
        sc = ax.scatter(
            lon,
            lat,
            s=sizes,
            c=depth,
            cmap=cmap,
            alpha=alpha,
            linewidths=0.3,
            edgecolors="black",
            transform=proj,
            zorder=2,
        )
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", color=COLOURS["grid"])
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title(title, pad=10)
    else:
        fig, ax = plt.subplots()
        sc = ax.scatter(
            lon,
            lat,
            s=sizes,
            c=depth,
            cmap=cmap,
            alpha=alpha,
            linewidths=0.3,
            edgecolors="black",
        )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, pad=10)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.90, pad=0.02)
    cbar.set_label("Depth (km)")

    # Marker-size legend (magnitude)
    mag_lo = float(np.quantile(mag, 0.25))
    mag_md = float(np.quantile(mag, 0.50))
    mag_hi = float(np.quantile(mag, 0.75))
    legend_mags = [mag_lo, mag_md, mag_hi]
    if np.isclose(mag_min, mag_max):
        legend_sizes = [0.5 * (s_min + s_max)] * 3
    else:
        legend_sizes = [s_min + (m - mag_min) * (s_max - s_min) / (mag_max - mag_min) for m in legend_mags]

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=np.sqrt(s),
            markerfacecolor="#999999",
            markeredgecolor="black",
            alpha=alpha,
            label=f"M={m:.2f}",
        )
        for m, s in zip(legend_mags, legend_sizes)
    ]
    ax.legend(handles=handles, title="Magnitude", loc="lower left", frameon=False)

    fig.savefig(out_path)
    plt.close(fig)
