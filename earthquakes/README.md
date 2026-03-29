# Hawkes Process Models for Earthquake Occurrences

A short empirical comparison of point process models on USGS earthquake data.

## Models

| Model | Parameters |
|---|---|
| Homogeneous Poisson | λ |
| Exponential Hawkes | μ, α, β |
| Power-law Hawkes | μ, K, c, p |
| Marked exponential Hawkes | μ, α, β, γ |
| Marked power-law Hawkes | μ, K, c, p, γ |

## Project structure

```
hawkes_earthquakes/
├── data/
│   └── earthquake.csv        # USGS catalogue
├── figures/                  # generated figures (git-ignored)
├── src/
│   ├── models.py             # negative log-likelihood functions
│   ├── estimation.py         # MLE (multi-start L-BFGS-B)
│   ├── diagnostics.py        # time-rescaling, KS and chi-squared tests
│   ├── metrics.py            # AIC / BIC
│   └── plotting.py           # all matplotlib figures
├── main.py                   # entry point
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Figures are saved to `figures/`.

## Data

Catalogue retrieved from the [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/earthquakes/search/).
Events: USA, 01/01/2025 – 04/05/2026.
