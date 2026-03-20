# Multifractal Analysis of Urban Street Networks

Compute the multifractal (generalised) dimensions of any city's street network using the **Chhabra–Jensen** box-counting method on data pulled from [OpenStreetMap](https://www.openstreetmap.org/) via [OSMnx](https://osmnx.readthedocs.io/).

## Features

- **Continuous street-length measure** — discretises every street edge at 2 m resolution so that box counts are proportional to physical road length, not just intersection counts.
- **Discrete intersection mode** — traditional node-based point pattern, selectable via config.
- **Boundary strategies** — full-square or circular mask, with an inscribed-square option for specialised use cases.
- **Quality filters** — dual R² thresholds for positive/negative q, optional α-monotonicity enforcement, and automated sanity checks (D_q ≤ 2, hierarchy D₀ ≥ D₁ ≥ D₂).
- **8 output panels** — scaling diagnostics, density map, dimension bars, singularity spectrum f(α), mass exponents τ(q), f(α) vs q, generalised dimensions D(q), and a 2D energy landscape α-map.
- **Historical comparison** — overlay singularity spectra from different years using local `.osm` files or OpenHistoricalMap.

## Quick Start

### 1. Install dependencies

```bash
pip install osmnx geopy geopandas pandas numpy scipy matplotlib
```

### 2. Run the analysis

```bash
python main.py
```

On first run the script downloads the street network for the configured city and caches it as a CSV. Subsequent runs load from cache.

### 3. Configure

Edit the **CONFIG** block at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `CITY_ADDRESS` | `"Manchester, UK"` | Any address [Nominatim](https://nominatim.openstreetmap.org/) can geocode |
| `RADIUS_METERS` | `7000` | Radius of the urban core to analyse |
| `Q_MIN`, `Q_MAX` | `-6`, `8` | Range of moment orders q |
| `N_Q` | `120` | Number of q steps |
| `GRID_SIZES` | `[16 … 128]` | Box-counting grid resolutions |
| `MEASURE_TYPE` | `"continuous"` | `"continuous"` or `"points"` |
| `BOUNDARY_KIND` | `"circular"` | `"circular"` or `"full"` |

## Project Structure

```
├── main.py                  # Entry point & configuration
├── multifractal/
│   ├── __init__.py
│   ├── data.py              # Data download, caching, normalisation
│   ├── boxcount.py          # Boundary strategies & Chhabra–Jensen loop
│   ├── dimensions.py        # Regression, quality filters, D₀/D₁/D₂
│   └── panels.py            # All plotting functions (8 panels)
├── historical_data/         # Place local .osm files here for historical runs
└── output_<city>_<radius>/  # Generated outputs (gitignored)
```

## Output Panels

| Panel | File suffix | Description |
|---|---|---|
| Scaling diagnostic | `_scaling_diagnostic.png` | log(Z_q) linearity, R² heatmap, box-counting slope |
| Density map | `_A_density.png` | 2D intersection/street density with boundary overlay |
| Dimension bars | `_B_dimensions.png` | Bar chart of D₀, D₁, D₂ |
| Singularity spectrum | `_C_spectrum.png` | f(α) vs α parabola |
| Mass exponents | `_D_tau.png` | τ(q) curve with monofractal reference |
| f(α) vs q | `_E_fq.png` | Subset dimension as a function of moment order |
| Generalised dimensions | `_F_Dq.png` | D(q) vs q with D₀/D₁/D₂ markers |
| Energy landscape | `_G_2D_energy_map.png` | 2D α-map showing local singularity strength |

## References

- Chhabra, A. & Jensen, R. V. (1989). *Direct determination of the f(α) singularity spectrum*. Physical Review Letters, 62(12), 1327.
- Chen, Y. & Wang, J. (2013). *Multifractal characterization of urban form and growth*. Environment and Planning B, 40(5), 884–904.
