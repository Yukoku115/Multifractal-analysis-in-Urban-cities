# main.py
"""
Multifractal analysis of urban street networks.

The ONLY things you should need to change are in the CONFIG block below.
"""

import numpy as np
from multifractal.data       import load_sipp, load_continuous_network
from multifractal.boxcount   import run_boxcount, make_circular_mask, make_full_square
from multifractal.dimensions import extract_dimensions
from multifractal            import panels

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here
# ─────────────────────────────────────────────────────────────────────────────
CITY_ADDRESS  = "Manchester, UK"
RADIUS_METERS = 7000
Q_MIN, Q_MAX = -10, 15
N_Q          = 60
GRID_SIZES   = np.array([16, 22, 32, 40, 50, 64, 75, 90, 110, 128])


R2_THRESH_POS = 0.97   # strict for q >= 0
R2_THRESH_NEG = 0.80    # looser for q < 0
ENFORCE_ALPHA_MONO = False  # True → remove q-values where α reverses (Legendre requirement)


# ── Choose measure type ───────────────────────────────────────────────────────
# "points"     → original node intersections (discrete)
# "continuous" → street edges discretised at 2m (simulates continuous length)
MEASURE_TYPE = "continuous"

# ── Choose boundary condition ─────────────────────────────────────────────────
# "full" → square boundary
# "circular"  → full grid with cells outside the circle zeroed
BOUNDARY_KIND = "circular"   # ← change this line to switch
# ─────────────────────────────────────────────────────────────────────────


def main():
    print(f"\n{'='*60}")
    print(f"  City     : {CITY_ADDRESS}")
    print(f"  Radius   : {RADIUS_METERS/1000:.1f} km")
    print(f"  Boundary : {BOUNDARY_KIND}")
    print(f"{'='*60}\n")

    # 1 — Data
    if MEASURE_TYPE == "continuous":
        x_norm, y_norm, meta = load_continuous_network(CITY_ADDRESS, RADIUS_METERS, resolution_meters=2.0)
    else:
        x_norm, y_norm, meta = load_sipp(CITY_ADDRESS, RADIUS_METERS)
    out_dir = meta['out_dir']


    # 2 — Boundary strategy
    if BOUNDARY_KIND == "full":
        boundary = make_full_square()
    elif BOUNDARY_KIND == "circular":
        boundary = make_circular_mask(circ_mask_radius=0.45)
    else:
        raise ValueError(f"Unknown BOUNDARY_KIND: {BOUNDARY_KIND!r}. "
                         f"Choose 'inscribed' or 'circular'.")

    # 3 — Box-counting
    q_values = np.linspace(Q_MIN, Q_MAX, N_Q)
    # For continuous measure, points are very dense (1 per 2m), so min_count=10 ≈ 20m of street.
    # For discrete points (intersections), min_count=5 means 5 intersections.
    mc = 10 if MEASURE_TYPE == "continuous" else 5
    matrices = run_boxcount(x_norm, y_norm, boundary, GRID_SIZES, q_values, min_count=mc)

    # 4 — Dimensions
    dims = extract_dimensions(matrices,
                              r2_thresh_pos=R2_THRESH_POS,
                              r2_thresh_neg=R2_THRESH_NEG,
                              enforce_alpha_mono=ENFORCE_ALPHA_MONO)

    # 5 — Panels (comment out any you don't need)
    print("\n[panels] Generating outputs...")
    panels.diagnostic   (matrices, meta, out_dir)
    panels.density      (x_norm, y_norm, boundary, meta, out_dir)
    panels.dimension_bars(dims, meta, out_dir)
    panels.spectrum     (dims, meta, out_dir)
    panels.tau          (dims, meta, out_dir)
    panels.fq           (dims, meta, out_dir)
    panels.Dq           (dims, meta, out_dir)

    # 6 — Summary log
    _write_log(dims, meta, boundary)

    print(f"\nDone. Outputs in: {out_dir}/")


def _write_log(dims, meta, boundary):
    D0, D1, D2   = dims['D0'], dims['D1'], dims['D2']
    delta_alpha  = dims['delta_alpha']
    n_valid      = dims['n_valid']
    n_total      = dims['n_total']
    violations   = dims['violations']

    summary = f"""
{'='*65}
  MULTIFRACTAL ANALYSIS LOG — {meta['city']}
{'='*65}

[1] LOCATION & NETWORK
─────────────────────────────────────────────────────────────────
Target Address   : {meta['city']}
Radius (meters)  : {meta['radius']:,}
Network Type     : {meta['network_type']}
Measure Type     : {meta.get('measure_type', 'points (intersections)')}

[2] DATASET PROPERTIES
─────────────────────────────────────────────────────────────────
Raw Intersections: {meta['raw_n']:,}
Retained (Crop)  : {meta['retained_n']:,} ({meta['retained_n']/meta['raw_n']*100:.1f}%)

[3] TUNING PARAMETERS
─────────────────────────────────────────────────────────────────
Grid Sizes (bins): {GRID_SIZES.tolist()}
Boundary         : {boundary['label']}
Q Range          : [{Q_MIN}, {Q_MAX}]
N_Q (Steps)      : {N_Q}
R² Threshold (+q): {dims['r2_thresh_pos']}
R² Threshold (-q): {dims['r2_thresh_neg']}

[4] FILTER SUMMARY
─────────────────────────────────────────────────────────────────
Valid q-values   : {n_valid} / {n_total}
Mono violations  : {violations} removed

[5] RESULTS
─────────────────────────────────────────────────────────────────
D0 (Capacity)    : {D0:.4f}
D1 (Information) : {D1:.4f}
D2 (Correlation) : {D2:.4f}
D0 - D2 Gap      : {D0 - D2:.4f}
Delta Alpha (Δα) : {delta_alpha:.4f}

[6] INTERPRETATION FLAGS
─────────────────────────────────────────────────────────────────
Gap Status       : {'Healthy (Organic)' if (D0 - D2) > 0.1 else 'Low (Grid-like/Planned)'}
Hierarchy Status : {'Passed (D0 >= D1 >= D2)' if (D0 >= D1 >= D2) else 'FAILED'}
Degeneracy       : {'WARNING (D0 approx D1)' if abs(D0 - D1) < 0.005 else 'Clear'}
{'='*65}
"""
    print(summary)
    log_path = os.path.join(meta['out_dir'], f"{meta['slug']}_run_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(summary)


if __name__ == "__main__":
    import os
    main()
