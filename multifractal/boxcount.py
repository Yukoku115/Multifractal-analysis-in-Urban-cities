# multifractal/boxcount.py
"""
Boundary strategies and box-counting loop.

Three boundary strategies:
    make_full_square()           → grid covers [0,1]² — correct for square OSM downloads
    make_circular_mask(radius)   → grid covers [0,1]², cells outside circle zeroed
    make_inscribed_square()      → grid covers largest square inside circle
                                   *** Only valid when input data was downloaded with a
                                       circular clip AND re-normalised with equal x/y scale.
                                       Do NOT use with square-bounded OSM downloads — it
                                       discards ~50% of real data and collapses R² for q<0. ***

Usage:
    boundary = make_full_square()
    matrices = run_boxcount(x_norm, y_norm, boundary, GRID_SIZES, q_values)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Boundary strategies
# Each factory returns a dict with:
#   'histogram' : callable(x, y, bins) → H  (2D numpy array, already masked)
#   'grid_lo'   : float  (lower edge of histogram range)
#   'grid_hi'   : float  (upper edge of histogram range)
#   'label'     : str    (for plot annotations and log)
#   'kind'      : str    (identifier)
#   'radius'    : float | None
# ─────────────────────────────────────────────────────────────────────────────

def make_full_square() -> dict:
    """
    Plain square grid on a square boundary — identical to city_multifractal_v5.py.

    np.histogram2d with range=[[0,1],[0,1]] and no masking whatsoever.
    This is the correct default for osmnx square-bbox downloads where data
    already fills the full [0,1]² domain after min-max normalisation.
    """
    def histogram(x_norm, y_norm, bins):
        H, _, _ = np.histogram2d(x_norm, y_norm, bins=bins,
                                 range=[[0, 1], [0, 1]])
        return H

    return dict(
        histogram=histogram,
        grid_lo=0.0,
        grid_hi=1.0,
        label="Full square [0, 1]²",
        kind="full_square",
        radius=None,
    )


def make_circular_mask(circ_mask_radius: float = 0.45) -> dict:
    """
    Full [0,1]² grid with cells outside the circle zeroed.
    Circle sits at (0.5, 0.5) with radius circ_mask_radius.

    Note: Shares the same mismatch issue as inscribed_square when the
    underlying download is a square bbox — corner data is silently
    dropped.  Use only when the raw data genuinely has circular support
    (e.g. downloaded with ox.graph_from_point and then hard-clipped to a
    disk before normalisation).
    """
    def histogram(x_norm, y_norm, bins):
        H, _, _ = np.histogram2d(
            x_norm, y_norm, bins=bins,
            range=[[0.0, 1.0], [0.0, 1.0]]
        )
        cx = (np.arange(bins) + 0.5) / bins
        cy = (np.arange(bins) + 0.5) / bins
        xx, yy = np.meshgrid(cx, cy, indexing='ij')
        outside = np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2) > circ_mask_radius
        H[outside] = 0
        return H

    return dict(
        histogram=histogram,
        grid_lo=0.0,
        grid_hi=1.0,
        label=f"Circular mask  r={circ_mask_radius}",
        kind="circular",
        radius=circ_mask_radius,
    )


def make_inscribed_square() -> dict:
    """
    Histogram range clipped to the largest square inscribed inside the circle.
    Every cell is guaranteed to lie within the circular support.

    *** PRE-CONDITION ***
        This strategy is only valid when:
          1. The raw OSM download used a TRUE circular clip (not a bbox), AND
          2. x_norm / y_norm were re-normalised with EQUAL x/y scale
             (i.e. divided by the same range, not independent min-max),
             so that the circular support maps to a true disk in [0,1]².

        If you use this with a standard square bbox download the inscribed
        square covers only ~50 % of the domain, silently discards real
        intersections in the corners, and almost all q-values will fail
        the R² filter (typically ≤ 1 survivor near q = 0).
    """
    half   = 0.5 / np.sqrt(2)
    lo, hi = 0.5 - half, 0.5 + half   # ≈ 0.146, 0.854

    def histogram(x_norm, y_norm, bins):
        H, _, _ = np.histogram2d(
            x_norm, y_norm, bins=bins,
            range=[[lo, hi], [lo, hi]]
        )
        return H

    return dict(
        histogram=histogram,
        grid_lo=lo,
        grid_hi=hi,
        label=f"Inscribed square  [{lo:.3f}, {hi:.3f}]",
        kind="inscribed",
        radius=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Box-counting loop
# ─────────────────────────────────────────────────────────────────────────────

def run_boxcount(x_norm, y_norm, boundary: dict,
                 grid_sizes: np.ndarray,
                 q_values:   np.ndarray,
                 min_count:  int = 3) -> dict:
    """
    Run the Chhabra-Jensen box-counting loop.

    Parameters
    ----------
    x_norm, y_norm : normalised coordinates in [0,1]²
    boundary       : strategy dict from make_full_square / make_circular_mask
                     / make_inscribed_square
    grid_sizes     : 1-D array of bin counts
    q_values       : 1-D array of moment orders
    min_count      : minimum intersections per cell to be included (default 3).
                     Keeping single-point cells (min_count=1) introduces
                     artifactual noise in Z_q at fine scales, which degrades
                     R² especially for negative q.  min_count=3 is the
                     recommended minimum; raise to 5 for very sparse networks.

    Returns
    -------
    matrices : dict with keys
        tau_q_matrix  (n_q, n_g)
        num_alpha     (n_q, n_g)
        num_f         (n_q, n_g)
        n_boxes       (n_g,)
        log_eps       (n_g,)
        q_values      echo
        grid_sizes    echo
    """
    histogram = boundary['histogram']
    n_q = len(q_values)
    n_g = len(grid_sizes)

    epsilons     = 1.0 / grid_sizes
    log_eps      = np.log(epsilons)
    num_alpha    = np.zeros((n_q, n_g))
    num_f        = np.zeros((n_q, n_g))
    tau_q_matrix = np.full((n_q, n_g), np.nan)
    n_boxes      = np.zeros(n_g)

    print(f"[boxcount] Running {n_g} grid sizes × {n_q} q-values "
          f"({boundary['label']}) ...")
    print(f"[boxcount] min_count per cell = {min_count}")

    for i, bins in enumerate(grid_sizes):
        H  = histogram(x_norm, y_norm, bins)

        # FIX: use min_count (default 3) not H > 0.
        # Single-point cells (count=1 or 2) create artifactual scaling noise
        # in Z_q at fine grid scales.  This noise is amplified by negative q
        # (which up-weights sparse cells) and causes R² to collapse for q < 0,
        # leaving very few valid q-values after filtering.
        nz = H[H >= min_count]

        n_boxes[i] = len(nz)
        if len(nz) == 0:
            continue
        P_i = nz / nz.sum()

        for j, q in enumerate(q_values):
            if np.isclose(q, 0.0):
                Z_q  = float(len(P_i))
                mu_i = np.ones(len(P_i)) / len(P_i)
            elif np.isclose(q, 1.0):
                mu_i = P_i
                Z_q  = 1.0
            else:
                P_q = P_i ** q
                Z_q = P_q.sum()
                if Z_q <= 0:
                    continue
                mu_i = P_q / Z_q

            tau_q_matrix[j, i] = np.log(max(Z_q, 1e-300))
            log_P  = np.where(P_i  > 0, np.log(P_i),  0.0)
            log_mu = np.where(mu_i > 0, np.log(mu_i), 0.0)
            num_alpha[j, i] = np.dot(mu_i, log_P)
            num_f[j, i]     = np.dot(mu_i, log_mu)

    return dict(
        tau_q_matrix=tau_q_matrix,
        num_alpha=num_alpha,
        num_f=num_f,
        n_boxes=n_boxes,
        log_eps=log_eps,
        q_values=q_values,
        grid_sizes=grid_sizes,
    )