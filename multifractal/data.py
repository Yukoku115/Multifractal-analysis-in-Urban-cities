# multifractal/data.py
"""
Handles all data acquisition and preprocessing.

    load_sipp(city, radius) → x_norm, y_norm, meta
"""

import os, re
import numpy as np
import pandas as pd


def load_sipp(city_address: str, radius_meters: int, network_type: str = "all"):
    """
    Load or download street intersection point pattern (SIPP).

    Downloads from OSMnx on first run and caches to CSV.
    Applies circular crop (r <= 0.95) and normalises to [0,1]².

    Returns
    -------
    x_norm, y_norm : np.ndarray
        Normalised coordinates of retained intersections.
    meta : dict
        city, radius, slug, out_dir, csv_path, raw_n, retained_n,
        x_circ, y_circ  (pre-normalisation, used for density ring overlay)
    """
    city_slug = re.sub(r'[^a-zA-Z0-9]+', '_', city_address).strip('_').lower()
    out_dir   = f"output_{city_slug}_{radius_meters // 1000}km"
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, f"{city_slug}_sipp.csv")

    # ── Load or download ──────────────────────────────────────────────────────
    if os.path.exists(csv_path):
        print(f"[data] Found cached CSV — {csv_path}")
        df = pd.read_csv(csv_path)
        x_raw, y_raw = df['x'].values, df['y'].values
    else:
        print(f"[data] Downloading {radius_meters}m network for '{city_address}'...")
        try:
            import osmnx as ox
            from geopy.geocoders import Nominatim
        except ImportError:
            raise ImportError("pip install osmnx geopy")

        geolocator = Nominatim(user_agent="multifractal_analysis")
        location   = geolocator.geocode(city_address)
        G = ox.graph_from_point(
            (location.latitude, location.longitude),
            dist=radius_meters, network_type=network_type, simplify=True
        )
        nodes, _ = ox.graph_to_gdfs(G)
        nodes[['x', 'y']].to_csv(csv_path, index=False)
        x_raw, y_raw = nodes['x'].values, nodes['y'].values
        print(f"    → {len(x_raw):,} intersections downloaded")

    raw_n = len(x_raw)
    print(f"[data] Raw points: {raw_n:,}")

    # ── Circular crop ─────────────────────────────────────────────────────────
    x_c = (x_raw - x_raw.mean()) / (x_raw.max() - x_raw.min()) * 2
    y_c = (y_raw - y_raw.mean()) / (y_raw.max() - y_raw.min()) * 2
    in_circle  = np.sqrt(x_c**2 + y_c**2) <= 0.95
    x_circ, y_circ = x_c[in_circle], y_c[in_circle]
    retained_n = in_circle.sum()
    print(f"[data] After crop: {retained_n:,} ({100*retained_n/raw_n:.1f}%)")

    # ── Normalise to [0,1]² ───────────────────────────────────────────────────
    x_norm = (x_circ - x_circ.min()) / (x_circ.max() - x_circ.min())
    y_norm = (y_circ - y_circ.min()) / (y_circ.max() - y_circ.min())

    meta = dict(
        city=city_address, radius=radius_meters, network_type=network_type,
        slug=city_slug, out_dir=out_dir, csv_path=csv_path,
        raw_n=raw_n, retained_n=retained_n,
        x_circ=x_circ, y_circ=y_circ,
    )
    return x_norm, y_norm, meta
