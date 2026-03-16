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
    Normalises to [0,1]² using independent min-max on each axis —
    identical to city_multifractal_v5.py.

    NO circular crop is applied.  osmnx downloads a square bounding box,
    so after independent min-max the data genuinely fills all four corners
    of [0,1]².  Cropping to a disk before normalisation leaves the corners
    empty, which makes full_square count ~22% dead air and collapses R²
    for negative q (only ~1 q-value survives the filter).

    Returns
    -------
    x_norm, y_norm : np.ndarray
        Normalised coordinates in [0,1]².
    meta : dict
        city, radius, slug, out_dir, csv_path, raw_n, retained_n,
        network_type.
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

    # ── Normalise to [0,1]² — independent min-max, no crop ───────────────────
    # This matches v5 exactly:
    #   x_norm = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    #   y_norm = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
    # Data fills the full square; full_square boundary is then correct.
    x_norm = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y_norm = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    print(f"[data] Normalised to [0,1]² (no crop). Retained: {raw_n:,} (100%)")

    meta = dict(
        city=city_address,
        radius=radius_meters,
        network_type=network_type,
        slug=city_slug,
        out_dir=out_dir,
        csv_path=csv_path,
        raw_n=raw_n,
        retained_n=raw_n,   # all points kept
    )
    return x_norm, y_norm, meta


def load_continuous_network(city_address: str, radius_meters: int, network_type: str = "all", resolution_meters: float = 2.0):
    """
    Load or download street edges and interpolate points every `resolution_meters`
    to create a continuous 1D length measure.

    This replaces the discrete node (intersection) point pattern with a dense
    sampling of the actual street lengths. By feeding these dense points into
    the standard box-counting algorithm, the count `N` inside a box is strictly
    proportional to the physical street length `L` inside that box.

    Returns
    -------
    x_norm, y_norm : np.ndarray
        Normalised coordinates in [0,1]².
    meta : dict
    """
    city_slug = re.sub(r'[^a-zA-Z0-9]+', '_', city_address).strip('_').lower()
    out_dir   = f"output_{city_slug}_{radius_meters // 1000}km"
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, f"{city_slug}_continuous_{resolution_meters}m.csv")

    if os.path.exists(csv_path):
        print(f"[data] Found cached continuous CSV — {csv_path}")
        df = pd.read_csv(csv_path)
        x_raw, y_raw = df['x'].values, df['y'].values
    else:
        print(f"[data] Downloading {radius_meters}m edges for '{city_address}'...")
        try:
            import osmnx as ox
            import geopandas as gpd
            from geopy.geocoders import Nominatim
        except ImportError:
            raise ImportError("pip install osmnx geopy geopandas")

        geolocator = Nominatim(user_agent="multifractal_analysis_continuous")
        location   = geolocator.geocode(city_address)
        
        # Download graph and get edges (lines)
        G = ox.graph_from_point(
            (location.latitude, location.longitude),
            dist=radius_meters, network_type=network_type, simplify=True
        )
        _, edges = ox.graph_to_gdfs(G)
        
        # Project to local UTM to work in meters for segmentize
        # (ox.project_gdf was removed in osmnx 2.x; use geopandas native projection)
        utm_crs    = edges.estimate_utm_crs()
        edges_proj = edges.to_crs(utm_crs)
        
        print(f"    → Discretising {len(edges_proj):,} street segments at {resolution_meters}m resolution...")
        # segmentize(max_segment_length) adds vertices so no gap > resolution_meters.
        dense_geom = edges_proj.geometry.segmentize(resolution_meters)
        
        # Extract all vertices from all dense linestrings
        # Stay in projected (metric) coordinates — we normalise to [0,1]² immediately,
        # so there's no need to reproject millions of points back to lat/lon.
        xs, ys = [], []
        for line in dense_geom:
            if line.geom_type == 'LineString':
                cc = line.coords
                xs.extend(c[0] for c in cc)
                ys.extend(c[1] for c in cc)
            elif line.geom_type == 'MultiLineString':
                for part in line.geoms:
                    cc = part.coords
                    xs.extend(c[0] for c in cc)
                    ys.extend(c[1] for c in cc)

        x_raw = np.array(xs, dtype=np.float64)
        y_raw = np.array(ys, dtype=np.float64)
        
        pd.DataFrame({'x': x_raw, 'y': y_raw}).to_csv(csv_path, index=False)
        print(f"    → Generated {len(x_raw):,} dense length points.")

    raw_n = len(x_raw)
    print(f"[data] Continuous length points: {raw_n:,}")

    x_norm = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y_norm = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    print(f"[data] Normalised continuous points to [0,1]². (100% retained)")

    meta = dict(
        city=city_address,
        radius=radius_meters,
        network_type=network_type,
        slug=city_slug,
        out_dir=out_dir,
        csv_path=csv_path,
        raw_n=raw_n,
        retained_n=raw_n,
        measure_type="continuous"
    )
    return x_norm, y_norm, meta