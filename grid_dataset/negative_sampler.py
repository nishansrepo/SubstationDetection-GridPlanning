"""
Negative sample location generation.

Generates coordinates for patches that do NOT contain substations.
Works with both curated (county-based bounds) and randomized (region-based
bounds) strategies.
"""

import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _get_county_bounds(osm_area_name: str):
    """Fetch bounding box via Nominatim. Returns (w, s, e, n) or None."""
    import requests
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": osm_area_name + ", United States",
                    "format": "json", "limit": 1},
            headers={"User-Agent": "grid-dataset-builder/0.2"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            bb = data[0]["boundingbox"]
            return (float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1]))
    except Exception:
        pass
    return None


def generate_negative_locations(
    bounds: tuple[float, float, float, float],
    substations: gpd.GeoDataFrame,
    n_samples: int,
    config: PipelineConfig,
    rng: np.random.Generator,
    source_id: str = "",
) -> list[tuple[float, float]]:
    """Generate negative sample locations within given bounds.

    Parameters
    ----------
    bounds : tuple
        (west, south, east, north) in WGS-84.
    substations : gpd.GeoDataFrame
        All substations for distance filtering.
    n_samples : int
        Target count.
    config : PipelineConfig
        For min distance and attempt limits.
    rng : np.random.Generator
        Random state.
    source_id : str
        Label for logging.

    Returns
    -------
    list of (lon, lat) tuples.
    """
    w, s, e, n = bounds
    min_dist = config.sampling.negative_min_distance_m
    max_attempts = config.sampling.negative_max_attempts

    if not substations.empty:
        subs_proj = substations.to_crs(epsg=3857)
        centroids = subs_proj.geometry.centroid
    else:
        centroids = None

    locations = []
    total_attempts = 0

    while len(locations) < n_samples and total_attempts < n_samples * max_attempts:
        batch = min(100, (n_samples - len(locations)) * 3)
        lons = rng.uniform(w, e, size=batch)
        lats = rng.uniform(s, n, size=batch)

        for lon, lat in zip(lons, lats):
            total_attempts += 1
            if len(locations) >= n_samples:
                break

            if centroids is not None and not centroids.empty:
                from pyproj import Transformer
                t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                px, py = t.transform(lon, lat)
                if centroids.distance(Point(px, py)).min() < min_dist:
                    continue

            locations.append((lon, lat))

    if len(locations) < n_samples:
        logger.warning(
            "%s: only %d/%d negatives after %d attempts",
            source_id, len(locations), n_samples, total_attempts,
        )
    return locations
