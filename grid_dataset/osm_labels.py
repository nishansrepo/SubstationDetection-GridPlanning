"""
OpenStreetMap substation label acquisition.

Supports two acquisition modes matching the pipeline strategies:
    - County-based (curated): area query per county name.
    - Bbox-based (randomized): bounding-box query per random region.

Handles polygon vs. point geometries, deduplication, and substation
type classification for per-type budget allocation.
"""

import logging
import time
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from .config import PipelineConfig
from .regions import CountySpec, RandomRegion

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ------------------------------------------------------------------
# Overpass query builders
# ------------------------------------------------------------------

def _build_county_query(county: CountySpec) -> str:
    """Overpass QL query for substations within a county by name."""
    return (
        f"[out:json][timeout:180];\n"
        f'area["name"="{county.osm_area_name}"]'
        f'["admin_level"="{county.admin_level}"]->.county;\n'
        f'(\n  nwr["power"="substation"](area.county);\n);\n'
        f"out body;\n>;\nout skel qt;"
    )


def _build_bbox_query(bbox: tuple[float, float, float, float]) -> str:
    """Overpass QL query for substations within a bounding box.

    bbox is (south, west, north, east) per Overpass convention.
    """
    s, w, n, e = bbox
    return (
        f"[out:json][timeout:180];\n"
        f'(\n  nwr["power"="substation"]({s},{w},{n},{e});\n);\n'
        f"out body;\n>;\nout skel qt;"
    )


def _query_overpass(query: str, max_retries: int = 4) -> dict:
    """Send query with exponential backoff."""
    import requests

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                OVERPASS_URL, data={"data": query}, timeout=300,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == max_retries:
                    resp.raise_for_status()
                wait = 2 ** attempt * 10
                logger.warning("Overpass %d, retry in %ds", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt * 10)

    raise RuntimeError("Overpass query failed after all retries")


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def _parse_overpass_to_gdf(data: dict, buffer_m: float) -> gpd.GeoDataFrame:
    """Convert raw Overpass JSON to a GeoDataFrame of substation footprints.

    Polygon ways use their actual geometry; point nodes are buffered.
    The 'substation_type' column holds the OSM substation=* tag value
    (e.g., 'transmission', 'distribution', '' if untagged).
    """
    node_coords: dict[int, tuple[float, float]] = {}
    for el in data.get("elements", []):
        if el["type"] == "node" and "lon" in el and "lat" in el:
            node_coords[el["id"]] = (el["lon"], el["lat"])

    records = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if tags.get("power") != "substation":
            continue

        geom = None
        geom_source = None

        if el["type"] == "node" and "lon" in el:
            geom = Point(el["lon"], el["lat"])
            geom_source = "point"

        elif el["type"] == "way" and "nodes" in el:
            coords = [node_coords[nid] for nid in el["nodes"] if nid in node_coords]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geom = Polygon(coords)
                geom_source = "polygon"
            elif len(coords) >= 2:
                centroid = np.mean(coords, axis=0)
                geom = Point(centroid[0], centroid[1])
                geom_source = "point"

        if geom is None:
            continue

        records.append({
            "geometry": geom,
            "osm_id": el["id"],
            "osm_type": el["type"],
            "geom_source": geom_source,
            "name": tags.get("name", ""),
            "operator": tags.get("operator", ""),
            "voltage": tags.get("voltage", ""),
            "substation_type": tags.get("substation", ""),
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=["geometry", "osm_id", "osm_type", "geom_source",
                      "name", "operator", "voltage", "substation_type"],
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Buffer point features
    point_mask = gdf["geom_source"] == "point"
    if point_mask.any():
        projected = gdf.loc[point_mask].to_crs(epsg=3857)
        projected["geometry"] = projected.geometry.buffer(buffer_m)
        gdf.loc[point_mask, "geometry"] = projected.to_crs(epsg=4326).geometry

    return gdf


def _deduplicate(gdf: gpd.GeoDataFrame, tolerance_m: float = 100.0) -> gpd.GeoDataFrame:
    """Remove near-duplicate substations, preferring polygons over points."""
    if gdf.empty or len(gdf) < 2:
        return gdf

    projected = gdf.to_crs(epsg=3857).copy()
    projected["centroid"] = projected.geometry.centroid
    keep = np.ones(len(projected), dtype=bool)

    for i in range(len(projected)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(projected)):
            if not keep[j]:
                continue
            dist = projected.iloc[i]["centroid"].distance(projected.iloc[j]["centroid"])
            if dist < tolerance_m:
                if (gdf.iloc[j]["geom_source"] == "polygon"
                        and gdf.iloc[i]["geom_source"] == "point"):
                    keep[i] = False
                else:
                    keep[j] = False

    removed = (~keep).sum()
    if removed:
        logger.info("Deduplication removed %d features", removed)
    return gdf.loc[keep].reset_index(drop=True)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def fetch_substations_for_county(
    county: CountySpec,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Fetch substations for a single county."""
    logger.info("Fetching substations for %s...", county.name)
    query = _build_county_query(county)
    raw = _query_overpass(query)
    gdf = _parse_overpass_to_gdf(raw, buffer_m)
    gdf = _deduplicate(gdf)
    gdf["county_geoid"] = county.geoid
    gdf["county_name"] = county.name
    gdf["region"] = county.region
    gdf["source_type"] = "curated"
    logger.info("  %s: %d substations", county.name, len(gdf))
    return gdf


def fetch_substations_for_bbox(
    region: RandomRegion,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Fetch substations for a random bounding-box region."""
    logger.info("Fetching substations for random region %d...", region.region_id)
    query = _build_bbox_query(region.bbox)
    raw = _query_overpass(query)
    gdf = _parse_overpass_to_gdf(raw, buffer_m)
    gdf = _deduplicate(gdf)
    gdf["county_geoid"] = f"rand_{region.region_id:04d}"
    gdf["county_name"] = f"random_region_{region.region_id:04d}"
    gdf["region"] = "randomized"
    gdf["source_type"] = "randomized"
    logger.info("  Region %d: %d substations", region.region_id, len(gdf))
    return gdf


def fetch_all_substations(
    counties: Optional[list[CountySpec]] = None,
    random_regions: Optional[list[RandomRegion]] = None,
    buffer_m: float = 75.0,
    delay_s: float = 5.0,
) -> gpd.GeoDataFrame:
    """Fetch substations from all specified sources.

    Accepts counties, random regions, or both. Results are concatenated.
    """
    frames = []

    if counties:
        for i, county in enumerate(counties):
            if i > 0:
                time.sleep(delay_s)
            try:
                gdf = fetch_substations_for_county(county, buffer_m)
                if not gdf.empty:
                    frames.append(gdf)
            except Exception:
                logger.exception("Failed: %s", county.name)

    if random_regions:
        for i, region in enumerate(random_regions):
            if i > 0 or frames:
                time.sleep(delay_s)
            try:
                gdf = fetch_substations_for_bbox(region, buffer_m)
                if not gdf.empty:
                    frames.append(gdf)
            except Exception:
                logger.exception("Failed: random region %d", region.region_id)

    if not frames:
        raise RuntimeError("No substations fetched from any source")

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs="EPSG:4326",
    )
    logger.info("Total substations: %d", len(combined))
    return combined
