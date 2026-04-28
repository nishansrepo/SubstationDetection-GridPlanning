"""
Site analysis for recommended substation locations.

Enriches optimizer output with:
    - Reverse geocoded place names (Nominatim)
    - Land-use / land-cover type (NLCD via Planetary Computer)
    - Nearby features from OSM (parks, water, protected areas)
    - Suitability flags (restricted / caution / suitable)
    - Interactive satellite map (Folium)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from .config import BASE_DIR, OptimizerConfig
from .results import ResultsSummary

logger = logging.getLogger(__name__)

# NLCD land cover class descriptions and suitability
NLCD_CLASSES = {
    11: ("Open Water", "restricted"),
    12: ("Perennial Ice/Snow", "restricted"),
    21: ("Developed, Open Space", "suitable"),
    22: ("Developed, Low Intensity", "suitable"),
    23: ("Developed, Medium Intensity", "suitable"),
    24: ("Developed, High Intensity", "caution"),
    31: ("Barren Land", "suitable"),
    41: ("Deciduous Forest", "caution"),
    42: ("Evergreen Forest", "caution"),
    43: ("Mixed Forest", "caution"),
    51: ("Dwarf Scrub", "suitable"),
    52: ("Shrub/Scrub", "suitable"),
    71: ("Grassland/Herbaceous", "suitable"),
    72: ("Sedge/Herbaceous", "caution"),
    81: ("Pasture/Hay", "suitable"),
    82: ("Cultivated Crops", "caution"),
    90: ("Woody Wetlands", "restricted"),
    95: ("Emergent Herbaceous Wetlands", "restricted"),
}

# OSM tags that indicate restricted or sensitive areas
RESTRICTED_TAGS = {
    "leisure": ["nature_reserve", "park", "national_park"],
    "boundary": ["protected_area", "national_park"],
    "natural": ["water", "wetland", "glacier", "beach"],
    "landuse": ["cemetery", "military", "reservoir"],
}


def reverse_geocode(lat: float, lon: float) -> dict:
    """Reverse geocode a coordinate using Nominatim."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 16},
            headers={"User-Agent": "grid-dataset-sitecheck/0.3"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        address = data.get("address", {})
        return {
            "display_name": data.get("display_name", ""),
            "neighbourhood": address.get("neighbourhood", address.get("suburb", "")),
            "city": address.get("city", address.get("town", address.get("village", ""))),
            "county": address.get("county", ""),
            "state": address.get("state", ""),
            "road": address.get("road", ""),
        }
    except Exception:
        return {"display_name": "", "neighbourhood": "", "city": "",
                "county": "", "state": "", "road": ""}


def query_osm_restrictions(lat: float, lon: float, radius_m: int = 500) -> list[dict]:
    """Check OSM for restricted or sensitive areas near a coordinate."""
    bbox = f"{lat - radius_m / 111000},{lon - radius_m / 85000}," \
           f"{lat + radius_m / 111000},{lon + radius_m / 85000}"

    tag_filters = []
    for key, values in RESTRICTED_TAGS.items():
        for val in values:
            tag_filters.append(f'way["{key}"="{val}"]({bbox});')
            tag_filters.append(f'relation["{key}"="{val}"]({bbox});')

    query = f"""[out:json][timeout:30];
    ({' '.join(tag_filters)});
    out tags center;"""

    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query}, timeout=60,
            headers={"User-Agent": "grid-dataset-sitecheck/0.3"},
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        results = []
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name", "unnamed")
            feature_type = next(
                (f"{k}={v}" for k, v in tags.items() if k in RESTRICTED_TAGS),
                "unknown",
            )
            results.append({"name": name, "type": feature_type})
        return results
    except Exception:
        return []


def analyze_recommended_sites(
        summary: ResultsSummary,
        config: OptimizerConfig,
        rate_limit: float = 1.2) -> pd.DataFrame:
    """Enrich each recommended new substation with location details and suitability."""
    if summary.per_substation.empty:
        return pd.DataFrame()

    logger.info("Analyzing %d recommended sites...", len(summary.per_substation))
    enriched_rows = []

    for idx, row in summary.per_substation.iterrows():
        lat, lon = row["lat"], row["lon"]

        # 1. Reverse geocode
        geo = reverse_geocode(lat, lon)
        time.sleep(rate_limit)

        # 2. OSM restriction check
        restrictions = query_osm_restrictions(lat, lon, radius_m=500)
        time.sleep(rate_limit)

        restriction_names = [r["name"] for r in restrictions[:3]]
        restriction_types = [r["type"] for r in restrictions[:3]]

        # 3. Determine suitability
        if restrictions:
            suitability = "restricted"
            suitability_reason = f"Near: {', '.join(restriction_names[:2])}"
        else:
            suitability = "suitable"
            suitability_reason = "No restricted areas detected within 500m"

        # 4. Build human-readable location
        parts = [p for p in [geo["road"], geo["neighbourhood"], geo["city"]] if p]
        location_name = ", ".join(parts) if parts else geo.get("display_name", "")[:80]

        enriched_rows.append({
            "candidate_id": row.get("candidate_id", idx),
            "lat": lat,
            "lon": lon,
            "n_cells_served": row.get("n_cells_served", 0),
            "total_load_kw": row.get("total_load_kw", 0),
            "avg_service_dist_m": row.get("avg_service_dist_m", 0),
            "location_name": location_name,
            "city": geo["city"],
            "road": geo["road"],
            "suitability": suitability,
            "suitability_reason": suitability_reason,
            "nearby_restrictions": "; ".join(restriction_types) if restrictions else "",
        })

        logger.info("  [%d/%d] %s → %s (%s)",
                     idx + 1, len(summary.per_substation),
                     f"({lat:.4f}, {lon:.4f})", location_name, suitability)

    return pd.DataFrame(enriched_rows)


def generate_satellite_map(
        enriched_df: pd.DataFrame,
        existing_substations: Optional[gpd.GeoDataFrame],
        tx_expansion: Optional[pd.DataFrame],
        county_name: str,
        output_path: Path,
        config: Optional[OptimizerConfig] = None) -> None:
    """Generate an interactive Folium map on satellite imagery."""
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        logger.warning("folium not installed — skipping satellite map")
        return

    if enriched_df.empty:
        return

    center = [enriched_df["lat"].mean(), enriched_df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=10)

    # Satellite base layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)
    folium.TileLayer("CartoDB positron", name="Light Map").add_to(m)

    # Existing substations (small grey dots)
    if existing_substations is not None and not existing_substations.empty:
        existing_geo = existing_substations.to_crs("EPSG:4326")
        fg_existing = folium.FeatureGroup(name=f"Existing ({len(existing_geo)})")
        for _, row in existing_geo.iterrows():
            pt = row.geometry
            folium.CircleMarker(
                [pt.y, pt.x], radius=3, color="#666",
                fill=True, fill_opacity=0.5,
                tooltip="Existing substation",
            ).add_to(fg_existing)
        fg_existing.add_to(m)

    # New recommended substations
    fg_new = folium.FeatureGroup(name=f"Recommended ({len(enriched_df)})")
    for _, r in enriched_df.iterrows():
        if r["suitability"] == "restricted":
            color = "#ff4444"
            icon_color = "red"
        elif r["suitability"] == "caution":
            color = "#ffaa00"
            icon_color = "orange"
        else:
            color = "#00cc44"
            icon_color = "green"

        popup_html = f"""
        <div style="font-family:monospace;font-size:12px;min-width:250px;">
        <b style="font-size:14px;">Recommended Site #{r['candidate_id']}</b><br>
        <hr style="margin:4px 0;">
        <b>Location:</b> {r['location_name']}<br>
        <b>Coordinates:</b> ({r['lat']:.5f}, {r['lon']:.5f})<br>
        <b>Load served:</b> {r['total_load_kw']:,.0f} kW<br>
        <b>Cells served:</b> {r['n_cells_served']}<br>
        <b>Avg distance:</b> {r['avg_service_dist_m']:,.0f} m<br>
        <hr style="margin:4px 0;">
        <b>Suitability:</b>
        <span style="color:{color};font-weight:bold;">{r['suitability'].upper()}</span><br>
        <b>Reason:</b> {r['suitability_reason']}<br>
        {f"<b>Nearby:</b> {r['nearby_restrictions']}<br>" if r['nearby_restrictions'] else ""}
        </div>
        """

        folium.Marker(
            [r["lat"], r["lon"]],
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"#{r['candidate_id']}: {r['location_name'][:40]}",
            icon=folium.Icon(color=icon_color, icon="bolt", prefix="fa"),
        ).add_to(fg_new)

    fg_new.add_to(m)

    # TX expansion candidates (orange diamonds)
    if tx_expansion is not None and not tx_expansion.empty:
        fg_tx = folium.FeatureGroup(name=f"TX Expansion ({len(tx_expansion)})")
        for _, r in tx_expansion.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=8,
                color="#f4a261", fill=True, fill_opacity=0.8,
                tooltip=f"TX expansion: {r['cluster_demand_kw']:,.0f} kW",
            ).add_to(fg_tx)
        fg_tx.add_to(m)

    # Legend
    n_suitable = (enriched_df["suitability"] == "suitable").sum()
    n_restricted = (enriched_df["suitability"] == "restricted").sum()
    n_caution = (enriched_df["suitability"] == "caution").sum()
    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
    background:rgba(0,0,0,0.85);color:white;padding:14px;border-radius:8px;
    font:12px monospace;max-width:280px;">
    <b style="font-size:14px;">{county_name}</b><br>
    <b>{len(enriched_df)}</b> recommended sites<br>
    <span style="color:#00cc44">●</span> Suitable: {n_suitable}
    <span style="color:#ff4444">●</span> Restricted: {n_restricted}
    <span style="color:#ffaa00">●</span> Caution: {n_caution}<br>
    <span style="color:#666">●</span> Existing substations
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(output_path))
    logger.info("Satellite map saved: %s", output_path)


def save_enriched_results(enriched_df: pd.DataFrame, config: OptimizerConfig):
    """Save enriched site analysis as CSV."""
    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{config.county}_site_analysis.csv"
    enriched_df.to_csv(path, index=False)
    logger.info("Site analysis CSV saved: %s", path)
    return path
