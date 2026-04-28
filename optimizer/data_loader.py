"""Load all input datasets: model detections, Census, buildings, transmission, data centers."""

from __future__ import annotations

import json
import zipfile
from typing import Optional
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd

from .config import CENSUS_DIR, DATA_DIR, TIGER_YEAR, OptimizerConfig
from .containers import InputData


def load_input_data(config: OptimizerConfig) -> InputData:
    """Load model substations, transmission lines (optional), Census blocks,
    building footprints (optional). All filtered to config.effective_bbox."""
    bbox = config.effective_bbox
    if bbox:
        w, s, e, n = bbox
        width_km = (e - w) * 85
        height_km = (n - s) * 111
        print(f"  Analysis bbox: [{w:.3f}, {s:.3f}, {e:.3f}, {n:.3f}] "
              f"(~{width_km:.0f} x {height_km:.0f} km)")

    existing = load_model_substations(config.metadata_path, config.county_geoid,
                                      config.crs_proj, bbox)
    lines = load_transmission_lines(config)  # may return None
    blockgroups = load_census_population(config)

    # Filter block groups to bbox if set
    if bbox is not None:
        from shapely.geometry import box as make_box
        bbox_proj = gpd.GeoDataFrame(
            geometry=[make_box(*bbox)], crs="EPSG:4326"
        ).to_crs(config.crs_proj).geometry[0]
        before = len(blockgroups)
        blockgroups = blockgroups[blockgroups.geometry.intersects(bbox_proj)].copy()
        print(f"  Block groups clipped to bbox: {before} -> {len(blockgroups)}")

    buildings = load_building_footprints(config, blockgroups)
    return InputData(existing, lines, blockgroups, buildings)


def load_building_footprints(config: OptimizerConfig,
                             blockgroups: gpd.GeoDataFrame
                             ) -> Optional[pd.DataFrame]:
    """Load Microsoft Building Footprints and aggregate total footprint
    area (m²) per block group.  Returns None if the file is missing.

    Manual download required (multi-GB state files). Save to:
      data/buildings/{state_fips}_footprints.parquet   (preferred)
      data/buildings/{state_fips}_footprints.geojson   (alternative)

    Source: https://github.com/microsoft/USBuildingFootprints
    """
    buildings_dir = DATA_DIR / "buildings"
    for ext in (".parquet", ".geojson"):
        path = buildings_dir / f"{config.state_name}_footprints{ext}"
        if path.exists():
            break
    else:
        print("  Building footprints not found — demand will use population only")
        print(f"    To add: save as data/buildings/{config.state_name}_footprints.parquet")
        return None

    print(f"  Loading building footprints from {path.name}...")
    if path.suffix == ".parquet":
        bldg = gpd.read_parquet(path)
    else:
        bldg = gpd.read_file(path)

    # Clip buildings to analysis bbox BEFORE reprojecting (much faster)
    bbox = config.effective_bbox
    if bbox is not None:
        from shapely.geometry import box as make_box
        bbox_geom = make_box(*bbox)
        before = len(bldg)
        bldg = bldg[bldg.geometry.intersects(bbox_geom)].copy()
        print(f"  Buildings clipped to bbox: {before:,} -> {len(bldg):,}")

    bldg = bldg.to_crs(config.crs_proj)
    bldg["area_m2"] = bldg.geometry.area

    bldg_with_bg = gpd.sjoin(
        bldg[["geometry", "area_m2"]],
        blockgroups[["geometry", "GEOID"]],
        how="inner", predicate="within",
    )
    area_by_bg = (
        bldg_with_bg.groupby("GEOID")["area_m2"].sum().reset_index()
        .rename(columns={"area_m2": "building_area_m2"})
    )
    print(f"  Building footprints: {len(bldg):,} buildings -> "
          f"{len(area_by_bg)} block groups, "
          f"{area_by_bg['building_area_m2'].sum()/1e6:,.1f} M m²")
    return area_by_bg


def load_model_substations(metadata_path: str, county_geoid: str,
                           crs_proj: str,
                           bbox: list[float] = None) -> gpd.GeoDataFrame:
    """
    Load detected substations from the model metadata CSV.
    Filters to positive detections in the target county and optional bbox.
    """
    df = pd.read_csv(metadata_path)
    # Filter to positive detections in our county
    mask = (df["label"] == "positive") & (df["county_geoid"] == int(county_geoid))
    df = df[mask].copy()

    if df.empty:
        raise ValueError(f"No positive detections for county_geoid={county_geoid}")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["center_lon"], df["center_lat"]),
        crs="EPSG:4326",
    )

    # Filter to bbox if provided
    if bbox is not None:
        from shapely.geometry import box
        bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
        gdf = gdf[gdf.geometry.within(bbox_geom)].copy()
        print(f"  Loaded {len(gdf)} detected substations within analysis bbox")
    else:
        print(f"  Loaded {len(gdf)} detected substations for {county_geoid}")

    if gdf.empty:
        raise ValueError(f"No detections within bbox {bbox}")

    return gdf.to_crs(crs_proj)


def load_transmission_lines(config: OptimizerConfig) -> Optional[gpd.GeoDataFrame]:
    """Load OSM transmission lines (line + minor_line) if available for this county."""
    county_to_osm_folder = {
        "maricopa":   "maricopa_az",
        "allegheny":  "allegheny_pa",
        "lima":       "lima_peru",
    }

    folder = county_to_osm_folder.get(config.county, config.county)
    path = DATA_DIR / "osm" / folder / "power_line.geojson"

    if not path.exists():
        print(f"  No transmission lines found at {path}; skipping filter.")
        return None

    lines = gpd.read_file(path)
    if lines.empty:
        print(f"  Transmission line file is empty: {path}")
        return None

    minor_path = DATA_DIR / "osm" / folder / "power_minor_line.geojson"
    if minor_path.exists():
        minor = gpd.read_file(minor_path)
        if not minor.empty:
            lines = pd.concat([lines, minor], ignore_index=True)

    print(f"  Loaded {len(lines)} transmission line features from {folder}/")
    return gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326").to_crs(config.crs_proj)


def load_census_population(config: OptimizerConfig) -> gpd.GeoDataFrame:
    """
    Load block group population and TIGER geometries for the target county.
    Downloads automatically from Census APIs if not already cached locally.

    Local cache layout:
      data/census/pop_{state}_{county}.json   — Census API JSON
      data/census/tl_{year}_{state}_bg/       — unzipped TIGER shapefile
    """
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    pop_df = _load_or_download_population(config)
    bg_gdf = _load_or_download_tiger(config)

    # Filter TIGER to our county (STATEFP + COUNTYFP)
    bg_gdf = bg_gdf[
        (bg_gdf["STATEFP"] == config.state_fips) &
        (bg_gdf["COUNTYFP"] == config.county_fips)
    ].copy()

    # Join population onto geometries
    merged = bg_gdf.merge(pop_df, on="GEOID", how="left")
    merged["population"] = merged["population"].fillna(0).astype(int)

    print(f"  Loaded {len(merged)} block groups for {config.county_name} "
          f"(total pop: {merged['population'].sum():,})")
    return merged[["GEOID", "population", "geometry"]].to_crs(config.crs_proj)


def _load_or_download_population(config: OptimizerConfig) -> pd.DataFrame:
    """Load population JSON from cache, or download from Census API."""
    pop_path = CENSUS_DIR / f"pop_{config.state_fips}_{config.county_fips}.json"

    # Also check legacy naming (pop_fresno.json etc.) and preset overrides
    legacy_names = [
        config.preset.get("pop_json"),
        f"pop_{config.county.lower()}.json",
        f"pop_{config.county_name.split()[0].lower()}.json",
    ]
    for name in legacy_names:
        if name and (CENSUS_DIR / name).exists():
            pop_path = CENSUS_DIR / name
            break

    if not pop_path.exists():
        url = (
            f"https://api.census.gov/data/2020/dec/pl"
            f"?get=P1_001N"
            f"&for=block%20group:*"
            f"&in=state:{config.state_fips}"
            f"&in=county:{config.county_fips}"
        )
        print(f"  Downloading Census population from API...")
        with urlopen(url) as resp:
            raw = json.loads(resp.read().decode())
        with open(pop_path, "w") as f:
            json.dump(raw, f)
        print(f"  Saved to {pop_path.name}")
    else:
        with open(pop_path) as f:
            raw = json.load(f)

    header = raw[0]
    pop_df = pd.DataFrame(raw[1:], columns=header)
    pop_df["population"] = pop_df["P1_001N"].astype(int)
    pop_df["GEOID"] = (
        pop_df["state"] + pop_df["county"] + pop_df["tract"] + pop_df["block group"]
    )
    return pop_df[["GEOID", "population"]]


def _load_or_download_tiger(config: OptimizerConfig) -> gpd.GeoDataFrame:
    """Load TIGER block group shapefile from cache, or download from Census."""
    tiger_dir_name = f"tl_{TIGER_YEAR}_{config.state_fips}_bg"
    tiger_dir = CENSUS_DIR / tiger_dir_name
    zip_path = CENSUS_DIR / f"{tiger_dir_name}.zip"

    # Honor preset override if present and already on disk
    preset_dir = config.preset.get("tiger_dir")
    if preset_dir and (CENSUS_DIR / preset_dir).exists():
        tiger_dir = CENSUS_DIR / preset_dir

    if not tiger_dir.exists():
        if not zip_path.exists():
            url = (
                f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/BG/"
                f"tl_{TIGER_YEAR}_{config.state_fips}_bg.zip"
            )
            print(f"  Downloading TIGER block group shapefile...")
            with urlopen(url) as resp:
                zip_path.write_bytes(resp.read())
            print(f"  Saved to {zip_path.name}")

        print(f"  Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tiger_dir)

    shp_files = list(tiger_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {tiger_dir}")
    return gpd.read_file(shp_files[0])


def load_data_centers(config: OptimizerConfig) -> Optional[gpd.GeoDataFrame]:
    """Load data center locations from the IM3 atlas, filtered to the target county."""
    path = config.dc_atlas_path
    if not path.exists():
        print(f"  Data center atlas not found at {path}; skipping.")
        return None

    df = pd.read_csv(path)

    # Filter to target county using integer FIPS comparison
    mask = (
        (df["state_id"].astype(int) == int(config.state_fips)) &
        (df["county_id"].astype(str).str.zfill(3) == config.county_fips)
    )
    df = df[mask & df["sqft"].notna()].copy()

    if df.empty:
        print(f"  No data centers found for {config.county_name}; skipping.")
        return None

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    print(f"  Loaded {len(gdf)} data centers for {config.county_name} "
          f"(total sqft: {df['sqft'].sum():,.0f})")
    return gdf.to_crs(config.crs_proj)
