#!/usr/bin/env python3
"""
Fetch auxiliary datasets for a county: MS Building Footprints + OSM power lines.

Downloads:
    1. Microsoft Building Footprints (state-level GeoJSON from GitHub,
       filtered to county bounding box, saved as Parquet)
    2. OSM power lines and minor lines via Overpass API

Saves to:
    data/buildings/{StateName}_footprints.parquet
    data/osm/{county_key}/power_line.geojson
    data/osm/{county_key}/power_minor_line.geojson

Usage:
    python scripts/fetch_data.py --county maricopa
    python scripts/fetch_data.py --county allegheny --skip-buildings
    python scripts/fetch_data.py --county fresno --skip-osm
"""

import argparse
import json
import logging
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimizer.config import COUNTY_PRESETS, DATA_DIR

logger = logging.getLogger(__name__)

# State FIPS → URL name as used in minedbuildings.z5.web.core.windows.net
STATE_FIPS_TO_URL_NAME = {
    "04": "Arizona",       "06": "California",     "08": "Colorado",
    "17": "Illinois",      "20": "Kansas",          "27": "Minnesota",
    "35": "NewMexico",     "37": "NorthCarolina",   "42": "Pennsylvania",
    "48": "Texas",         "53": "Washington",
    # Extended for all preset counties
    "01": "Alabama",       "02": "Alaska",          "05": "Arkansas",
    "09": "Connecticut",   "10": "Delaware",        "11": "DistrictofColumbia",
    "12": "Florida",       "13": "Georgia",         "15": "Hawaii",
    "16": "Idaho",         "18": "Indiana",         "19": "Iowa",
    "21": "Kentucky",      "22": "Louisiana",       "23": "Maine",
    "24": "Maryland",      "25": "Massachusetts",   "26": "Michigan",
    "28": "Mississippi",   "29": "Missouri",        "30": "Montana",
    "31": "Nebraska",      "32": "Nevada",          "33": "NewHampshire",
    "34": "NewJersey",     "36": "NewYork",         "38": "NorthDakota",
    "39": "Ohio",          "40": "Oklahoma",        "41": "Oregon",
    "44": "RhodeIsland",   "45": "SouthCarolina",   "46": "SouthDakota",
    "47": "Tennessee",     "49": "Utah",            "50": "Vermont",
    "51": "Virginia",      "54": "WestVirginia",    "55": "Wisconsin",
    "56": "Wyoming",
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM folder names expected by data_loader.py
COUNTY_OSM_FOLDERS = {
    "maricopa": "maricopa_az", "allegheny": "allegheny_pa",
    "fresno": "fresno_ca",     "harris": "harris_tx",
    "king": "king_wa",         "lancaster": "lancaster_pa",
    "hennepin": "hennepin_mn", "wake": "wake_nc",
    "boulder": "boulder_co",   "sedgwick": "sedgwick_ks",
    "bernalillo": "bernalillo_nm",
}


def get_county_bbox(county_name, state_name):
    """Get county bounding box [south, west, north, east] from Nominatim."""
    query = f"{county_name}, {state_name}, United States"
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": "grid-dataset-fetch/0.3"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"Could not find boundary for {query}")
    bb = data[0]["boundingbox"]
    return float(bb[0]), float(bb[2]), float(bb[1]), float(bb[3])


# ================================================================
# Microsoft Building Footprints
# ================================================================

def fetch_ms_buildings(preset, output_dir):
    """Download MS Building Footprints for the state, filter to county bbox,
    save as Parquet."""
    state_fips = preset["state_fips"]
    url_name = STATE_FIPS_TO_URL_NAME.get(state_fips)
    if not url_name:
        logger.error("No building footprint URL mapping for state FIPS %s", state_fips)
        return None

    county_name = preset["county_name"]
    state_name = preset["state_name"]
    out_path = output_dir / f"{state_name}_footprints.parquet"

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        logger.info("Buildings file exists: %s (%.1f MB)", out_path.name, size_mb)
        return out_path

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the state GeoJSON zip
    url = (f"https://minedbuildings.z5.web.core.windows.net/"
           f"legacy/usbuildings-v2/{url_name}.geojson.zip")
    zip_path = output_dir / f"{url_name}.geojson.zip"
    geojson_path = output_dir / f"{url_name}.geojson"

    if not zip_path.exists() and not geojson_path.exists():
        logger.info("Downloading %s building footprints...", url_name)
        logger.info("  URL: %s", url)
        logger.info("  This is a large file (hundreds of MB). Please be patient.")

        req = Request(url, headers={"User-Agent": "grid-dataset-fetch/0.3"})
        try:
            with urlopen(req, timeout=600) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(zip_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            print(f"\r  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB "
                                  f"({pct:.0f}%)", end="", flush=True)
                        else:
                            print(f"\r  {downloaded / 1e6:.0f} MB", end="", flush=True)
                print()
            logger.info("  Downloaded: %s (%.0f MB)",
                        zip_path.name, zip_path.stat().st_size / 1e6)
        except Exception as e:
            logger.error("Download failed: %s", e)
            if zip_path.exists():
                zip_path.unlink()
            return None

    # Extract zip
    if zip_path.exists() and not geojson_path.exists():
        logger.info("  Extracting %s...", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            geojson_file = next((n for n in names if n.endswith(".geojson")), names[0])
            with zf.open(geojson_file) as src, open(geojson_path, "wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
        logger.info("  Extracted: %s", geojson_path.name)

    # Filter to county bbox and save as Parquet
    logger.info("  Loading and filtering to %s bbox...", county_name)
    south, west, north, east = get_county_bbox(county_name, state_name)
    bbox_geom = box(west, south, east, north)

    try:
        bldg = gpd.read_file(geojson_path, bbox=(west, south, east, north))
    except Exception:
        logger.info("  bbox filter not supported, loading full file...")
        bldg = gpd.read_file(geojson_path)
        bldg = bldg[bldg.geometry.intersects(bbox_geom)]

    logger.info("  Filtered to %d buildings within county bbox", len(bldg))

    bldg.to_parquet(out_path)
    logger.info("  Saved: %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)

    # Clean up large intermediate files
    if geojson_path.exists():
        geojson_path.unlink()
        logger.info("  Cleaned up: %s", geojson_path.name)
    if zip_path.exists():
        zip_path.unlink()
        logger.info("  Cleaned up: %s", zip_path.name)

    return out_path


# ================================================================
# OSM Power Infrastructure
# ================================================================

def fetch_osm_power(preset, county_key, output_dir):
    """Download OSM power lines via Overpass API."""
    county_name = preset["county_name"]
    state_name = preset["state_name"]

    osm_dir = output_dir / county_key
    line_path = osm_dir / "power_line.geojson"
    minor_path = osm_dir / "power_minor_line.geojson"

    if line_path.exists() and minor_path.exists():
        logger.info("OSM power data exists: %s/", county_key)
        return line_path, minor_path

    logger.info("Fetching OSM power lines for %s...", county_name)
    south, west, north, east = get_county_bbox(county_name, state_name)
    bbox_str = f"{south},{west},{north},{east}"
    logger.info("  Bbox: %s", bbox_str)

    osm_dir.mkdir(parents=True, exist_ok=True)

    for name, power_val, out_path in [
        ("power lines", "line", line_path),
        ("minor lines", "minor_line", minor_path),
    ]:
        if out_path.exists():
            logger.info("  %s already cached", name)
            continue

        query = f"""[out:json][timeout:120];
        (way["power"="{power_val}"]({bbox_str}););
        out body geom;"""

        logger.info("  Fetching %s from Overpass...", name)
        try:
            resp = requests.post(
                OVERPASS_URL, data={"data": query}, timeout=180,
                headers={"User-Agent": "grid-dataset-fetch/0.3"},
            )
            resp.raise_for_status()
            data = resp.json()

            features = []
            for elem in data.get("elements", []):
                if elem["type"] == "way" and "geometry" in elem:
                    coords = [(n["lon"], n["lat"]) for n in elem["geometry"]]
                    if len(coords) >= 2:
                        from shapely.geometry import LineString
                        props = dict(elem.get("tags", {}))
                        props["osm_id"] = elem["id"]
                        features.append({
                            "type": "Feature",
                            "geometry": LineString(coords).__geo_interface__,
                            "properties": props,
                        })

            geojson = {"type": "FeatureCollection", "features": features}
            with open(out_path, "w") as f:
                json.dump(geojson, f)
            logger.info("  Saved %d %s → %s", len(features), name, out_path.name)
            time.sleep(5)

        except Exception as e:
            logger.warning("  Failed to fetch %s: %s", name, e)
            with open(out_path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)

    return line_path, minor_path


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser(
        description="Fetch MS Building Footprints + OSM power lines for a county.")
    p.add_argument("--county", required=True, choices=list(COUNTY_PRESETS.keys()))
    p.add_argument("--skip-buildings", action="store_true")
    p.add_argument("--skip-osm", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",
    )

    preset = COUNTY_PRESETS[args.county]
    print(f"{'=' * 60}")
    print(f"  Fetching data for {preset['county_name']}")
    print(f"{'=' * 60}")

    if not args.skip_buildings:
        fetch_ms_buildings(preset, DATA_DIR / "buildings")

    if not args.skip_osm:
        folder = COUNTY_OSM_FOLDERS.get(args.county, args.county)
        fetch_osm_power(preset, folder, DATA_DIR / "osm")

    print(f"\n{'=' * 60}")
    print(f"  Done. Data saved to data/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
