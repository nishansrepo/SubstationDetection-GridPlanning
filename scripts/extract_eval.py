#!/usr/bin/env python3
"""
Extract evaluation and application datasets from NEW counties.

This script extracts four separate datasets, all from counties NOT used
in training or original validation/test, to provide uncontaminated
evaluation and application data:

    1. val_expanded:   Expanded validation set (2 new counties)
    2. test_general:   Generalizability test (3 new counties, diverse)
    3. test_location:  Location-specific test (Allegheny County, PA)
    4. full_inventory: EVERY substation in 2 counties for optimization
                       equation testing (no sampling — exhaustive)

Counties already used in training (CONTAMINATED — never extract these):
    06019 Fresno, 04013 Maricopa, 48201 Harris, 53033 King,
    20173 Sedgwick, 08013 Boulder, 27053 Hennepin, 42071 Lancaster,
    35001 Bernalillo, 37183 Wake

Usage:
    python scripts/extract_eval.py -o eval_datasets -v
    python scripts/extract_eval.py -o eval_datasets --skip-full-inventory
    python scripts/extract_eval.py -o eval_datasets --local-naip /path/to/ca.tif
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grid_dataset.config import (
    DistanceRasterConfig,
    LabelAuditConfig,
    NaipConfig,
    PatchConfig,
    PipelineConfig,
    SamplingConfig,
)
from grid_dataset.label_audit import run_label_audit
from grid_dataset.naip_source import NaipSource
from grid_dataset.negative_sampler import generate_negative_locations
from grid_dataset.osm_labels import fetch_substations_for_county
from grid_dataset.patch_extractor import extract_negative_patch, extract_positive_patch
from grid_dataset.quality import validate_dataset
from grid_dataset.regions import CountySpec

logger = logging.getLogger(__name__)

# ================================================================
# CONTAMINATED COUNTIES — these were used in model training
# ================================================================
CONTAMINATED = {
    "06019", "04013", "48201", "53033", "20173",
    "08013", "27053", "42071", "35001", "37183",
}

# ================================================================
# NEW COUNTY DEFINITIONS — all unseen by the trained model
# ================================================================

# Expanded validation: 2 counties from underrepresented landscapes
VAL_COUNTIES = [
    CountySpec("39049", "Franklin County",  "OH", "ohio_valley",
               "Franklin County", admin_level="6"),
    CountySpec("47037", "Davidson County",  "TN", "mid_south",
               "Davidson County", admin_level="6"),
]

# Test (generalizability): 3 counties spanning diverse, unseen landscapes
TEST_GENERAL_COUNTIES = [
    CountySpec("13121", "Fulton County",    "GA", "deep_south",
               "Fulton County", admin_level="6"),
    CountySpec("41051", "Multnomah County", "OR", "pacific_northwest",
               "Multnomah County", admin_level="6"),
    CountySpec("29189", "St. Louis County", "MO", "river_valley",
               "St. Louis County", admin_level="6"),
]

# Test (location-specific): for proof-of-concept application
TEST_LOCATION_COUNTIES = [
    CountySpec("42003", "Allegheny County", "PA", "appalachian",
               "Allegheny County", admin_level="6"),
]

# Full inventory: EVERY substation extracted (no sampling)
# Used for testing the optimization/siting equation
FULL_INVENTORY_COUNTIES = [
    CountySpec("42003", "Allegheny County", "PA", "appalachian",
               "Allegheny County", admin_level="6"),
    CountySpec("48453", "Travis County",    "TX", "hill_country",
               "Travis County", admin_level="6"),
]


def _get_county_bounds(county: CountySpec):
    """Fetch bounding box via Nominatim."""
    import requests
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": county.osm_area_name + ", " + county.state + ", United States",
                    "format": "json", "limit": 1},
            headers={"User-Agent": "grid-dataset-eval/0.2"},
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


def fetch_labels(counties, config, delay=5.0):
    """Fetch substations for a list of counties. Returns GeoDataFrame."""
    frames = []
    for i, county in enumerate(counties):
        if i > 0:
            time.sleep(delay)
        try:
            gdf = fetch_substations_for_county(county, config.patch.point_buffer_m)
            if not gdf.empty:
                frames.append(gdf)
                logger.info("  %s: %d substations", county.name, len(gdf))
            else:
                logger.warning("  %s: 0 substations found", county.name)
        except Exception:
            logger.exception("  Failed to fetch %s", county.name)

    if not frames:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")


def extract_sampled_set(
    name: str,
    counties: list,
    config: PipelineConfig,
    naip: NaipSource,
    output_dir: Path,
    rng: np.random.Generator,
    neg_ratio: float = 1.25,
):
    """Extract a sampled dataset (val or test) from given counties.

    Extracts ALL available substations (no sampling cap) plus proportional
    negatives. This is different from the training pipeline which caps
    per-county counts — for evaluation we want every substation.
    """
    logger.info("=" * 65)
    logger.info("Extracting: %s", name)
    logger.info("  Counties: %s", [c.name for c in counties])
    logger.info("=" * 65)

    raw_dir = output_dir / name / "raw"
    (raw_dir / "images").mkdir(parents=True, exist_ok=True)
    (raw_dir / "masks").mkdir(parents=True, exist_ok=True)
    if config.distance_raster.enabled:
        (raw_dir / "distances").mkdir(parents=True, exist_ok=True)

    # Fetch labels
    labels_path = output_dir / name / "substations.geojson"
    if labels_path.exists():
        substations = gpd.read_file(labels_path)
        logger.info("  Loaded cached labels: %d substations", len(substations))
    else:
        substations = fetch_labels(counties, config)
        if not substations.empty:
            (output_dir / name).mkdir(parents=True, exist_ok=True)
            substations.to_file(labels_path, driver="GeoJSON")

    if substations.empty:
        logger.warning("  No substations found, skipping %s", name)
        return pd.DataFrame()

    # Log type distribution
    type_dist = substations["substation_type"].value_counts()
    logger.info("  Type distribution:\n%s", type_dist.to_string())

    all_metadata = []

    # ---- Positive: extract EVERY substation ----
    logger.info("  Extracting positive patches (all %d substations)...",
                len(substations))

    for county in counties:
        county_subs = substations[substations["county_geoid"] == county.geoid]
        if county_subs.empty:
            continue

        extracted = 0
        for idx, (_, row) in enumerate(county_subs.iterrows()):
            centroid = row.geometry.centroid
            jitter = config.sampling.jitter_m / 111000
            lon = centroid.x + rng.uniform(-jitter, jitter)
            lat = centroid.y + rng.uniform(-jitter, jitter)

            stype = row.get("substation_type", "")
            pid = f"pos_{county.geoid}_{stype}_{row['osm_id']}_{idx:04d}"

            if (raw_dir / "images" / f"{pid}.tif").exists():
                continue

            result = extract_positive_patch(
                lon, lat, substations, naip, config, raw_dir, pid,
            )
            if result is not None:
                result.update({
                    "county_geoid": county.geoid,
                    "county_name": county.name,
                    "region": county.region,
                    "osm_id": int(row["osm_id"]),
                    "geom_source": row["geom_source"],
                    "voltage": row.get("voltage", ""),
                    "substation_type": stype,
                    "substation_name": row.get("name", ""),
                    "operator": row.get("operator", ""),
                    "split": name,
                })
                all_metadata.append(result)
                extracted += 1

        logger.info("    %s: %d/%d substations extracted",
                    county.name, extracted, len(county_subs))

    # ---- Negative: proportional to positive count ----
    n_pos = len([m for m in all_metadata if m["label"] == "positive"])
    n_neg_target = int(n_pos * neg_ratio)
    logger.info("  Extracting %d negative patches...", n_neg_target)

    # Distribute negatives proportionally across counties
    county_pos_counts = {}
    for m in all_metadata:
        if m["label"] == "positive":
            g = m["county_geoid"]
            county_pos_counts[g] = county_pos_counts.get(g, 0) + 1

    neg_extracted = 0
    for county in counties:
        county_pos = county_pos_counts.get(county.geoid, 0)
        if county_pos == 0:
            continue

        county_neg_target = max(1, int(
            county_pos / max(n_pos, 1) * n_neg_target
        ))

        bounds = _get_county_bounds(county)
        if bounds is None:
            logger.warning("    Cannot get bounds for %s", county.name)
            continue

        locs = generate_negative_locations(
            bounds, substations, county_neg_target, config, rng, county.name,
        )

        for i, (lon, lat) in enumerate(locs):
            pid = f"neg_{county.geoid}_{i:04d}"
            if (raw_dir / "images" / f"{pid}.tif").exists():
                continue

            result = extract_negative_patch(lon, lat, naip, config, raw_dir, pid)
            if result is not None:
                result.update({
                    "county_geoid": county.geoid,
                    "county_name": county.name,
                    "region": county.region,
                    "osm_id": -1,
                    "geom_source": "none",
                    "voltage": "",
                    "substation_type": "",
                    "substation_name": "",
                    "operator": "",
                    "split": name,
                })
                all_metadata.append(result)
                neg_extracted += 1

    logger.info("    Total negatives: %d", neg_extracted)

    # Save metadata
    meta_df = pd.DataFrame(all_metadata)
    meta_df.to_csv(raw_dir / "metadata.csv", index=False)

    # Summary
    n_pos_final = (meta_df["label"] == "positive").sum()
    n_neg_final = (meta_df["label"] == "negative").sum()
    logger.info("  %s complete: %d positive, %d negative, %d total",
                name, n_pos_final, n_neg_final, len(meta_df))

    return meta_df


def extract_full_inventory(
    county: CountySpec,
    config: PipelineConfig,
    naip: NaipSource,
    output_dir: Path,
    rng: np.random.Generator,
):
    """Extract EVERY substation in a county — no sampling, no skipping.

    This produces a complete inventory for optimization equation testing.
    Unlike the sampled extraction, this:
        - Extracts every substation regardless of mask pixel count
        - Uses zero jitter (exact centroid positioning)
        - Records full substation metadata including voltage and operator
        - Saves a separate GeoJSON with all substation geometries + attributes
    """
    name = f"inventory_{county.geoid}"
    logger.info("=" * 65)
    logger.info("Full inventory: %s (%s)", county.name, county.geoid)
    logger.info("=" * 65)

    inv_dir = output_dir / name
    raw_dir = inv_dir / "raw"
    (raw_dir / "images").mkdir(parents=True, exist_ok=True)
    (raw_dir / "masks").mkdir(parents=True, exist_ok=True)
    if config.distance_raster.enabled:
        (raw_dir / "distances").mkdir(parents=True, exist_ok=True)

    # Fetch ALL substations
    labels_path = inv_dir / "substations_complete.geojson"
    if labels_path.exists():
        substations = gpd.read_file(labels_path)
        logger.info("  Loaded cached inventory: %d substations", len(substations))
    else:
        substations = fetch_labels([county], config, delay=0)
        if not substations.empty:
            inv_dir.mkdir(parents=True, exist_ok=True)
            substations.to_file(labels_path, driver="GeoJSON")

    if substations.empty:
        logger.warning("  No substations found for %s", county.name)
        return pd.DataFrame()

    logger.info("  Total substations in %s: %d", county.name, len(substations))

    # Type breakdown
    type_dist = substations["substation_type"].value_counts()
    logger.info("  Types:\n%s", type_dist.to_string())

    # Voltage breakdown
    voltages = substations["voltage"].value_counts().head(10)
    if not voltages.empty:
        logger.info("  Top voltages:\n%s", voltages.to_string())

    # Operator breakdown
    operators = substations["operator"].value_counts().head(10)
    if not operators.empty:
        logger.info("  Top operators:\n%s", operators.to_string())

    # Extract every substation — NO jitter, NO min pixel filter
    all_metadata = []
    extracted = 0
    skipped_no_imagery = 0

    for idx, (_, row) in enumerate(substations.iterrows()):
        centroid = row.geometry.centroid
        lon = centroid.x  # exact centroid, no jitter
        lat = centroid.y

        stype = row.get("substation_type", "")
        pid = f"inv_{county.geoid}_{row['osm_id']}_{idx:04d}"

        if (raw_dir / "images" / f"{pid}.tif").exists():
            continue

        result = extract_positive_patch(
            lon, lat, substations, naip, config, raw_dir, pid,
        )

        if result is not None:
            result.update({
                "county_geoid": county.geoid,
                "county_name": county.name,
                "region": county.region,
                "osm_id": int(row["osm_id"]),
                "geom_source": row["geom_source"],
                "voltage": row.get("voltage", ""),
                "substation_type": stype,
                "substation_name": row.get("name", ""),
                "operator": row.get("operator", ""),
                "split": "inventory",
            })
            all_metadata.append(result)
            extracted += 1
        else:
            skipped_no_imagery += 1
            # Still record it in the inventory with what we know
            all_metadata.append({
                "patch_id": pid,
                "label": "positive_no_imagery",
                "center_lon": lon,
                "center_lat": lat,
                "positive_pixels": 0,
                "positive_fraction": 0,
                "total_pixels": 0,
                "source": "no_coverage",
                "stac_item_id": "",
                "crs": "",
                "patch_west": "", "patch_east": "",
                "patch_north": "", "patch_south": "",
                "patch_width_m": "", "patch_height_m": "",
                "acquisition_date": "",
                "resolution_x": "", "resolution_y": "",
                "actual_band_count": "",
                "sha256": "", "possibly_corrupt": False,
                "mask_bbox_row_min": "", "mask_bbox_row_max": "",
                "mask_bbox_col_min": "", "mask_bbox_col_max": "",
                "mask_bbox_width_px": "", "mask_bbox_height_px": "",
                "mask_bbox_width_m": "", "mask_bbox_height_m": "",
                "mask_centroid_x": "", "mask_centroid_y": "",
                "county_geoid": county.geoid,
                "county_name": county.name,
                "region": county.region,
                "osm_id": int(row["osm_id"]),
                "geom_source": row["geom_source"],
                "voltage": row.get("voltage", ""),
                "substation_type": stype,
                "substation_name": row.get("name", ""),
                "operator": row.get("operator", ""),
                "split": "inventory",
            })

        if (idx + 1) % 20 == 0:
            logger.info("    Progress: %d/%d extracted, %d no imagery",
                        extracted, idx + 1, skipped_no_imagery)

    # Save metadata
    meta_df = pd.DataFrame(all_metadata)
    meta_df.to_csv(raw_dir / "metadata.csv", index=False)

    # Save a summary CSV with just the inventory (for the optimization equation)
    inventory_summary = meta_df[[
        "patch_id", "osm_id", "center_lon", "center_lat",
        "substation_type", "substation_name", "operator", "voltage",
        "geom_source", "positive_pixels", "mask_bbox_width_m",
        "mask_bbox_height_m", "label",
    ]].copy()
    inventory_summary.to_csv(inv_dir / "substation_inventory.csv", index=False)

    # Summary
    logger.info("  Inventory complete: %d/%d extracted, %d no imagery",
                extracted, len(substations), skipped_no_imagery)
    logger.info("  Full inventory saved: %s", inv_dir / "substation_inventory.csv")
    logger.info("  Substation geometries: %s", labels_path)

    return meta_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract evaluation datasets from new (unseen) counties.",
    )
    parser.add_argument("-o", "--output", default="eval_datasets",
                        help="Output directory (default: eval_datasets)")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--local-naip", nargs="+", default=[],
                        help="Local NAIP GeoTIFF paths")
    parser.add_argument("--year-min", type=int, default=2020)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip expanded validation extraction")
    parser.add_argument("--skip-test-general", action="store_true",
                        help="Skip generalizability test extraction")
    parser.add_argument("--skip-test-location", action="store_true",
                        help="Skip location-specific test extraction")
    parser.add_argument("--skip-full-inventory", action="store_true",
                        help="Skip full county inventory extraction")
    parser.add_argument("--distance-raster", action="store_true",
                        help="Generate distance rasters")
    parser.add_argument("--label-audit", action="store_true",
                        help="Run NDVI/NDWI label audit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Verify no contamination
    all_new = (
        [c.geoid for c in VAL_COUNTIES]
        + [c.geoid for c in TEST_GENERAL_COUNTIES]
        + [c.geoid for c in TEST_LOCATION_COUNTIES]
        + [c.geoid for c in FULL_INVENTORY_COUNTIES]
    )
    overlap = set(all_new) & CONTAMINATED
    if overlap:
        logger.error("CONTAMINATION: counties %s overlap with training set!", overlap)
        sys.exit(1)

    logger.info("All extraction counties are unseen by the trained model.")

    config = PipelineConfig(
        output_dir=args.output,
        seed=args.seed,
        patch=PatchConfig(patch_size=512, resolution=0.6, point_buffer_m=75.0,
                          min_substation_pixels=50),
        sampling=SamplingConfig(jitter_m=60.0, negative_min_distance_m=500.0),
        naip=NaipConfig(
            local_naip_paths=args.local_naip,
            year_range=(args.year_min, args.year_max),
            max_retries=4,
            verify_checksum=True,
        ),
        distance_raster=DistanceRasterConfig(enabled=args.distance_raster),
        label_audit=LabelAuditConfig(
            enabled=args.label_audit, ndwi_threshold=0.3,
        ),
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_results = {}

    with NaipSource(config) as naip:

        # 1. Expanded validation
        if not args.skip_val:
            meta = extract_sampled_set(
                "val_expanded", VAL_COUNTIES, config, naip, output_dir, rng,
            )
            all_results["val_expanded"] = meta
            if args.label_audit and not meta.empty:
                run_label_audit(meta, output_dir / "val_expanded" / "raw", config)

        # 2. Generalizability test
        if not args.skip_test_general:
            meta = extract_sampled_set(
                "test_general", TEST_GENERAL_COUNTIES, config, naip, output_dir, rng,
            )
            all_results["test_general"] = meta
            if args.label_audit and not meta.empty:
                run_label_audit(meta, output_dir / "test_general" / "raw", config)

        # 3. Location-specific test
        if not args.skip_test_location:
            meta = extract_sampled_set(
                "test_location", TEST_LOCATION_COUNTIES, config, naip, output_dir, rng,
            )
            all_results["test_location"] = meta
            if args.label_audit and not meta.empty:
                run_label_audit(meta, output_dir / "test_location" / "raw", config)

        # 4. Full county inventories
        if not args.skip_full_inventory:
            # Use relaxed config for inventory — capture even tiny substations
            inv_config = PipelineConfig(
                output_dir=args.output,
                seed=args.seed,
                patch=PatchConfig(
                    patch_size=512, resolution=0.6, point_buffer_m=75.0,
                    min_substation_pixels=1,  # capture everything
                ),
                sampling=SamplingConfig(jitter_m=0.0),  # exact centroid
                naip=NaipConfig(
                    local_naip_paths=args.local_naip,
                    year_range=(args.year_min, args.year_max),
                    max_retries=4, verify_checksum=True,
                ),
                distance_raster=DistanceRasterConfig(enabled=args.distance_raster),
                label_audit=LabelAuditConfig(enabled=False),
            )
            for county in FULL_INVENTORY_COUNTIES:
                meta = extract_full_inventory(
                    county, inv_config, naip, output_dir, rng,
                )
                all_results[f"inventory_{county.geoid}"] = meta

    # ---- Summary ----
    logger.info("=" * 65)
    logger.info("  EXTRACTION SUMMARY")
    logger.info("=" * 65)

    summary = {}
    for name, meta in all_results.items():
        if meta is None or meta.empty:
            continue
        n_pos = (meta["label"] == "positive").sum()
        n_neg = (meta["label"] == "negative").sum()
        n_no_img = (meta["label"] == "positive_no_imagery").sum()
        counties = meta["county_geoid"].unique().tolist()
        logger.info("  %-20s  %4d pos  %4d neg  %3d no_img  counties=%s",
                    name, n_pos, n_neg, n_no_img, counties)
        summary[name] = {
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "n_no_imagery": int(n_no_img),
            "counties": counties,
        }

    with open(output_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("  Summary saved: %s", output_dir / "extraction_summary.json")
    logger.info("=" * 65)
    logger.info("Output structure:")
    logger.info("  %s/", output_dir)
    logger.info("    val_expanded/raw/{images,masks}/  — expanded validation")
    logger.info("    test_general/raw/{images,masks}/  — generalizability test")
    logger.info("    test_location/raw/{images,masks}/ — location-specific test")
    logger.info("    inventory_42003/                  — Allegheny County full")
    logger.info("      raw/{images,masks}/             — all substation patches")
    logger.info("      substation_inventory.csv        — complete inventory table")
    logger.info("      substations_complete.geojson    — all geometries")
    logger.info("    inventory_48453/                  — Travis County full")
    logger.info("      (same structure)")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
