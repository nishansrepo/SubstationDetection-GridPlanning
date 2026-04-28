"""
Dataset builder — orchestrates the full extraction pipeline.

Supports two strategies:
    'curated'    — Sample from pre-selected counties.
    'randomized' — Sample from randomly generated US regions.

Per-type budgets allow requesting specific counts for transmission,
distribution, and other substation types.

Resumable: checks for existing patches and skips them.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .label_audit import run_label_audit
from .naip_source import NaipSource
from .negative_sampler import generate_negative_locations
from .osm_labels import fetch_all_substations
from .patch_extractor import extract_negative_patch, extract_positive_patch
from .quality import compute_dataset_statistics, validate_dataset
from .regions import (
    COUNTY_REGISTRY,
    generate_random_regions,
    get_counties_for_geoids,
)
from .splitter import assign_splits, organize_by_split
from .temporal_check import check_temporal_alignment

logger = logging.getLogger(__name__)


def _allocate_typed_budget(
    substations: gpd.GeoDataFrame,
    type_budgets: dict[str, int],
    balance: bool,
) -> dict[str, dict[str, int]]:
    """Allocate per-type budgets across source regions.

    Returns nested dict: {substation_type: {county_geoid: count}}.
    """
    allocation: dict[str, dict[str, int]] = {}

    for stype, budget in type_budgets.items():
        # Filter substations by type
        if stype == "":
            mask = substations["substation_type"] == ""
        else:
            mask = substations["substation_type"] == stype

        typed = substations[mask]
        if typed.empty:
            logger.warning("No substations of type '%s' found", stype)
            allocation[stype] = {}
            continue

        per_source = typed.groupby("county_geoid").size().to_dict()

        if balance:
            allocation[stype] = _balanced_allocate(per_source, budget)
        else:
            allocation[stype] = _proportional_allocate(per_source, budget)

    return allocation


def _allocate_untyped_budget(
    substations: gpd.GeoDataFrame,
    total: int,
    balance: bool,
) -> dict[str, int]:
    """Allocate flat budget (no type distinction) across sources."""
    per_source = substations.groupby("county_geoid").size().to_dict()
    if balance:
        return _balanced_allocate(per_source, total)
    return _proportional_allocate(per_source, total)


def _balanced_allocate(per_source: dict[str, int], total: int) -> dict[str, int]:
    """Equal per-source, capped by availability."""
    sources = list(per_source.keys())
    if not sources:
        return {}

    target = total / len(sources)
    alloc = {}
    remaining = total

    # First pass: cap small sources
    leftover_sources = []
    for s in sources:
        avail = per_source[s]
        if avail <= target:
            alloc[s] = int(avail)
            remaining -= int(avail)
        else:
            leftover_sources.append(s)

    # Second pass: distribute remainder
    if leftover_sources:
        per_left = remaining / len(leftover_sources)
        for s in leftover_sources:
            alloc[s] = min(int(per_left), per_source[s])

    return alloc


def _proportional_allocate(per_source: dict[str, int], total: int) -> dict[str, int]:
    """Proportional to availability."""
    grand = sum(per_source.values())
    if grand == 0:
        return {s: 0 for s in per_source}
    return {s: int(c / grand * total) for s, c in per_source.items()}


def _get_source_bounds(
    geoid: str,
    counties_lookup: dict,
    regions_lookup: dict,
) -> tuple[float, float, float, float] | None:
    """Get bounding box for a source (county or random region)."""
    if geoid in regions_lookup:
        r = regions_lookup[geoid]
        return (r.west, r.south, r.east, r.north)

    if geoid in counties_lookup:
        from .negative_sampler import _get_county_bounds
        return _get_county_bounds(counties_lookup[geoid].osm_area_name)

    return None


def build_dataset(config: PipelineConfig) -> Path:
    """Execute the full pipeline. Returns path to final dataset."""
    errors = config.validate()
    if errors:
        raise ValueError("Config errors:\n" + "\n".join(f"  - {e}" for e in errors))

    rng = np.random.default_rng(config.seed)
    output = config.output_path
    raw_dir = output / "raw"
    (raw_dir / "images").mkdir(parents=True, exist_ok=True)
    (raw_dir / "masks").mkdir(parents=True, exist_ok=True)
    if config.distance_raster.enabled:
        (raw_dir / "distances").mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(output / "config.json", "w") as f:
        json.dump({
            "strategy": config.sampling.strategy,
            "patch_size": config.patch.patch_size,
            "resolution": config.patch.resolution,
            "n_positive_effective": config.sampling.n_positive_effective,
            "n_negative_total": config.sampling.n_negative_total,
            "type_budgets": config.sampling.type_budgets,
            "seed": config.seed,
            "year_range": list(config.naip.year_range),
            "split_method": config.split.method,
        }, f, indent=2)

    # ---- Step 1: Resolve sources and fetch labels ----
    logger.info("=" * 60)
    logger.info("Step 1: Fetching substation labels (%s strategy)",
                config.sampling.strategy)
    logger.info("=" * 60)

    counties = None
    random_regions_list = None
    counties_lookup: dict[str, object] = {}
    regions_lookup: dict[str, object] = {}

    if config.sampling.strategy == "curated":
        counties = get_counties_for_geoids(config.curated.county_geoids)
        counties_lookup = {c.geoid: c for c in counties}

    elif config.sampling.strategy == "randomized":
        random_regions_list = generate_random_regions(
            n_regions=config.randomized.n_random_regions,
            region_size_km=config.randomized.region_size_km,
            us_bounds=config.randomized.us_bounds,
            rng=rng,
        )
        regions_lookup = {
            f"rand_{r.region_id:04d}": r for r in random_regions_list
        }
        logger.info("Generated %d random regions", len(random_regions_list))

    labels_path = output / "substations.geojson"
    if labels_path.exists():
        substations = gpd.read_file(labels_path)
        logger.info("Loaded cached labels: %d substations", len(substations))
    else:
        substations = fetch_all_substations(
            counties=counties,
            random_regions=random_regions_list,
            buffer_m=config.patch.point_buffer_m,
        )
        substations.to_file(labels_path, driver="GeoJSON")

    # Log type distribution
    type_dist = substations["substation_type"].value_counts()
    logger.info("Substation type distribution:\n%s", type_dist.to_string())

    # ---- Step 2: Allocate budgets ----
    logger.info("=" * 60)
    logger.info("Step 2: Allocating budgets")
    logger.info("=" * 60)

    balance = (config.sampling.strategy == "curated"
               and config.curated.balance_across_counties)

    if config.sampling.type_budgets:
        typed_alloc = _allocate_typed_budget(
            substations, config.sampling.type_budgets, balance,
        )
        # Flatten to per-source totals for logging
        source_totals: dict[str, int] = {}
        for stype, per_source in typed_alloc.items():
            for geoid, count in per_source.items():
                source_totals[geoid] = source_totals.get(geoid, 0) + count
            logger.info("  Type '%s': %d patches across %d sources",
                        stype, sum(per_source.values()), len(per_source))
    else:
        flat_alloc = _allocate_untyped_budget(
            substations, config.sampling.n_positive_total, balance,
        )
        typed_alloc = {"__all__": flat_alloc}
        source_totals = flat_alloc

    # Negative allocation proportional to positive
    total_pos = config.sampling.n_positive_effective
    total_neg = config.sampling.n_negative_total
    neg_alloc: dict[str, int] = {}
    if total_pos > 0:
        for geoid, pos_count in source_totals.items():
            neg_alloc[geoid] = max(1, int(pos_count / total_pos * total_neg))

    # ---- Step 3 & 4: Extract patches ----
    logger.info("=" * 60)
    logger.info("Step 3: Extracting positive patches")
    logger.info("=" * 60)

    all_metadata = []

    with NaipSource(config) as naip:
        # Positive patches
        for stype, per_source in typed_alloc.items():
            for geoid, budget in per_source.items():
                if budget <= 0:
                    continue

                if stype == "__all__":
                    type_mask = pd.Series(True, index=substations.index)
                elif stype == "":
                    type_mask = substations["substation_type"] == ""
                else:
                    type_mask = substations["substation_type"] == stype

                source_mask = substations["county_geoid"] == geoid
                pool = substations[type_mask & source_mask]

                if pool.empty:
                    continue

                source_name = pool["county_name"].iloc[0]
                logger.info("  %s (type='%s'): %d patches from %d substations",
                            source_name, stype, budget, len(pool))

                replace = budget > len(pool)
                indices = rng.choice(len(pool), size=budget, replace=replace)
                extracted = 0

                for i, idx in enumerate(indices):
                    row = pool.iloc[idx]
                    centroid = row.geometry.centroid
                    jitter = config.sampling.jitter_m / 111000
                    lon = centroid.x + rng.uniform(-jitter, jitter)
                    lat = centroid.y + rng.uniform(-jitter, jitter)

                    type_tag = stype if stype != "__all__" else row["substation_type"]
                    pid = f"pos_{geoid}_{type_tag}_{row['osm_id']}_{i:04d}"

                    if (raw_dir / "images" / f"{pid}.tif").exists():
                        continue

                    result = extract_positive_patch(
                        lon, lat, substations, naip, config, raw_dir, pid,
                    )
                    if result is not None:
                        result.update({
                            "county_geoid": geoid,
                            "county_name": source_name,
                            "region": row.get("region", ""),
                            "osm_id": int(row["osm_id"]),
                            "geom_source": row["geom_source"],
                            "voltage": row.get("voltage", ""),
                            "substation_type": type_tag,
                            "substation_name": row.get("name", ""),
                            "operator": row.get("operator", ""),
                        })
                        all_metadata.append(result)
                        extracted += 1

                        # Detailed per-patch log
                        bbox_w = result.get("mask_bbox_width_m", "?")
                        bbox_h = result.get("mask_bbox_height_m", "?")
                        logger.debug(
                            "    [+] %s\n"
                            "        Location:    (%.6f, %.6f)  county=%s\n"
                            "        Patch CRS:   %s  extent=[%.1f,%.1f → %.1f,%.1f]\n"
                            "        Source:      %s  tile=%s  acquired=%s\n"
                            "        Substation:  OSM/%s  type=%s  voltage=%s\n"
                            "        Label geom:  %s  name='%s'  operator='%s'\n"
                            "        Mask:        %d px (%.2f%%)  bbox=%sx%s m\n"
                            "                     pixel rows [%s→%s] cols [%s→%s]",
                            pid,
                            lat, lon, source_name,
                            result.get("crs", ""), result.get("patch_west", ""),
                            result.get("patch_south", ""), result.get("patch_east", ""),
                            result.get("patch_north", ""),
                            result.get("source", ""), result.get("stac_item_id", ""),
                            result.get("acquisition_date", ""),
                            row["osm_id"], type_tag, row.get("voltage", ""),
                            row["geom_source"], row.get("name", ""),
                            row.get("operator", ""),
                            result["positive_pixels"],
                            result.get("positive_fraction", 0) * 100,
                            bbox_w, bbox_h,
                            result.get("mask_bbox_row_min", ""),
                            result.get("mask_bbox_row_max", ""),
                            result.get("mask_bbox_col_min", ""),
                            result.get("mask_bbox_col_max", ""),
                        )
                    else:
                        logger.debug(
                            "    [-] SKIP osm/%s at (%.6f,%.6f): "
                            "no imagery or insufficient mask pixels",
                            row["osm_id"], lat, lon,
                        )

                logger.info("    Extracted %d/%d", extracted, budget)

        # Negative patches
        logger.info("=" * 60)
        logger.info("Step 4: Extracting negative patches")
        logger.info("=" * 60)

        for geoid, budget in neg_alloc.items():
            if budget <= 0:
                continue

            bounds = _get_source_bounds(geoid, counties_lookup, regions_lookup)
            if bounds is None:
                logger.warning("Cannot determine bounds for %s", geoid)
                continue

            source_name = substations[
                substations["county_geoid"] == geoid
            ]["county_name"].iloc[0] if geoid in substations["county_geoid"].values else geoid

            logger.info("  %s: %d negatives", source_name, budget)

            locs = generate_negative_locations(
                bounds, substations, budget, config, rng, source_name,
            )
            extracted = 0
            for i, (lon, lat) in enumerate(locs):
                pid = f"neg_{geoid}_{i:04d}"
                if (raw_dir / "images" / f"{pid}.tif").exists():
                    continue
                result = extract_negative_patch(lon, lat, naip, config, raw_dir, pid)
                if result is not None:
                    result.update({
                        "county_geoid": geoid,
                        "county_name": source_name,
                        "region": substations[substations["county_geoid"] == geoid]["region"].iloc[0]
                            if geoid in substations["county_geoid"].values else "randomized",
                        "osm_id": -1,
                        "geom_source": "none",
                        "voltage": "",
                        "substation_type": "",
                        "substation_name": "",
                        "operator": "",
                    })
                    all_metadata.append(result)
                    extracted += 1

                    logger.debug(
                        "    [–] %s\n"
                        "        Location:  (%.6f, %.6f)  county=%s\n"
                        "        Patch CRS: %s  extent=[%.1f,%.1f → %.1f,%.1f]\n"
                        "        Source:    %s  tile=%s  acquired=%s",
                        pid,
                        lat, lon, source_name,
                        result.get("crs", ""), result.get("patch_west", ""),
                        result.get("patch_south", ""), result.get("patch_east", ""),
                        result.get("patch_north", ""),
                        result.get("source", ""), result.get("stac_item_id", ""),
                        result.get("acquisition_date", ""),
                    )
                else:
                    logger.debug(
                        "    [-] SKIP neg at (%.6f,%.6f): no imagery",
                        lat, lon,
                    )
            logger.info("    Extracted %d/%d", extracted, budget)

    # ---- Step 5: Validate ----
    logger.info("=" * 60)
    logger.info("Step 5: Validation")
    logger.info("=" * 60)

    metadata = pd.DataFrame(all_metadata)
    metadata.to_csv(raw_dir / "metadata.csv", index=False)

    qa = validate_dataset(
        raw_dir, metadata,
        expected_bands=len(config.naip.bands),
        expected_size=config.patch.patch_size,
        sample_fraction=min(1.0, 200 / max(len(metadata), 1)),
    )

    stats = compute_dataset_statistics(metadata)
    with open(output / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---- Step 5b: Temporal alignment check (Feature 1) ----
    if config.temporal.enabled:
        logger.info("=" * 60)
        logger.info("Step 5b: Temporal alignment check")
        logger.info("=" * 60)
        metadata = check_temporal_alignment(metadata, config)
        metadata.to_csv(raw_dir / "metadata.csv", index=False)

    # ---- Step 5c: Label noise audit (Feature 4) ----
    if config.label_audit.enabled:
        logger.info("=" * 60)
        logger.info("Step 5c: Label noise audit")
        logger.info("=" * 60)
        run_label_audit(metadata, raw_dir, config)

    # ---- Step 6: Split ----
    logger.info("=" * 60)
    logger.info("Step 6: Splitting (%s)", config.split.method)
    logger.info("=" * 60)

    valid_ids = set(qa[qa["is_valid"]]["patch_id"])
    unsampled = set(metadata["patch_id"]) - set(qa["patch_id"])
    keep = valid_ids | unsampled
    metadata_clean = metadata[metadata["patch_id"].isin(keep)].copy()

    metadata_split = assign_splits(metadata_clean, config)
    final_dir = output / "final"
    organize_by_split(metadata_split, raw_dir, final_dir)

    logger.info("=" * 60)
    logger.info("Complete! %d patches → %s", len(metadata_clean), final_dir)
    logger.info("=" * 60)

    return final_dir
