#!/usr/bin/env python3
"""
Build a training dataset for substation detection.

Examples:

    # Curated strategy, default 10 counties, per-type budgets
    python scripts/build_dataset.py \\
        --strategy curated \\
        --type-budget transmission=500 distribution=300 ""=200 \\
        --n-negative 1000

    # Randomized strategy, 30 random US regions
    python scripts/build_dataset.py \\
        --strategy randomized --n-random-regions 30 \\
        --n-positive 2000 --n-negative 2000

    # Quick test
    python scripts/build_dataset.py --n-positive 50 --n-negative 50 -v

    # With local NAIP for CA and TX
    python scripts/build_dataset.py \\
        --local-naip /data/ca_naip.tif /data/tx_naip.tif \\
        --n-positive 1000 --n-negative 1000

    # Custom counties and explicit test split
    python scripts/build_dataset.py \\
        --counties 06019 48201 53033 42071 \\
        --test-geoids 06019 --val-geoids 53033

    # Random split instead of geographic
    python scripts/build_dataset.py --split-method random
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grid_dataset.config import (
    CuratedStrategyConfig,
    DistanceRasterConfig,
    LabelAuditConfig,
    NaipConfig,
    PatchConfig,
    PipelineConfig,
    RandomizedStrategyConfig,
    SamplingConfig,
    SplitConfig,
    TemporalConfig,
)
from grid_dataset.dataset_builder import build_dataset


def _parse_type_budgets(values: list[str]) -> dict[str, int]:
    """Parse 'key=count' pairs into a dict. '\"\"=200' → {'': 200}."""
    budgets = {}
    for v in values:
        parts = v.split("=", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid type budget '{v}'. Use 'type=count'.")
        key = parts[0].strip().strip('"').strip("'")
        budgets[key] = int(parts[1])
    return budgets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a substation detection training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Output and reproducibility
    p.add_argument("-o", "--output", default="dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")

    # Strategy
    p.add_argument("--strategy", choices=["curated", "randomized"],
                   default="curated",
                   help="Sampling strategy (default: curated)")

    # Budget — flat
    p.add_argument("--n-positive", type=int, default=2000,
                   help="Total positive patches (ignored if --type-budget set)")
    p.add_argument("--n-negative", type=int, default=2000)

    # Budget — per type
    p.add_argument("--type-budget", nargs="+", default=[],
                   metavar="TYPE=N",
                   help='Per-type budgets, e.g.: transmission=500 distribution=300 ""=200')

    # Patch geometry
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--buffer-m", type=float, default=75.0)
    p.add_argument("--jitter-m", type=float, default=50.0)

    # Curated strategy
    p.add_argument("--counties", nargs="+", default=None,
                   metavar="GEOID",
                   help="County GEOIDs for curated strategy (overrides defaults)")
    p.add_argument("--no-balance", action="store_true",
                   help="Disable balanced per-county allocation")

    # Randomized strategy
    p.add_argument("--n-random-regions", type=int, default=30)
    p.add_argument("--region-size-km", type=float, default=50.0)
    p.add_argument("--min-inter-patch-m", type=float, default=2000.0)

    # NAIP
    p.add_argument("--local-naip", nargs="+", default=[],
                   metavar="PATH",
                   help="Local NAIP GeoTIFF paths (tried before STAC)")
    p.add_argument("--year-min", type=int, default=2022)
    p.add_argument("--year-max", type=int, default=2024)

    # Splitting
    p.add_argument("--split-method", choices=["geographic", "random"],
                   default="geographic")
    p.add_argument("--test-geoids", nargs="+", default=[])
    p.add_argument("--val-geoids", nargs="+", default=[])
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--val-fraction", type=float, default=0.15)

    # Feature 1: Temporal alignment
    p.add_argument("--temporal-check", action="store_true",
                   help="Enable imagery-label temporal alignment checking")
    p.add_argument("--temporal-max-gap", type=float, default=3.0,
                   help="Max years gap between NAIP and OSM timestamps (default: 3)")
    p.add_argument("--temporal-exclude", action="store_true",
                   help="Exclude temporally mismatched patches (default: flag only)")

    # Feature 2: Distance raster
    p.add_argument("--distance-raster", action="store_true",
                   help="Generate distance-to-substation rasters alongside masks")

    # Feature 3: COG robustness
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retry attempts for failed STAC fetches (default: 3)")
    p.add_argument("--verify-checksum", action="store_true",
                   help="Compute SHA-256 of each fetched patch for corruption detection")

    # Feature 4: Label noise audit
    p.add_argument("--label-audit", action="store_true",
                   help="Run NDVI/water label noise audit on positive patches")
    p.add_argument("--ndvi-threshold", type=float, default=0.4,
                   help="NDVI threshold for flagging vegetation in labels (default: 0.4)")
    p.add_argument("--ndwi-threshold", type=float, default=0.0,
                   help="NDWI threshold for flagging water in labels (default: 0.0)")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    type_budgets = _parse_type_budgets(args.type_budget) if args.type_budget else {}

    curated_geoids = (
        args.counties if args.counties
        else CuratedStrategyConfig().county_geoids
    )

    config = PipelineConfig(
        output_dir=args.output,
        seed=args.seed,
        patch=PatchConfig(
            patch_size=args.patch_size,
            point_buffer_m=args.buffer_m,
        ),
        sampling=SamplingConfig(
            strategy=args.strategy,
            n_positive_total=args.n_positive,
            n_negative_total=args.n_negative,
            type_budgets=type_budgets,
            jitter_m=args.jitter_m,
        ),
        curated=CuratedStrategyConfig(
            county_geoids=curated_geoids,
            balance_across_counties=not args.no_balance,
        ),
        randomized=RandomizedStrategyConfig(
            n_random_regions=args.n_random_regions,
            region_size_km=args.region_size_km,
            min_inter_patch_distance_m=args.min_inter_patch_m,
        ),
        split=SplitConfig(
            method=args.split_method,
            test_fraction=args.test_fraction,
            val_fraction=args.val_fraction,
            test_geoids=args.test_geoids,
            val_geoids=args.val_geoids,
        ),
        naip=NaipConfig(
            local_naip_paths=args.local_naip,
            year_range=(args.year_min, args.year_max),
            max_retries=args.max_retries,
            verify_checksum=args.verify_checksum,
        ),
        temporal=TemporalConfig(
            enabled=args.temporal_check,
            max_gap_years=args.temporal_max_gap,
            exclude_mismatched=args.temporal_exclude,
        ),
        distance_raster=DistanceRasterConfig(
            enabled=args.distance_raster,
        ),
        label_audit=LabelAuditConfig(
            enabled=args.label_audit,
            ndvi_threshold=args.ndvi_threshold,
            ndwi_threshold=args.ndwi_threshold,
        ),
    )

    log = logging.getLogger(__name__)
    log.info("Strategy: %s", config.sampling.strategy)
    log.info("Budget: %d positive + %d negative",
             config.sampling.n_positive_effective, config.sampling.n_negative_total)
    if type_budgets:
        log.info("Per-type budgets: %s", type_budgets)
    features = []
    if config.temporal.enabled:
        features.append("temporal-check")
    if config.distance_raster.enabled:
        features.append("distance-raster")
    if config.naip.verify_checksum:
        features.append("checksum-verify")
    if config.label_audit.enabled:
        features.append("label-audit")
    if features:
        log.info("Enabled features: %s", ", ".join(features))

    build_dataset(config)


if __name__ == "__main__":
    main()
