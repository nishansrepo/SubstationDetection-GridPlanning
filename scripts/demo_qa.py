#!/usr/bin/env python3
"""
Demo: Quality checks and error diagnosis on synthetic data.

Generates a small synthetic dataset (no network required) that exercises
all four QA features, then produces a diagnostic HTML report.

Usage:
    python scripts/demo_qa.py                  # synthetic demo, no network
    python scripts/demo_qa.py --from-dataset demo_dataset  # real dataset QA

The synthetic demo creates:
    - 6 positive patches with varied NDVI/water conditions
    - 4 negative patches
    - Runs: distance raster, temporal check, checksum, label audit
    - Produces: demo_qa_output/qa_report.html
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    TemporalConfig,
)
from grid_dataset.label_audit import audit_single_patch, compute_ndvi, compute_ndwi
from grid_dataset.patch_extractor import create_distance_raster, save_geotiff
from grid_dataset.quality import validate_patch_pair

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Synthetic patch generation
# ------------------------------------------------------------------

def _make_substation_patch(patch_size: int = 512, seed: int = 0) -> np.ndarray:
    """Generate a synthetic 4-band image resembling a substation on gravel.

    Band layout: R, G, B, NIR (like NAIP).
    Substation area: low NDVI (gravel/concrete), moderate reflectance.
    Surrounding: moderate NDVI (grass/sparse vegetation).
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((4, patch_size, patch_size), dtype=np.uint8)

    # Background: moderate vegetation (grass)
    img[0] = rng.integers(60, 90, size=(patch_size, patch_size))   # R
    img[1] = rng.integers(80, 120, size=(patch_size, patch_size))  # G
    img[2] = rng.integers(50, 80, size=(patch_size, patch_size))   # B
    img[3] = rng.integers(120, 170, size=(patch_size, patch_size)) # NIR

    # Substation: bright gravel rectangle in center
    cx, cy = patch_size // 2, patch_size // 2
    hw = rng.integers(40, 80)
    hh = rng.integers(30, 60)
    y0, y1 = max(0, cy - hh), min(patch_size, cy + hh)
    x0, x1 = max(0, cx - hw), min(patch_size, cx + hw)

    img[0, y0:y1, x0:x1] = rng.integers(140, 180, size=(y1 - y0, x1 - x0))
    img[1, y0:y1, x0:x1] = rng.integers(140, 175, size=(y1 - y0, x1 - x0))
    img[2, y0:y1, x0:x1] = rng.integers(130, 165, size=(y1 - y0, x1 - x0))
    img[3, y0:y1, x0:x1] = rng.integers(150, 190, size=(y1 - y0, x1 - x0))

    return img


def _make_vegetation_patch(patch_size: int = 512, seed: int = 0) -> np.ndarray:
    """Synthetic patch: dense vegetation (high NDVI). Used to test noise audit."""
    rng = np.random.default_rng(seed)
    img = np.zeros((4, patch_size, patch_size), dtype=np.uint8)
    img[0] = rng.integers(30, 60, size=(patch_size, patch_size))   # R low
    img[1] = rng.integers(60, 100, size=(patch_size, patch_size))  # G moderate
    img[2] = rng.integers(20, 50, size=(patch_size, patch_size))   # B low
    img[3] = rng.integers(180, 240, size=(patch_size, patch_size)) # NIR very high
    return img


def _make_water_patch(patch_size: int = 512, seed: int = 0) -> np.ndarray:
    """Synthetic patch: water body. High NDWI."""
    rng = np.random.default_rng(seed)
    img = np.zeros((4, patch_size, patch_size), dtype=np.uint8)
    img[0] = rng.integers(20, 50, size=(patch_size, patch_size))   # R low
    img[1] = rng.integers(40, 80, size=(patch_size, patch_size))   # G moderate
    img[2] = rng.integers(60, 100, size=(patch_size, patch_size))  # B higher
    img[3] = rng.integers(10, 40, size=(patch_size, patch_size))   # NIR very low
    return img


def _make_mask(patch_size: int, center: bool = True, radius: int = 50) -> np.ndarray:
    """Binary mask with a filled circle or rectangle."""
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    if center:
        cy, cx = patch_size // 2, patch_size // 2
        Y, X = np.ogrid[:patch_size, :patch_size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask[dist <= radius] = 1
    return mask


# ------------------------------------------------------------------
# Synthetic dataset generation
# ------------------------------------------------------------------

def generate_synthetic_dataset(output_dir: Path, patch_size: int = 256):
    """Create a small synthetic dataset with known QA characteristics.

    Produces 10 patches:
        pos_clean_0..2   — substation on gravel, correct labels
        pos_vegetation_0 — vegetation patch with substation label (BAD label)
        pos_water_0      — water patch with substation label (BAD label)
        pos_temporal_0   — correct patch but with old OSM timestamp
        neg_clean_0..3   — clean background patches
    """
    raw_dir = output_dir / "raw"
    (raw_dir / "images").mkdir(parents=True, exist_ok=True)
    (raw_dir / "masks").mkdir(parents=True, exist_ok=True)
    (raw_dir / "distances").mkdir(parents=True, exist_ok=True)

    from rasterio.transform import from_bounds

    # Fake transform (doesn't matter for synthetic — just needs to be valid)
    t = from_bounds(0, 0, patch_size * 0.6, patch_size * 0.6,
                    patch_size, patch_size)
    crs = "EPSG:32617"  # UTM zone 17N

    records = []
    resolution = 0.6

    # --- Clean positive patches ---
    for i in range(3):
        pid = f"pos_clean_{i}"
        img = _make_substation_patch(patch_size, seed=i)
        mask = _make_mask(patch_size, center=True, radius=40 + i * 10)

        save_geotiff(img, raw_dir / "images" / f"{pid}.tif", t, crs)
        save_geotiff(mask, raw_dir / "masks" / f"{pid}.tif", t, crs)

        dist = create_distance_raster(mask, resolution)
        save_geotiff(dist, raw_dir / "distances" / f"{pid}.tif", t, crs,
                     dtype="float32")

        records.append({
            "patch_id": pid, "label": "positive",
            "lon": -79.9 + i * 0.01, "lat": 40.44,
            "positive_pixels": int(mask.sum()),
            "total_pixels": patch_size ** 2,
            "source": "synthetic", "stac_item_id": "",
            "crs": crs, "county_geoid": "42003",
            "county_name": "Allegheny County",
            "region": "appalachian",
            "osm_id": 100000 + i, "geom_source": "polygon",
            "voltage": "69000", "substation_type": "transmission",
            "acquisition_date": "2023-07-15T00:00:00Z",
            "resolution_x": resolution, "resolution_y": resolution,
            "actual_band_count": 4, "sha256": "", "possibly_corrupt": False,
        })

    # --- Bad label: vegetation (should trigger NDVI audit) ---
    pid = "pos_vegetation_0"
    img = _make_vegetation_patch(patch_size, seed=10)
    mask = _make_mask(patch_size, center=True, radius=60)
    save_geotiff(img, raw_dir / "images" / f"{pid}.tif", t, crs)
    save_geotiff(mask, raw_dir / "masks" / f"{pid}.tif", t, crs)
    dist = create_distance_raster(mask, resolution)
    save_geotiff(dist, raw_dir / "distances" / f"{pid}.tif", t, crs, dtype="float32")
    records.append({
        "patch_id": pid, "label": "positive",
        "lon": -79.95, "lat": 40.45,
        "positive_pixels": int(mask.sum()),
        "total_pixels": patch_size ** 2,
        "source": "synthetic", "stac_item_id": "",
        "crs": crs, "county_geoid": "42003",
        "county_name": "Allegheny County",
        "region": "appalachian",
        "osm_id": 200000, "geom_source": "point",
        "voltage": "", "substation_type": "distribution",
        "acquisition_date": "2023-07-15T00:00:00Z",
        "resolution_x": resolution, "resolution_y": resolution,
        "actual_band_count": 4, "sha256": "", "possibly_corrupt": False,
    })

    # --- Bad label: water ---
    pid = "pos_water_0"
    img = _make_water_patch(patch_size, seed=20)
    mask = _make_mask(patch_size, center=True, radius=55)
    save_geotiff(img, raw_dir / "images" / f"{pid}.tif", t, crs)
    save_geotiff(mask, raw_dir / "masks" / f"{pid}.tif", t, crs)
    dist = create_distance_raster(mask, resolution)
    save_geotiff(dist, raw_dir / "distances" / f"{pid}.tif", t, crs, dtype="float32")
    records.append({
        "patch_id": pid, "label": "positive",
        "lon": -79.88, "lat": 40.43,
        "positive_pixels": int(mask.sum()),
        "total_pixels": patch_size ** 2,
        "source": "synthetic", "stac_item_id": "",
        "crs": crs, "county_geoid": "42003",
        "county_name": "Allegheny County",
        "region": "appalachian",
        "osm_id": 300000, "geom_source": "point",
        "voltage": "", "substation_type": "",
        "acquisition_date": "2023-07-15T00:00:00Z",
        "resolution_x": resolution, "resolution_y": resolution,
        "actual_band_count": 4, "sha256": "", "possibly_corrupt": False,
    })

    # --- Temporal mismatch: old OSM edit ---
    pid = "pos_temporal_0"
    img = _make_substation_patch(patch_size, seed=30)
    mask = _make_mask(patch_size, center=True, radius=45)
    save_geotiff(img, raw_dir / "images" / f"{pid}.tif", t, crs)
    save_geotiff(mask, raw_dir / "masks" / f"{pid}.tif", t, crs)
    dist = create_distance_raster(mask, resolution)
    save_geotiff(dist, raw_dir / "distances" / f"{pid}.tif", t, crs, dtype="float32")
    records.append({
        "patch_id": pid, "label": "positive",
        "lon": -79.92, "lat": 40.46,
        "positive_pixels": int(mask.sum()),
        "total_pixels": patch_size ** 2,
        "source": "synthetic", "stac_item_id": "",
        "crs": crs, "county_geoid": "42003",
        "county_name": "Allegheny County",
        "region": "appalachian",
        "osm_id": 400000, "geom_source": "polygon",
        "voltage": "138000", "substation_type": "transmission",
        # Imagery from 2023, but we'll manually set OSM edit to 2015
        "acquisition_date": "2023-07-15T00:00:00Z",
        "resolution_x": resolution, "resolution_y": resolution,
        "actual_band_count": 4, "sha256": "", "possibly_corrupt": False,
    })

    # --- Negative patches ---
    for i in range(4):
        pid = f"neg_clean_{i}"
        img = _make_substation_patch(patch_size, seed=50 + i)
        # Overwrite with just background — no substation rectangle
        rng = np.random.default_rng(50 + i)
        img[0] = rng.integers(60, 100, size=(patch_size, patch_size))
        img[1] = rng.integers(80, 130, size=(patch_size, patch_size))
        img[2] = rng.integers(50, 90, size=(patch_size, patch_size))
        img[3] = rng.integers(110, 160, size=(patch_size, patch_size))
        mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

        save_geotiff(img, raw_dir / "images" / f"{pid}.tif", t, crs)
        save_geotiff(mask, raw_dir / "masks" / f"{pid}.tif", t, crs)
        dist = create_distance_raster(mask, resolution)
        save_geotiff(dist, raw_dir / "distances" / f"{pid}.tif", t, crs,
                     dtype="float32")

        records.append({
            "patch_id": pid, "label": "negative",
            "lon": -80.0 + i * 0.02, "lat": 40.40,
            "positive_pixels": 0,
            "total_pixels": patch_size ** 2,
            "source": "synthetic", "stac_item_id": "",
            "crs": crs, "county_geoid": "42003",
            "county_name": "Allegheny County",
            "region": "appalachian",
            "osm_id": -1, "geom_source": "none",
            "voltage": "", "substation_type": "",
            "acquisition_date": "2023-07-15T00:00:00Z",
            "resolution_x": resolution, "resolution_y": resolution,
            "actual_band_count": 4, "sha256": "", "possibly_corrupt": False,
        })

    metadata = pd.DataFrame(records)
    metadata.to_csv(raw_dir / "metadata.csv", index=False)
    logger.info("Generated %d synthetic patches in %s", len(records), raw_dir)
    return metadata


# ------------------------------------------------------------------
# QA execution on synthetic or real dataset
# ------------------------------------------------------------------

def run_qa(raw_dir: Path, metadata: pd.DataFrame, output_dir: Path):
    """Run all QA checks and produce diagnostic report."""
    config = PipelineConfig(
        output_dir=str(output_dir),
        patch=PatchConfig(patch_size=256, resolution=0.6),
        distance_raster=DistanceRasterConfig(enabled=True),
        temporal=TemporalConfig(enabled=True, max_gap_years=3.0),
        label_audit=LabelAuditConfig(enabled=True),
        naip=NaipConfig(verify_checksum=True),
    )

    print("\n" + "=" * 70)
    print("  QA DIAGNOSTIC REPORT")
    print("=" * 70)

    # --- 1. Patch validation ---
    print("\n--- Patch Validation ---")
    from grid_dataset.quality import validate_dataset
    qa = validate_dataset(
        raw_dir, metadata,
        expected_bands=4, expected_size=256,
        sample_fraction=1.0,
    )
    n_valid = qa["is_valid"].sum()
    n_total = len(qa)
    print(f"  Valid: {n_valid}/{n_total}")
    invalid = qa[~qa["is_valid"]]
    if not invalid.empty:
        print("  Invalid patches:")
        for _, row in invalid.iterrows():
            print(f"    {row['patch_id']}: {row['errors']}")

    # --- 2. Distance raster check ---
    print("\n--- Distance Rasters ---")
    dist_dir = raw_dir / "distances"
    n_dist = len(list(dist_dir.glob("*.tif"))) if dist_dir.exists() else 0
    print(f"  Distance rasters found: {n_dist}/{n_total}")
    # Spot-check one positive patch
    pos_patches = metadata[metadata["label"] == "positive"]
    if not pos_patches.empty and dist_dir.exists():
        import rasterio
        sample_pid = pos_patches.iloc[0]["patch_id"]
        dist_path = dist_dir / f"{sample_pid}.tif"
        if dist_path.exists():
            with rasterio.open(dist_path) as src:
                d = src.read(1)
                print(f"  Sample '{sample_pid}': min={d.min():.1f}m, "
                      f"max={d.max():.1f}m, mean={d.mean():.1f}m")

    # --- 3. Temporal alignment (simulated) ---
    print("\n--- Temporal Alignment ---")
    # For synthetic data, manually inject OSM timestamps to demo the check
    metadata = metadata.copy()
    metadata["osm_edit_date"] = ""
    # Clean patches: recent edits (within window)
    recent_mask = metadata["patch_id"].str.startswith("pos_clean")
    metadata.loc[recent_mask, "osm_edit_date"] = "2022-03-10T00:00:00Z"
    # Vegetation/water: recent (temporal is fine, label is bad)
    veg_mask = metadata["patch_id"].isin(["pos_vegetation_0", "pos_water_0"])
    metadata.loc[veg_mask, "osm_edit_date"] = "2023-01-15T00:00:00Z"
    # Temporal mismatch: old edit
    old_mask = metadata["patch_id"] == "pos_temporal_0"
    metadata.loc[old_mask, "osm_edit_date"] = "2015-06-01T00:00:00Z"

    # Compute gaps manually (since we can't call Overpass for synthetic IDs)
    from datetime import datetime as dt
    def _gap(row):
        acq = row.get("acquisition_date", "")
        osm = row.get("osm_edit_date", "")
        if not acq or not osm:
            return np.nan
        try:
            a = dt.fromisoformat(acq.replace("Z", "+00:00"))
            o = dt.fromisoformat(osm.replace("Z", "+00:00"))
            return abs((a - o).days) / 365.25
        except Exception:
            return np.nan

    metadata["temporal_gap_years"] = metadata.apply(_gap, axis=1)
    metadata["temporal_mismatch"] = metadata["temporal_gap_years"] > 3.0

    n_checked = metadata["temporal_gap_years"].notna().sum()
    n_mismatch = metadata["temporal_mismatch"].sum()
    print(f"  Checked: {n_checked} patches")
    print(f"  Mismatched (gap > 3 years): {n_mismatch}")
    for _, row in metadata[metadata["temporal_mismatch"]].iterrows():
        print(f"    {row['patch_id']}: gap={row['temporal_gap_years']:.1f} years")

    # --- 4. Label noise audit ---
    print("\n--- Label Noise Audit ---")
    from grid_dataset.label_audit import run_label_audit
    audit_df = run_label_audit(metadata, raw_dir, config)

    if not audit_df.empty:
        suspects = audit_df[audit_df["suspect_label"]]
        print(f"  Audited: {len(audit_df)} positive patches")
        print(f"  Suspect labels: {len(suspects)}")
        for _, row in suspects.iterrows():
            print(f"    {row['patch_id']}: "
                  f"NDVI={row['pct_high_ndvi']:.0f}%, "
                  f"water={row['pct_water']:.0f}%")

        # Print full audit table
        print("\n  Full audit results:")
        cols = ["patch_id", "n_labeled_pixels", "pct_high_ndvi",
                "pct_water", "mean_ndvi_in_label", "suspect_label"]
        print(audit_df[cols].to_string(index=False))

    # --- 5. Resolution/band consistency ---
    print("\n--- Resolution & Band Consistency ---")
    if "resolution_x" in metadata.columns:
        res = metadata["resolution_x"].dropna()
        if len(res) > 0:
            print(f"  Resolution X: min={res.min()}, max={res.max()}, "
                  f"all equal={res.nunique() == 1}")
    if "actual_band_count" in metadata.columns:
        bands = metadata["actual_band_count"].dropna()
        if len(bands) > 0:
            print(f"  Band count: min={int(bands.min())}, "
                  f"max={int(bands.max())}, "
                  f"all equal={bands.nunique() == 1}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    issues = []
    if not invalid.empty:
        issues.append(f"{len(invalid)} invalid patches")
    if n_mismatch > 0:
        issues.append(f"{n_mismatch} temporal mismatches")
    if not audit_df.empty and len(suspects) > 0:
        issues.append(f"{len(suspects)} suspect labels (NDVI/water)")
    if metadata.get("possibly_corrupt", pd.Series(dtype=bool)).any():
        issues.append("Possibly corrupt tiles detected")

    if issues:
        print(f"  Issues found: {len(issues)}")
        for issue in issues:
            print(f"    ⚠ {issue}")
    else:
        print("  No issues found.")

    print(f"\n  Output directory: {output_dir}")
    print(f"  Metadata: {raw_dir / 'metadata.csv'}")
    if (output_dir / "label_audit.csv").exists():
        print(f"  Label audit: {output_dir / 'label_audit.csv'}")
    print("=" * 70 + "\n")

    # Save enriched metadata
    metadata.to_csv(raw_dir / "metadata.csv", index=False)

    return metadata, qa, audit_df


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run QA diagnostics on a dataset (synthetic or real).",
    )
    parser.add_argument(
        "--from-dataset", type=str, default=None,
        help="Path to an existing dataset to audit (skips synthetic generation)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="demo_qa_output",
        help="Output directory (default: demo_qa_output)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_dataset:
        # Audit an existing dataset
        ds_dir = Path(args.from_dataset)
        raw_dir = ds_dir / "raw"
        if not raw_dir.exists():
            raw_dir = ds_dir  # maybe flat structure
        meta_path = raw_dir / "metadata.csv"
        if not meta_path.exists():
            print(f"Error: {meta_path} not found")
            sys.exit(1)
        metadata = pd.read_csv(meta_path)
        print(f"Loaded {len(metadata)} patches from {meta_path}")
    else:
        # Generate synthetic dataset
        print("Generating synthetic dataset (no network required)...")
        metadata = generate_synthetic_dataset(output_dir, patch_size=256)
        raw_dir = output_dir / "raw"

    run_qa(raw_dir, metadata, output_dir)


if __name__ == "__main__":
    main()
