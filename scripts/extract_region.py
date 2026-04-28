#!/usr/bin/env python3
"""
Extract fresh NAIP patches covering a bounding box at 0.6m resolution.

NAIP imagery is 0.3m native resolution. This script reads 1024×1024
native pixel windows (307×307 m ground) and downsamples to 512×512
to produce patches at 0.6m — matching the model's training resolution.

Deduplicates across overlapping NAIP tiles so each ground cell is
extracted only once.

Usage:
    python scripts/extract_region.py --county maricopa -o demo_data -v
    python scripts/extract_region.py --county maricopa --dry-run
    python scripts/extract_region.py --county maricopa --bbox -112.10 33.38 -111.90 33.52 -o demo_data -v
    python scripts/extract_region.py --county maricopa -o demo_data --skip-existing -v
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimizer.config import COUNTY_PRESETS

logger = logging.getLogger(__name__)

# Output patch size (pixels) and target resolution
OUT_SIZE = 512
TARGET_RES = 0.6  # meters per pixel
GROUND_SPAN = OUT_SIZE * TARGET_RES  # 307.2 meters per patch edge

# How close two patch centers (in meters) can be before one is a duplicate
DEDUP_DIST_M = GROUND_SPAN * 0.5  # ~154m


def search_naip_tiles(bbox, year_min=2021, year_max=2024, max_tiles=300):
    import planetary_computer
    import pystac_client
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["naip"], bbox=bbox,
        datetime=f"{year_min}/{year_max}",
        sortby=[{"field": "datetime", "direction": "desc"}],
        max_items=max_tiles,
    )
    return list(search.items())


def downsample(data, target_h, target_w):
    """Simple area-average downsample from (C, H, W) to (C, target_h, target_w)."""
    from PIL import Image
    out = np.zeros((data.shape[0], target_h, target_w), dtype=data.dtype)
    for c in range(data.shape[0]):
        img = Image.fromarray(data[c])
        out[c] = np.array(img.resize((target_w, target_h), Image.LANCZOS))
    return out


def extract_patches_from_tile(item, bbox, img_dir, covered_cells,
                               skip_existing=False):
    """Extract downsampled patches from one NAIP tile within the bbox.
    Skips cells already in covered_cells. Returns list of metadata dicts."""
    href = item.assets["image"].href

    try:
        src = rasterio.open(href)
    except Exception as e:
        logger.warning("  Failed to open: %s", e)
        return []

    H, W = src.height, src.width
    transform = src.transform
    crs = src.crs
    native_res = abs(transform.a)
    n_bands = min(src.count, 4)

    # How many native pixels = one output patch
    native_window = int(round(GROUND_SPAN / native_res))  # ~1024 at 0.3m

    # CRS transformers
    from pyproj import Transformer
    if crs and not rasterio.crs.CRS(crs).is_geographic:
        to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    else:
        to_geo = to_proj = None

    # Convert WGS-84 bbox to pixel bounds
    if to_proj:
        bx0, by0 = to_proj.transform(bbox[0], bbox[1])
        bx1, by1 = to_proj.transform(bbox[2], bbox[3])
    else:
        bx0, by0, bx1, by1 = bbox[0], bbox[1], bbox[2], bbox[3]

    col0, row1 = ~transform * (min(bx0, bx1), min(by0, by1))
    col1, row0 = ~transform * (max(bx0, bx1), max(by0, by1))
    col0 = max(0, int(col0))
    row0 = max(0, int(row0))
    col1 = min(W, int(col1))
    row1 = min(H, int(row1))

    if col1 - col0 < native_window or row1 - row0 < native_window:
        src.close()
        return []

    patches = []
    step = native_window  # non-overlapping at ground level

    for r_off in range(row0, row1 - native_window + 1, step):
        for c_off in range(col0, col1 - native_window + 1, step):
            # Center in projected CRS
            cx = transform.c + (c_off + native_window / 2) * transform.a
            cy = transform.f + (r_off + native_window / 2) * transform.e

            if to_geo:
                lon, lat = to_geo.transform(cx, cy)
            else:
                lon, lat = cx, cy

            # Check center is within bbox
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue

            # Dedup: quantize to grid cell
            cell_key = (round(lon * 1000), round(lat * 1000))
            if cell_key in covered_cells:
                continue
            covered_cells.add(cell_key)

            tile_id = item.id
            pid = f"patch_{round(lat*10000):07d}_{round(abs(lon)*10000):07d}"
            out_path = img_dir / f"{pid}.tif"

            if skip_existing and out_path.exists():
                patches.append({
                    "patch_id": pid, "center_lon": lon, "center_lat": lat,
                    "tile_id": tile_id, "label": "unknown", "county_geoid": 0,
                })
                continue

            try:
                window = Window(c_off, r_off, native_window, native_window)
                data = src.read(list(range(1, n_bands + 1)), window=window)

                if data.shape[0] < 4:
                    pad = np.zeros((4 - data.shape[0], native_window, native_window),
                                   dtype=data.dtype)
                    data = np.concatenate([pad, data] if data.shape[0] == 3 else [data, pad], axis=0)
                    data = data[:4]

                if data.max() == 0:
                    covered_cells.discard(cell_key)
                    continue

                # Downsample to 512×512
                data_ds = downsample(data, OUT_SIZE, OUT_SIZE)

                # Build output transform at 0.6m resolution
                out_left = transform.c + c_off * transform.a
                out_top = transform.f + r_off * transform.e
                out_transform = rasterio.transform.from_bounds(
                    out_left, out_top + native_window * transform.e,
                    out_left + native_window * transform.a, out_top,
                    OUT_SIZE, OUT_SIZE,
                )

                with rasterio.open(
                    out_path, "w", driver="GTiff",
                    height=OUT_SIZE, width=OUT_SIZE,
                    count=4, dtype=data_ds.dtype,
                    crs=crs, transform=out_transform,
                ) as dst:
                    dst.write(data_ds)

                patches.append({
                    "patch_id": pid, "center_lon": lon, "center_lat": lat,
                    "tile_id": tile_id, "label": "unknown", "county_geoid": 0,
                })

            except Exception as e:
                logger.debug("  Patch failed: %s", e)
                covered_cells.discard(cell_key)

    src.close()
    return patches


def main():
    p = argparse.ArgumentParser(
        description="Extract NAIP patches at 0.6m resolution for a bounding box.")
    p.add_argument("--county", required=True, choices=list(COUNTY_PRESETS.keys()))
    p.add_argument("--bbox", type=float, nargs=4, default=None,
                   metavar=("W", "S", "E", "N"))
    p.add_argument("-o", "--output", default="demo_data")
    p.add_argument("--year-min", type=int, default=2021)
    p.add_argument("--year-max", type=int, default=2024)
    p.add_argument("--max-tiles", type=int, default=300)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",
    )
    for _n in ["rasterio", "fiona", "urllib3", "botocore", "pyproj"]:
        logging.getLogger(_n).setLevel(logging.WARNING)

    preset = COUNTY_PRESETS[args.county]
    geoid = int(preset["geoid"])

    if args.bbox:
        bbox = args.bbox
    elif "bbox" in preset:
        bbox = preset["bbox"]
    else:
        print(f"[!] No bbox for {args.county}. Use --bbox W S E N")
        sys.exit(1)

    width_km = (bbox[2] - bbox[0]) * 85
    height_km = (bbox[3] - bbox[1]) * 111
    area_km2 = width_km * height_km

    # At 0.6m, each 512px patch = 307m ground
    patches_x = int(width_km * 1000 / GROUND_SPAN)
    patches_y = int(height_km * 1000 / GROUND_SPAN)
    est_patches = patches_x * patches_y
    est_mb = est_patches * 0.4

    print("=" * 60)
    print(f"  Region Extraction — {preset['county_name']}")
    print("=" * 60)
    print(f"  Bbox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"  Size: ~{width_km:.0f} × {height_km:.0f} km ({area_km2:.0f} km²)")
    print(f"  Resolution: 0.6m (downsampled from 0.3m NAIP)")
    print(f"  Patch: 512×512 px = 307×307 m ground")
    print(f"  Estimated patches: ~{est_patches} ({patches_x} × {patches_y})")
    print(f"  Estimated disk: ~{est_mb:.0f} MB")

    # Search tiles
    print(f"\n[1/3] Searching NAIP tiles ({args.year_min}–{args.year_max})...")
    items = search_naip_tiles(bbox, args.year_min, args.year_max, args.max_tiles)
    print(f"  Found {len(items)} NAIP tiles")

    if not items:
        print("[!] No tiles found.")
        sys.exit(1)

    if args.dry_run:
        print(f"\n[DRY RUN] ~{est_patches} patches from {len(items)} tiles, ~{est_mb:.0f} MB")
        return

    out = Path(args.output)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] Extracting patches...")
    all_patches = []
    covered_cells = set()

    for i, item in enumerate(items):
        tile_id = item.id
        dt = item.datetime
        year = dt.year if dt else "?"

        print(f"  [{i+1}/{len(items)}] {tile_id} ({year}) ... ", end="", flush=True)

        patches = extract_patches_from_tile(
            item, bbox, img_dir, covered_cells,
            skip_existing=args.skip_existing,
        )

        for patch in patches:
            patch["county_geoid"] = geoid

        all_patches.extend(patches)
        print(f"{len(patches)} patches (coverage: {len(covered_cells)} cells)")

        time.sleep(0.3)

    # Build metadata
    print(f"\n[3/3] Building metadata.csv...")
    if not all_patches:
        print("[!] No patches extracted.")
        sys.exit(1)

    df = pd.DataFrame(all_patches)
    df = df.drop(columns=["tile_id"], errors="ignore")
    df = df.drop_duplicates(subset=["patch_id"])

    meta_path = out / "metadata.csv"
    df.to_csv(meta_path, index=False)

    on_disk = len(list(img_dir.glob("*.tif")))
    disk_mb = sum(f.stat().st_size for f in img_dir.glob("*.tif")) / 1e6

    print(f"\n{'=' * 60}")
    print(f"  Extraction Complete")
    print(f"{'=' * 60}")
    print(f"  Patches: {on_disk} ({disk_mb:.0f} MB)")
    print(f"  Metadata: {len(df)} rows → {meta_path}")
    print(f"  Output: {out}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
