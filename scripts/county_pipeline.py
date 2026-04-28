#!/usr/bin/env python3
"""
Full end-to-end pipeline: detect substations → optimize new placement.

    Input:  A county key (e.g. "maricopa", "allegheny")
    Output: Where to build new substations + maps + CSVs

Pipeline steps (all logged transparently):
    1. Search NAIP tiles covering the county via Planetary Computer STAC
    2. Process each tile individually: stream → model → detections
       (no giant mosaic — memory-bounded by one tile at a time)
    3. Deduplicate detections near tile boundaries
    4. Save optimizer-compatible metadata.csv
    5. Run p-median MILP optimizer → new substation siting recommendations
    6. Generate visualizations (results map, coverage heatmap)

Usage:
    # Full pipeline
    python scripts/county_pipeline.py \\
        --county maricopa \\
        --model-path model/final_model.pt \\
        -o output/maricopa -v

    # Skip NAIP — use an existing local raster
    python scripts/county_pipeline.py \\
        --county allegheny \\
        --naip-raster /path/to/allegheny_naip.tif \\
        --model-path model/final_model.pt \\
        -o output/allegheny -v

    # Skip detection — use existing metadata.csv
    python scripts/county_pipeline.py \\
        --county maricopa \\
        --skip-detection \\
        --metadata output/maricopa/metadata.csv \\
        -o output/maricopa -v

    # Tune optimizer
    python scripts/county_pipeline.py \\
        --county fresno \\
        --model-path model/final_model.pt \\
        --max-new 30 --build-cost 2e6 --time-limit 600 \\
        -o output/fresno -v

    # Sensitivity analysis
    python scripts/county_pipeline.py \\
        --county maricopa \\
        --model-path model/final_model.pt \\
        --sensitivity \\
        -o output/maricopa -v
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import torch
from shapely.geometry import Point
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.config import COUNTY_PRESETS, OptimizerConfig
from optimizer.pipeline import run_optimizer
from optimizer.sensitivity import run_sensitivity_analysis

logger = logging.getLogger(__name__)


# ================================================================
# Step 1: Search NAIP tiles (no download)
# ================================================================

def search_naip_tiles(county_name, state_name, year_min=2020, year_max=2025,
                      max_tiles=200):
    """Search Planetary Computer for NAIP tiles covering the county.
    Returns list of STAC items (with signed URLs). Does NOT download."""
    import requests
    import planetary_computer
    import pystac_client

    # Get county bbox from Nominatim
    query = f"{county_name}, {state_name}, United States"
    logger.info("  Querying Nominatim: %s", query)
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": "grid-dataset-pipeline/0.3"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"Could not find boundary for {query}")

    bb = data[0]["boundingbox"]
    bbox = [float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])]
    logger.info("  County bbox (WGS84): [%.4f, %.4f, %.4f, %.4f]", *bbox)
    width_km = (bbox[2] - bbox[0]) * 85
    height_km = (bbox[3] - bbox[1]) * 111
    logger.info("  Approximate extent: %.0f × %.0f km", width_km, height_km)

    # Search STAC
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    logger.info("  Searching NAIP collection (%d–%d)...", year_min, year_max)
    search = catalog.search(
        collections=["naip"], bbox=bbox,
        datetime=f"{year_min}/{year_max}",
        sortby=[{"field": "datetime", "direction": "desc"}],
        max_items=max_tiles,
    )
    items = list(search.items())
    logger.info("  Found %d NAIP tiles", len(items))

    if not items:
        raise RuntimeError("No NAIP tiles found for this area/time range")

    years = set()
    for item in items:
        dt = item.datetime
        if dt:
            years.add(dt.year)
    logger.info("  NAIP years: %s", sorted(years))

    return items, bbox


# ================================================================
# Step 2: Per-tile detection (stream → model → detections)
# ================================================================

@torch.no_grad()
def detect_in_tile(model, tile_href, device, mean_t, std_t,
                   patch_size=512, overlap=64, threshold=0.5,
                   batch_size=8, min_area_m2=200):
    """Open one NAIP tile via URL, run inference, extract detections.
    Returns list of detection dicts (in WGS-84) or empty list."""
    from scipy.ndimage import label as ndimage_label

    try:
        src = rasterio.open(tile_href)
    except Exception as e:
        logger.warning("  Failed to open tile: %s", e)
        return []

    H, W = src.height, src.width
    n_bands = src.count
    transform = src.transform
    crs = src.crs
    resolution = abs(transform.a)

    # Generate tile grid
    step = patch_size - overlap
    tiles = [(r, c) for r in range(0, H - patch_size + 1, step)
             for c in range(0, W - patch_size + 1, step)]

    if not tiles:
        src.close()
        return []

    # Run inference
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float32)
    batch_tensors, batch_coords = [], []

    bands_to_read = [1, 2, 3, 4] if n_bands >= 4 else list(range(1, n_bands + 1))

    for row, col in tiles:
        window = Window(col, row, patch_size, patch_size)
        data = src.read(bands_to_read, window=window).astype(np.float32)
        if data.shape[0] < 4:
            data = np.concatenate(
                [data, np.zeros((4 - data.shape[0], patch_size, patch_size),
                                dtype=np.float32)], axis=0)
        if data.max() > 1.0:
            data /= 255.0

        t = (torch.from_numpy(data) - mean_t) / std_t
        batch_tensors.append(t)
        batch_coords.append((row, col))

        if len(batch_tensors) == batch_size or (row, col) == tiles[-1]:
            b = torch.stack(batch_tensors).to(device)
            probs = torch.sigmoid(model(b)).squeeze(1).cpu().numpy()
            for j, (r, c) in enumerate(batch_coords):
                prob_sum[r:r + patch_size, c:c + patch_size] += probs[j]
                count[r:r + patch_size, c:c + patch_size] += 1
            batch_tensors.clear()
            batch_coords.clear()

    src.close()

    count = np.maximum(count, 1)
    prob = (prob_sum / count).astype(np.float32)
    binary = (prob >= threshold).astype(np.uint8)

    if binary.sum() == 0:
        return []

    # Connected components → detections
    labeled, n_components = ndimage_label(binary)
    min_px = min_area_m2 / (resolution ** 2)
    detections = []

    # Set up CRS transformer
    geo_transformer = None
    if crs and not rasterio.crs.CRS(crs).is_geographic:
        from pyproj import Transformer
        geo_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    for comp in range(1, n_components + 1):
        mask = labeled == comp
        area_px = mask.sum()
        if area_px < min_px:
            continue

        rows_with = np.where(mask.any(axis=1))[0]
        cols_with = np.where(mask.any(axis=0))[0]
        r0, r1 = rows_with[0], rows_with[-1]
        c0, c1 = cols_with[0], cols_with[-1]

        cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
        gx = transform.c + cx * transform.a
        gy = transform.f + cy * transform.e

        if geo_transformer:
            lon, lat = geo_transformer.transform(gx, gy)
        else:
            lon, lat = gx, gy

        detections.append({
            "center_lon": float(lon),
            "center_lat": float(lat),
            "positive_pixels": int(area_px),
            "area_m2": float(area_px * resolution * resolution),
            "mean_confidence": float(prob[mask].mean()),
            "max_confidence": float(prob[mask].max()),
            "bbox_width_m": float((c1 - c0 + 1) * resolution),
            "bbox_height_m": float((r1 - r0 + 1) * resolution),
        })

    return detections


def deduplicate_detections(detections, min_dist_m=100):
    """Remove duplicate detections near tile boundaries.
    Keeps the one with higher confidence."""
    if len(detections) < 2:
        return detections

    from scipy.spatial import KDTree

    # Convert to approximate meters for distance calc
    coords = np.array([
        [d["center_lon"] * 85000, d["center_lat"] * 111000]
        for d in detections
    ])
    tree = KDTree(coords)

    keep = np.ones(len(detections), dtype=bool)
    for i in range(len(detections)):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], r=min_dist_m)
        for j in neighbors:
            if j <= i or not keep[j]:
                continue
            # Keep higher confidence
            if detections[j]["mean_confidence"] > detections[i]["mean_confidence"]:
                keep[i] = False
                break
            else:
                keep[j] = False

    n_removed = (~keep).sum()
    if n_removed > 0:
        logger.info("  Deduplication: removed %d near-boundary duplicates", n_removed)

    return [d for d, k in zip(detections, keep) if k]


def run_tiled_detection(model, items, device, mean, std,
                        threshold=0.5, batch_size=8, min_area_m2=200,
                        overlap=64):
    """Process all NAIP tiles one at a time. Returns list of detection dicts."""
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    all_detections = []
    tiles_with_detections = 0

    for i, item in enumerate(tqdm(items, desc="  Processing tiles")):
        href = item.assets["image"].href
        tile_id = item.id

        dets = detect_in_tile(
            model, href, device, mean_t, std_t,
            patch_size=512, overlap=overlap, threshold=threshold,
            batch_size=batch_size, min_area_m2=min_area_m2,
        )

        if dets:
            tiles_with_detections += 1
            all_detections.extend(dets)
            logger.debug("  Tile %s: %d detections", tile_id, len(dets))

    logger.info("  Tiles processed: %d, tiles with detections: %d",
                len(items), tiles_with_detections)
    logger.info("  Raw detections (before dedup): %d", len(all_detections))

    # Deduplicate near tile boundaries
    all_detections = deduplicate_detections(all_detections, min_dist_m=100)
    logger.info("  Final detections: %d", len(all_detections))

    return all_detections


# ================================================================
# Single-raster detection (for --naip-raster)
# ================================================================

@torch.no_grad()
def run_single_raster_detection(model, raster_path, device, mean, std,
                                threshold=0.5, batch_size=8, min_area_m2=200,
                                overlap=64):
    """Run detection on a single local raster file."""
    from scipy.ndimage import label as ndimage_label

    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        n_bands = src.count
        transform = src.transform
        crs = src.crs
        resolution = abs(transform.a)

        step = 512 - overlap
        tiles = [(r, c) for r in range(0, H - 512 + 1, step)
                 for c in range(0, W - 512 + 1, step)]
        logger.info("  Raster: %d×%d, %d bands, %d tiles", W, H, n_bands, len(tiles))

        prob_sum = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float32)
        batch_t, batch_c = [], []
        bands = [1, 2, 3, 4] if n_bands >= 4 else list(range(1, n_bands + 1))

        for row, col in tqdm(tiles, desc="  Predicting"):
            window = Window(col, row, 512, 512)
            data = src.read(bands, window=window).astype(np.float32)
            if data.shape[0] < 4:
                data = np.concatenate(
                    [data, np.zeros((4 - data.shape[0], 512, 512),
                                    dtype=np.float32)], axis=0)
            if data.max() > 1.0:
                data /= 255.0
            t = (torch.from_numpy(data) - mean_t) / std_t
            batch_t.append(t)
            batch_c.append((row, col))

            if len(batch_t) == batch_size or (row, col) == tiles[-1]:
                b = torch.stack(batch_t).to(device)
                probs = torch.sigmoid(model(b)).squeeze(1).cpu().numpy()
                for j, (r, c) in enumerate(batch_c):
                    prob_sum[r:r + 512, c:c + 512] += probs[j]
                    count[r:r + 512, c:c + 512] += 1
                batch_t.clear()
                batch_c.clear()

    count = np.maximum(count, 1)
    prob = (prob_sum / count).astype(np.float32)
    binary = (prob >= threshold).astype(np.uint8)

    # Connected components
    labeled, n_components = ndimage_label(binary)
    min_px = min_area_m2 / (resolution ** 2)

    geo_transformer = None
    if crs and not rasterio.crs.CRS(crs).is_geographic:
        from pyproj import Transformer
        geo_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    detections = []
    for comp in range(1, n_components + 1):
        mask = labeled == comp
        area_px = mask.sum()
        if area_px < min_px:
            continue
        rows_w = np.where(mask.any(axis=1))[0]
        cols_w = np.where(mask.any(axis=0))[0]
        cy, cx = (rows_w[0] + rows_w[-1]) / 2, (cols_w[0] + cols_w[-1]) / 2
        gx = transform.c + cx * transform.a
        gy = transform.f + cy * transform.e
        if geo_transformer:
            lon, lat = geo_transformer.transform(gx, gy)
        else:
            lon, lat = gx, gy

        detections.append({
            "center_lon": float(lon),
            "center_lat": float(lat),
            "positive_pixels": int(area_px),
            "area_m2": float(area_px * resolution * resolution),
            "mean_confidence": float(prob[mask].mean()),
            "max_confidence": float(prob[mask].max()),
            "bbox_width_m": float((cols_w[-1] - cols_w[0] + 1) * resolution),
            "bbox_height_m": float((rows_w[-1] - rows_w[0] + 1) * resolution),
        })

    logger.info("  %d detections from %d components (min area %.0f m²)",
                len(detections), n_components, min_area_m2)
    return detections


# ================================================================
# Detection → metadata.csv
# ================================================================

def detections_to_metadata(detections, county_geoid):
    """Convert detection list to optimizer-compatible DataFrame."""
    rows = []
    for i, d in enumerate(detections):
        rows.append({
            "patch_id": f"det_{i:05d}",
            "label": "positive",
            "center_lon": d["center_lon"],
            "center_lat": d["center_lat"],
            "county_geoid": int(county_geoid),
            "positive_pixels": d["positive_pixels"],
            "area_m2": d["area_m2"],
            "mean_confidence": d["mean_confidence"],
            "max_confidence": d["max_confidence"],
            "bbox_width_m": d["bbox_width_m"],
            "bbox_height_m": d["bbox_height_m"],
            "voltage": "",
            "substation_type": "detected",
            "substation_name": "",
            "operator": "",
            "osm_id": -1,
            "geom_source": "model",
            "source": "model_prediction",
            "split": "prediction",
        })
    return pd.DataFrame(rows)


def create_detection_map(detections_df, county_name, output_path):
    """Interactive Folium map of detections."""
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        logger.warning("folium not installed — skipping map")
        return

    if detections_df.empty:
        return

    center = [detections_df["center_lat"].mean(), detections_df["center_lon"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False,
    ).add_to(m)

    cluster = MarkerCluster(name="Detected Substations")
    for _, r in detections_df.iterrows():
        color = ("#f44336" if r["mean_confidence"] > 0.8 else
                 "#ff9800" if r["mean_confidence"] > 0.6 else "#ffeb3b")
        popup = (f"<b>{r['patch_id']}</b><br>"
                 f"Conf: {r['mean_confidence']:.2f}<br>"
                 f"Area: {r['area_m2']:.0f} m²")
        folium.CircleMarker(
            [r["center_lat"], r["center_lon"]],
            radius=max(4, min(15, r["area_m2"] / 500)),
            color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(cluster)

    cluster.add_to(m)
    legend = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
              f'background:rgba(0,0,0,0.8);color:white;padding:12px;'
              f'border-radius:6px;font:12px monospace;">'
              f'<b>{county_name}</b><br>{len(detections_df)} detections<br>'
              f'<span style="color:#f44336">●</span> High (&gt;0.8) '
              f'<span style="color:#ff9800">●</span> Med '
              f'<span style="color:#ffeb3b">●</span> Low</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)
    m.save(str(output_path))


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser(
        description="Full pipeline: detect substations → optimize placement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--county", required=True, choices=list(COUNTY_PRESETS.keys()),
                   help="County key (e.g. maricopa, allegheny, fresno)")

    # Detection
    p.add_argument("--model-path", default="model/final_model.pt")
    p.add_argument("--naip-raster", default=None,
                   help="Skip STAC search — use this local raster instead")
    p.add_argument("--channel-stats", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area-m2", type=float, default=200)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--year-min", type=int, default=2020)
    p.add_argument("--year-max", type=int, default=2025)
    p.add_argument("--max-tiles", type=int, default=200,
                   help="Max NAIP tiles to process (default: 200)")

    # Skip flags
    p.add_argument("--skip-detection", action="store_true",
                   help="Skip detection — use existing --metadata instead")
    p.add_argument("--metadata", default=None,
                   help="Path to existing metadata.csv (with --skip-detection)")

    # Optimizer
    p.add_argument("--grid-cell-size", type=float, default=3000.0)
    p.add_argument("--build-cost", type=float, default=1.5e6)
    p.add_argument("--max-new", type=int, default=50)
    p.add_argument("--max-radius", type=float, default=20000.0)
    p.add_argument("--max-coverage-dist", type=float, default=15000.0)
    p.add_argument("--time-limit", type=int, default=300)
    p.add_argument("--sensitivity", action="store_true")

    p.add_argument("-o", "--output", default="output")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers
    for _noisy in ["rasterio", "fiona", "matplotlib", "urllib3", "botocore",
                   "requests", "shapely", "pyproj", "PIL", "GDAL",
                   "rasterio._env", "rasterio.env", "rasterio._io"]:
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    preset = COUNTY_PRESETS[args.county]
    county_name = preset["county_name"]
    county_geoid = preset["geoid"]
    state_name = preset["state_name"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("=" * 70)
    print(f"  SUBSTATION DETECTION + SITING PIPELINE")
    print(f"  County:  {county_name} (GEOID {county_geoid})")
    print(f"  Output:  {out}")
    print("=" * 70)

    # ==============================================================
    # DETECTION PHASE
    # ==============================================================

    if args.skip_detection:
        metadata_path = Path(args.metadata) if args.metadata else out / "metadata.csv"
        if not metadata_path.exists():
            logger.error("Metadata file not found: %s", metadata_path)
            sys.exit(1)
        detections_df = pd.read_csv(metadata_path)
        n_det = (detections_df["label"] == "positive").sum()
        print(f"\n[SKIP] Using existing metadata: {metadata_path} ({n_det} detections)")

    else:
        # Channel stats
        if args.channel_stats and Path(args.channel_stats).exists():
            with open(args.channel_stats) as f:
                s = json.load(f)
            mean = np.array(s["mean"], dtype=np.float32)
            std = np.array(s["std"], dtype=np.float32)
        else:
            mean = np.array([0.485, 0.456, 0.406, 0.456], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225, 0.224], dtype=np.float32)

        # Load model
        import segmentation_models_pytorch as smp
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=4, classes=1, activation=None)
        model.load_state_dict(
            torch.load(args.model_path, map_location=device, weights_only=True))
        model.to(device).eval()
        print(f"  Model loaded → {device}")

        if args.naip_raster:
            # Single local raster
            print(f"\n[1/5] Using local raster: {args.naip_raster}")
            print(f"\n[2/5] Running detection on raster...")
            detections = run_single_raster_detection(
                model, args.naip_raster, device, mean, std,
                threshold=args.threshold, batch_size=args.batch_size,
                min_area_m2=args.min_area_m2, overlap=args.overlap,
            )
        else:
            # Stream NAIP tiles from STAC (no download, no merge)
            print(f"\n[1/5] Searching NAIP tiles...")
            items, bbox = search_naip_tiles(
                county_name, state_name,
                year_min=args.year_min, year_max=args.year_max,
                max_tiles=args.max_tiles,
            )

            print(f"\n[2/5] Running per-tile detection ({len(items)} tiles)...")
            print(f"  Each tile is streamed, processed, and discarded.")
            print(f"  No giant mosaic is created — memory stays bounded.")
            detections = run_tiled_detection(
                model, items, device, mean, std,
                threshold=args.threshold, batch_size=args.batch_size,
                min_area_m2=args.min_area_m2, overlap=args.overlap,
            )

        # Convert to metadata DataFrame
        print(f"\n[3/5] Building metadata.csv...")
        detections_df = detections_to_metadata(detections, county_geoid)
        metadata_path = out / "metadata.csv"
        detections_df.to_csv(metadata_path, index=False)
        logger.info("  Saved: %s (%d detections)", metadata_path, len(detections_df))

        # GeoJSON + map
        if not detections_df.empty:
            det_gdf = gpd.GeoDataFrame(
                detections_df,
                geometry=gpd.points_from_xy(
                    detections_df["center_lon"], detections_df["center_lat"]),
                crs="EPSG:4326",
            )
            det_gdf.to_file(out / "detections.geojson", driver="GeoJSON")
            logger.info("  Saved: %s", out / "detections.geojson")
        create_detection_map(detections_df, county_name,
                             out / "detection_map.html")

        # Summary stats
        if not detections_df.empty:
            areas = detections_df["area_m2"]
            confs = detections_df["mean_confidence"]
            print(f"\n  Detection summary:")
            print(f"    Count:       {len(detections_df)}")
            print(f"    Area range:  {areas.min():.0f} – {areas.max():.0f} m²")
            print(f"    Confidence:  {confs.min():.3f} – {confs.max():.3f}")
            print(f"    Median area: {areas.median():.0f} m²")

    # ==============================================================
    # OPTIMIZATION PHASE
    # ==============================================================

    if detections_df.empty or (detections_df["label"] == "positive").sum() == 0:
        print("\n[!] No detections found — cannot run optimizer.")
        sys.exit(0)

    metadata_path = out / "metadata.csv"
    if not metadata_path.exists():
        detections_df.to_csv(metadata_path, index=False)

    n_pos = (detections_df["label"] == "positive").sum()
    print(f"\n[4/5] Running p-median optimizer...")
    print(f"  Input: {n_pos} detected substations")
    print(f"  Parameters: max_new={args.max_new}, "
          f"build_cost=${args.build_cost:,.0f}, grid={args.grid_cell_size:.0f}m")

    opt_config = OptimizerConfig(
        county=args.county,
        metadata_path=metadata_path,
        grid_cell_size_m=args.grid_cell_size,
        fixed_build_cost=args.build_cost,
        max_new_substations=args.max_new,
        max_service_radius_m=args.max_radius,
        max_coverage_dist_m=args.max_coverage_dist,
        solver_time_limit_s=args.time_limit,
    )

    summary = run_optimizer(opt_config, save_outputs=True, visualize=True)

    # Sensitivity (optional)
    if args.sensitivity:
        print(f"\n[5/5] Running sensitivity analysis...")
        run_sensitivity_analysis(opt_config)

    elapsed = time.time() - start_time

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================

    print()
    print("=" * 70)
    print(f"  PIPELINE COMPLETE — {county_name}")
    print("=" * 70)
    print(f"  Detected substations:  {summary.n_existing_substations}")
    print(f"  New substations built: {summary.n_new_substations}")
    print(f"  Load shifted (kW):     {summary.total_new_load_served_kw:,.0f}")
    print(f"  New coverage (km²):    {summary.new_coverage_area_km2:,.1f}")
    print(f"  Avg service dist:      {summary.avg_service_dist_before_m:,.0f}m → "
          f"{summary.avg_service_dist_after_m:,.0f}m")
    print(f"  Coverage gap:          {summary.coverage_gap_before_pct:.1f}% → "
          f"{summary.coverage_gap_after_pct:.1f}%")
    print(f"  TX-constrained cells:  {summary.n_transmission_constrained}")
    print(f"  Elapsed time:          {elapsed:.1f}s")
    print("-" * 70)
    print(f"  Outputs in {out}/:")
    if not args.skip_detection:
        print(f"    metadata.csv                  ← detected substations")
        print(f"    detections.geojson            ← QGIS")
        print(f"    detection_map.html            ← interactive map")
    print(f"    output/{args.county}_new_substations.csv  ← build recommendations")
    print(f"    output/{args.county}_tx_expansion.csv     ← TX expansion spots")
    print(f"    output/{args.county}_results_map.png      ← results map")
    print(f"    output/{args.county}_coverage_map.png     ← coverage heatmap")
    if args.sensitivity:
        print(f"    output/{args.county}_sensitivity.csv/png")
    print("=" * 70)


if __name__ == "__main__":
    main()
