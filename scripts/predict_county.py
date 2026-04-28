#!/usr/bin/env python3
"""
Predict substations from a NAIP raster and produce optimizer-compatible output.

Fully generalized — no default county. You specify everything.

Pipeline:
    NAIP raster → tile with overlap → U-Net inference → merge predictions →
    connected components → area filter → metadata.csv + GeoJSON + map

The output metadata.csv is drop-in compatible with the optimizer:
    python -m optimizer --metadata <output>/metadata.csv --county <county_key>

Inputs you provide:
    --model-path:     path to best_model.pt
    --naip-raster:    path to a NAIP GeoTIFF covering your county
    --county-geoid:   5-digit FIPS GEOID (e.g. 42003 for Allegheny)
    --channel-stats:  channel_stats.json from training

Usage:
    python scripts/predict_county.py \
        --model-path /path/to/best_model.pt \
        --naip-raster /path/to/county_naip.tif \
        --county-geoid 42003 \
        --channel-stats /path/to/channel_stats.json \
        -o predictions/my_county -v
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import torch
from shapely.geometry import Point
from tqdm import tqdm

import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


# ================================================================
# Tiling
# ================================================================

def generate_tiles(height, width, patch_size=512, overlap=64):
    step = patch_size - overlap
    tiles = []
    for row in range(0, height - patch_size + 1, step):
        for col in range(0, width - patch_size + 1, step):
            tiles.append((row, col))
    return tiles


# ================================================================
# Inference
# ================================================================

@torch.no_grad()
def run_raster_inference(model, raster_path, device, mean, std,
                         patch_size=512, overlap=64, threshold=0.5,
                         batch_size=8):
    """Tile raster, run model, merge overlapping predictions."""
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        transform = src.transform
        crs = src.crs

        tiles = generate_tiles(H, W, patch_size, overlap)
        logger.info("Raster: %d×%d  res=%.2f m  CRS=%s  → %d tiles",
                    W, H, abs(transform.a), crs, len(tiles))

        prob_sum = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float32)

        batch_tensors, batch_coords = [], []

        for row, col in tqdm(tiles, desc="Predicting"):
            window = Window(col, row, patch_size, patch_size)
            data = src.read([1, 2, 3, 4], window=window).astype(np.float32)
            if data.max() > 1.0:
                data /= 255.0

            t = torch.from_numpy(data)
            t = (t - mean_t) / std_t
            batch_tensors.append(t)
            batch_coords.append((row, col))

            if len(batch_tensors) == batch_size or (row, col) == tiles[-1]:
                batch = torch.stack(batch_tensors).to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

                for j, (r, c) in enumerate(batch_coords):
                    prob_sum[r:r + patch_size, c:c + patch_size] += probs[j]
                    count[r:r + patch_size, c:c + patch_size] += 1

                batch_tensors.clear()
                batch_coords.clear()

    count = np.maximum(count, 1)
    prob_mosaic = (prob_sum / count).astype(np.float32)
    binary = (prob_mosaic >= threshold).astype(np.uint8)

    return prob_mosaic, binary, transform, crs


# ================================================================
# Post-processing
# ================================================================

def extract_detections(binary, prob, transform, crs,
                       min_area_m2=200, resolution=0.6):
    """Connected components → GeoDataFrame of detections."""
    from scipy.ndimage import label as ndimage_label

    labeled, n = ndimage_label(binary)
    logger.info("Connected components: %d", n)

    min_px = min_area_m2 / (resolution ** 2)
    detections = []

    for comp in range(1, n + 1):
        mask = labeled == comp
        area_px = mask.sum()
        if area_px < min_px:
            continue

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]

        # Centroid in pixel coords → geographic
        cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
        gx = transform.c + cx * transform.a
        gy = transform.f + cy * transform.e

        # To WGS-84
        if crs and not rasterio.crs.CRS(crs).is_geographic:
            from pyproj import Transformer
            t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = t.transform(gx, gy)
        else:
            lon, lat = gx, gy

        mean_conf = float(prob[mask].mean())
        area_m2 = area_px * resolution * resolution

        detections.append({
            "center_lon": float(lon),
            "center_lat": float(lat),
            "positive_pixels": int(area_px),
            "area_m2": float(area_m2),
            "mean_confidence": mean_conf,
            "max_confidence": float(prob[mask].max()),
            "bbox_width_m": float((c1 - c0 + 1) * resolution),
            "bbox_height_m": float((r1 - r0 + 1) * resolution),
            "geometry": Point(lon, lat),
        })

    gdf = gpd.GeoDataFrame(detections, crs="EPSG:4326")
    logger.info("Detections after area filter (≥%.0f m²): %d", min_area_m2, len(gdf))
    return gdf


def to_optimizer_metadata(detections, county_geoid, output_path):
    """Write metadata.csv in the exact format the optimizer expects.

    Critical columns for optimizer compatibility:
        label           — must be 'positive'
        center_lon      — WGS-84 longitude
        center_lat      — WGS-84 latitude
        county_geoid    — integer-castable FIPS GEOID
        voltage         — optional, used for capacity estimation
    """
    if detections.empty:
        logger.warning("No detections — writing empty metadata")

    meta = detections.drop(columns=["geometry"], errors="ignore").copy()

    # Required columns
    meta["label"] = "positive"
    meta["county_geoid"] = int(county_geoid)

    # Optional columns the optimizer may read
    meta["patch_id"] = [f"det_{i:05d}" for i in range(len(meta))]
    meta["voltage"] = ""         # unknown from model; optimizer uses default capacity
    meta["substation_type"] = "detected"
    meta["substation_name"] = ""
    meta["operator"] = ""
    meta["osm_id"] = -1
    meta["geom_source"] = "model"
    meta["source"] = "model_prediction"
    meta["split"] = "prediction"

    meta.to_csv(output_path, index=False)
    logger.info("Optimizer-compatible metadata: %s  (%d rows)", output_path, len(meta))
    return meta


def create_map(detections, output_path):
    """Interactive Folium map of detections."""
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        logger.warning("folium not installed — skipping map (pip install folium)")
        return

    if detections.empty:
        return

    center = [detections["center_lat"].mean(), detections["center_lon"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    cluster = MarkerCluster(name="Detected Substations")

    for _, r in detections.iterrows():
        color = "red" if r["mean_confidence"] > 0.7 else "orange" if r["mean_confidence"] > 0.5 else "yellow"
        popup = (f"Conf: {r['mean_confidence']:.2f}<br>"
                 f"Area: {r['area_m2']:.0f} m²<br>"
                 f"{r['bbox_width_m']:.0f}×{r['bbox_height_m']:.0f} m")
        folium.CircleMarker(
            [r["center_lat"], r["center_lon"]],
            radius=max(3, min(12, r["area_m2"] / 500)),
            color=color, fill=True, fill_opacity=0.7, popup=popup,
        ).add_to(cluster)

    cluster.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    logger.info("Interactive map: %s", output_path)


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser(
        description="Predict substations from a NAIP raster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/predict_county.py \\
        --model-path fold_05/best_model.pt \\
        --naip-raster /data/allegheny_naip.tif \\
        --county-geoid 42003 \\
        --channel-stats fold_05/channel_stats.json \\
        -o predictions/allegheny -v

Then feed to the optimizer:
    python -m optimizer --metadata predictions/allegheny/metadata.csv --county allegheny
""",
    )
    p.add_argument("--model-path", required=True, help="Path to best_model.pt")
    p.add_argument("--naip-raster", required=True, help="NAIP GeoTIFF for the county")
    p.add_argument("--county-geoid", required=True,
                   help="5-digit FIPS GEOID (e.g. 42003, 06019)")
    p.add_argument("--channel-stats", default=None, help="channel_stats.json from training")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area-m2", type=float, default=200,
                   help="Min detection area in m² (default: 200)")
    p.add_argument("--overlap", type=int, default=64, help="Tile overlap px (default: 64)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Channel stats
    if args.channel_stats and Path(args.channel_stats).exists():
        with open(args.channel_stats) as f:
            s = json.load(f)
        mean = np.array(s["mean"], dtype=np.float32)
        std = np.array(s["std"], dtype=np.float32)
    else:
        mean = np.array([0.485, 0.456, 0.406, 0.456], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225, 0.224], dtype=np.float32)
        logger.warning("Using default channel stats")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=4, classes=1, activation=None)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    logger.info("Model → %s", device)

    # Resolution from raster
    with rasterio.open(args.naip_raster) as src:
        resolution = abs(src.transform.a)

    # Inference
    prob, binary, transform, crs = run_raster_inference(
        model, args.naip_raster, device, mean, std,
        patch_size=512, overlap=args.overlap,
        threshold=args.threshold, batch_size=args.batch_size,
    )

    # Save probability raster
    with rasterio.open(
        out / "prediction_probability.tif", "w", driver="GTiff",
        height=prob.shape[0], width=prob.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, compress="lzw",
    ) as dst:
        dst.write(prob, 1)

    # Detections
    detections = extract_detections(binary, prob, transform, crs,
                                    min_area_m2=args.min_area_m2,
                                    resolution=resolution)

    # Outputs
    detections.to_file(out / "detections.geojson", driver="GeoJSON")
    to_optimizer_metadata(detections, args.county_geoid, out / "metadata.csv")
    create_map(detections, out / "detection_map.html")

    # Summary
    logger.info("=" * 60)
    logger.info("  RESULTS")
    logger.info("=" * 60)
    logger.info("  Detections: %d", len(detections))
    if not detections.empty:
        logger.info("  Mean confidence: %.3f", detections["mean_confidence"].mean())
        logger.info("  Median area: %.0f m²", detections["area_m2"].median())
        s = (detections["area_m2"] < 1000).sum()
        m = ((detections["area_m2"] >= 1000) & (detections["area_m2"] < 10000)).sum()
        l = (detections["area_m2"] >= 10000).sum()
        logger.info("  Size: %d small, %d medium, %d large", s, m, l)
    logger.info("=" * 60)
    logger.info("Outputs:")
    logger.info("  %s/metadata.csv               ← optimizer input", out)
    logger.info("  %s/detections.geojson          ← QGIS", out)
    logger.info("  %s/detection_map.html          ← browser", out)
    logger.info("  %s/prediction_probability.tif  ← heatmap", out)
    logger.info("")
    logger.info("Next: python -m optimizer --metadata %s/metadata.csv --county <key>", out)


if __name__ == "__main__":
    main()
