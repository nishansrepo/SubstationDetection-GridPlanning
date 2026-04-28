#!/usr/bin/env python3
"""
Local Dataset Pipeline: detect substations → optimize new placement.

Instead of streaming NAIP tiles from the cloud, this script uses a local
eval_dataset. It reads the metadata CSV, finds patches for the target county,
runs the ensemble model on them, and feeds the detections into the optimizer.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import torch
from scipy.ndimage import label as ndimage_label
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.config import COUNTY_PRESETS, OptimizerConfig
from optimizer.pipeline import run_optimizer
from optimizer.sensitivity import run_sensitivity_analysis

logger = logging.getLogger(__name__)


# ================================================================
# Local Batch Inference
# ================================================================

@torch.no_grad()
def process_local_patches(models, patch_paths, device, mean_t, std_t,
                          threshold=0.5, batch_size=64, min_area_m2=200):
    """Run ensemble inference on a list of local 512x512 TIFF patches."""
    detections = []
    
    # Process in batches
    for i in tqdm(range(0, len(patch_paths), batch_size), desc="  Predicting Patches"):
        batch_paths = patch_paths[i : i + batch_size]
        batch_tensors = []
        batch_transforms = []
        batch_crs = []
        valid_paths = []

        # Read images
        for path in batch_paths:
            try:
                with rasterio.open(path) as src:
                    n_bands = src.count
                    transform = src.transform
                    crs = src.crs
                    
                    bands_to_read = [1, 2, 3, 4] if n_bands >= 4 else list(range(1, n_bands + 1))
                    data = src.read(bands_to_read).astype(np.float32)
                    
                    # Pad bands if less than 4
                    if data.shape[0] < 4:
                        zeros_band = np.zeros((4 - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
                        data = np.concatenate([data, zeros_band], axis=0)

                    # Normalize
                    if data.max() > 1.0:
                        data /= 255.0

                    # Pad spatially if not exactly 512x512
                    if data.shape[1] < 512 or data.shape[2] < 512:
                        pad_y = max(0, 512 - data.shape[1])
                        pad_x = max(0, 512 - data.shape[2])
                        data = np.pad(data, ((0, 0), (0, pad_y), (0, pad_x)), mode='constant')
                    else:
                        data = data[:, :512, :512]

                    t = (torch.from_numpy(data) - mean_t) / std_t
                    batch_tensors.append(t)
                    batch_transforms.append(transform)
                    batch_crs.append(crs)
                    valid_paths.append(path)
            except Exception as e:
                logger.warning(f"  Failed to read {path.name}: {e}")

        if not batch_tensors:
            continue

        # GPU Inference
        b = torch.stack(batch_tensors).to(device)
        ensemble_probs = 0  # PyTorch will automatically shape this to (64, 512, 512)
        
        with torch.autocast(device_type=device.type):
            for m in models:
                ensemble_probs += torch.sigmoid(m(b)).squeeze(1)
        
        probs = (ensemble_probs / len(models)).cpu().to(torch.float32).numpy()

        # Extract Detections
        for j, prob_map in enumerate(probs):
            binary = (prob_map >= threshold).astype(np.uint8)
            if binary.sum() == 0:
                continue

            transform = batch_transforms[j]
            crs = batch_crs[j]
            resolution = abs(transform.a)
            min_px = min_area_m2 / (resolution ** 2)

            labeled, n_components = ndimage_label(binary)

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
                    "patch_id": valid_paths[j].stem,
                    "center_lon": float(lon),
                    "center_lat": float(lat),
                    "positive_pixels": int(area_px),
                    "area_m2": float(area_px * resolution * resolution),
                    "mean_confidence": float(prob_map[mask].mean()),
                    "max_confidence": float(prob_map[mask].max()),
                    "bbox_width_m": float((c1 - c0 + 1) * resolution),
                    "bbox_height_m": float((r1 - r0 + 1) * resolution),
                })

    return detections


def deduplicate_detections(detections, min_dist_m=100):
    if len(detections) < 2:
        return detections

    from scipy.spatial import KDTree
    coords = np.array([[d["center_lon"] * 85000, d["center_lat"] * 111000] for d in detections])
    tree = KDTree(coords)
    keep = np.ones(len(detections), dtype=bool)

    for i in range(len(detections)):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], r=min_dist_m)
        for j in neighbors:
            if j <= i or not keep[j]:
                continue
            if detections[j]["mean_confidence"] > detections[i]["mean_confidence"]:
                keep[i] = False
                break
            else:
                keep[j] = False

    n_removed = (~keep).sum()
    if n_removed > 0:
        logger.info("  Deduplication: removed %d close-proximity duplicates", n_removed)

    return [d for d, k in zip(detections, keep) if k]


def detections_to_metadata(detections, county_geoid):
    rows = []
    for d in detections:
        rows.append({
            "patch_id": d["patch_id"],
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
            "source": "local_inference",
            "split": "prediction",
        })
    return pd.DataFrame(rows)


def create_detection_map(detections_df, county_name, output_path):
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        return

    if detections_df.empty:
        return

    center = [detections_df["center_lat"].mean(), detections_df["center_lon"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False,
    ).add_to(m)

    cluster = MarkerCluster(name="Detected Substations")
    for _, r in detections_df.iterrows():
        color = ("#f44336" if r["mean_confidence"] > 0.8 else "#ff9800" if r["mean_confidence"] > 0.6 else "#ffeb3b")
        popup = (f"<b>{r['patch_id']}</b><br>Conf: {r['mean_confidence']:.2f}<br>Area: {r['area_m2']:.0f} m²")
        folium.CircleMarker(
            [r["center_lat"], r["center_lon"]],
            radius=max(4, min(15, r["area_m2"] / 500)),
            color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(cluster)

    cluster.add_to(m)
    m.save(str(output_path))


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser(description="Local Eval Pipeline: detect from local patches → optimize.")
    p.add_argument("--county", required=True, choices=list(COUNTY_PRESETS.keys()))
    
    # Dataset args
    p.add_argument("--eval-csv", default="../eval_dataset/eval_dataset_metadata.csv")
    p.add_argument("--img-dir", default="../eval_dataset/images", help="Directory where patch .tif files live")
    
    # Model args
    p.add_argument("--model-path", default="model/final_model.pt")
    p.add_argument("--channel-stats", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area-m2", type=float, default=200)
    p.add_argument("--batch-size", type=int, default=64)

    # Optimizer args
    p.add_argument("--grid-cell-size", type=float, default=3000.0)
    p.add_argument("--build-cost", type=float, default=1.5e6)
    p.add_argument("--max-new", type=int, default=50)
    p.add_argument("--max-radius", type=float, default=20000.0)
    p.add_argument("--max-coverage-dist", type=float, default=15000.0)
    p.add_argument("--time-limit", type=int, default=300)
    p.add_argument("--bbox", type=float, nargs=4, default=None,
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                   help="Analysis bounding box in WGS-84 (overrides preset bbox)")
    p.add_argument("--no-capacity", action="store_true",
                   help="Skip capacity constraints (faster solve, recommended for large areas)")
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--skip-analysis", action="store_true",
                   help="Skip site suitability analysis (reverse geocoding + OSM check)")

    p.add_argument("-o", "--output", default="output")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # Configure logging: our code gets verbose output, third-party libs stay quiet
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    # Silence noisy third-party loggers
    for noisy in ["rasterio", "fiona", "matplotlib", "urllib3", "botocore",
                   "requests", "shapely", "pyproj", "PIL", "GDAL",
                   "rasterio._env", "rasterio.env", "rasterio._io"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    preset = COUNTY_PRESETS[args.county]
    county_name = preset["county_name"]
    county_geoid = preset["geoid"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print("=" * 70)
    print(f"  LOCAL EVAL PIPELINE: {county_name} (GEOID {county_geoid})")
    print("=" * 70)

    # 1. Read Metadata & Find Local Patches
    csv_path = Path(args.eval_csv)
    img_dir = Path(args.img_dir)
    
    if not csv_path.exists():
        print(f"[!] Could not find CSV at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    # Filter for the target county
    county_df = df[df["county_geoid"] == int(county_geoid)]
    
    if county_df.empty:
        print(f"[!] No entries found in CSV for geoid {county_geoid} ({county_name})")
        sys.exit(1)

    print(f"  Found {len(county_df)} potential patches for {county_name} in CSV.")

    # Locate the physical .tif files
    patch_paths = []
    for patch_id in county_df["patch_id"]:
        # Tries standard root, or recursive search if placed in train/test subfolders
        direct_path = img_dir / f"{patch_id}.tif"
        if direct_path.exists():
            patch_paths.append(direct_path)
        else:
            # Fallback recursive search
            found = list(img_dir.rglob(f"{patch_id}.tif"))
            if found:
                patch_paths.append(found[0])
                
    print(f"  Located {len(patch_paths)} .tif files on disk to process.")
    
    if not patch_paths:
        print("[!] No images found matching the IDs. Check your --img-dir argument.")
        sys.exit(1)

    # 2. Load Model & Process
    if args.channel_stats and Path(args.channel_stats).exists():
        with open(args.channel_stats) as f:
            s = json.load(f)
        mean_t = torch.tensor(s["mean"], dtype=torch.float32).view(-1, 1, 1)
        std_t = torch.tensor(s["std"], dtype=torch.float32).view(-1, 1, 1)
    else:
        mean_t = torch.tensor([0.485, 0.456, 0.406, 0.456], dtype=torch.float32).view(-1, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225, 0.224], dtype=torch.float32).view(-1, 1, 1)

    import segmentation_models_pytorch as smp
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
        
    print(f"\n[1/4] Loading ensemble weights → {device}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    ensemble_models = []
    
    fold_keys = [k for k in checkpoint.keys() if isinstance(k, str) and "fold" in k.lower()]
    if "folds" in checkpoint:
        for fold_data in checkpoint["folds"]:
            m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
            m.load_state_dict(fold_data.get("state_dict", fold_data.get("model_state_dict", fold_data)))
            m.to(device).eval()
            ensemble_models.append(m)
    elif len(fold_keys) > 0:
        for k in sorted(fold_keys):
            m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
            m.load_state_dict(checkpoint[k].get("state_dict", checkpoint[k].get("model_state_dict", checkpoint[k])))
            m.to(device).eval()
            ensemble_models.append(m)
    else:
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
        m.load_state_dict(checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint)))
        m.to(device).eval()
        ensemble_models.append(m)

    print(f"\n[2/4] Running inference on {len(patch_paths)} local patches...")
    detections = process_local_patches(
        ensemble_models, patch_paths, device, mean_t, std_t,
        threshold=args.threshold, batch_size=args.batch_size, min_area_m2=args.min_area_m2
    )

    detections = deduplicate_detections(detections, min_dist_m=100)
    
    print(f"\n[3/4] Building metadata.csv...")
    detections_df = detections_to_metadata(detections, county_geoid)
    metadata_path = out / "metadata.csv"
    detections_df.to_csv(metadata_path, index=False)
    
    if not detections_df.empty:
        det_gdf = gpd.GeoDataFrame(
            detections_df, geometry=gpd.points_from_xy(detections_df["center_lon"], detections_df["center_lat"]), crs="EPSG:4326")
        det_gdf.to_file(out / "detections.geojson", driver="GeoJSON")
    create_detection_map(detections_df, county_name, out / "detection_map.html")
    
    if detections_df.empty:
        print("\n[!] No detections found — cannot run optimizer.")
        sys.exit(0)

    print(f"\n[4/5] Running optimizer...")
    opt_config = OptimizerConfig(
        county=args.county, metadata_path=metadata_path, grid_cell_size_m=args.grid_cell_size,
        fixed_build_cost=args.build_cost, max_new_substations=args.max_new,
        max_service_radius_m=args.max_radius, max_coverage_dist_m=args.max_coverage_dist,
        solver_time_limit_s=args.time_limit, skip_capacity=args.no_capacity,
        bbox=args.bbox,
    )
    summary = run_optimizer(opt_config, save_outputs=True, visualize=True)

    if args.sensitivity:
        run_sensitivity_analysis(opt_config)

    # Site analysis: reverse geocode, land-use validation, satellite map
    if not args.skip_analysis and summary.n_new_substations > 0:
        print(f"\n[5/5] Analyzing recommended sites...")
        from optimizer.site_analysis import (
            analyze_recommended_sites, generate_satellite_map, save_enriched_results
        )
        from optimizer.data_loader import load_model_substations

        enriched = analyze_recommended_sites(summary, opt_config, rate_limit=1.2)
        if not enriched.empty:
            save_enriched_results(enriched, opt_config)

            # Load existing substations for the map
            try:
                existing_gdf = load_model_substations(
                    metadata_path, opt_config.county_geoid, opt_config.crs_proj)
            except Exception:
                existing_gdf = None

            from pathlib import Path as _P
            map_path = _P(__file__).resolve().parent.parent / "output" / f"{args.county}_satellite_map.html"
            generate_satellite_map(
                enriched, existing_gdf,
                summary.transmission_expansion_candidates,
                county_name, map_path, opt_config,
            )

            # Print site analysis summary
            n_suitable = (enriched["suitability"] == "suitable").sum()
            n_restricted = (enriched["suitability"] == "restricted").sum()
            print(f"\n  Site Analysis: {n_suitable} suitable, {n_restricted} restricted "
                  f"out of {len(enriched)} recommended sites")
            for _, r in enriched.iterrows():
                flag = "✓" if r["suitability"] == "suitable" else "⚠" if r["suitability"] == "caution" else "✗"
                print(f"    {flag} #{int(r['candidate_id']):3d}  {r['location_name'][:50]:<50s}  "
                      f"{r['total_load_kw']:>8,.0f} kW  [{r['suitability']}]")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — {county_name} (Elapsed: {elapsed:.1f}s)")
    print("=" * 70)

if __name__ == "__main__":
    main()