#!/usr/bin/env python3
"""
Detailed evaluation of the substation detection model.

Produces per-type, per-county, per-size, per-geometry breakdowns plus
error analysis. Fully generalized — user supplies all paths.

Inputs:
    --model-path:    path to best_model.pt
    --images-dir:    directory of image GeoTIFFs
    --masks-dir:     directory of mask GeoTIFFs
    --metadata:      CSV with patch_id, label, county_name, substation_type,
                     voltage, geom_source, positive_pixels, etc.
    --channel-stats: JSON with mean/std arrays (from training)

Place your evaluation images and masks wherever you like. The script
locates patches by matching patch_id from the metadata to filenames
in the images and masks directories.

Usage:
    python scripts/evaluate_model.py \
        --model-path /path/to/best_model.pt \
        --images-dir /path/to/images \
        --masks-dir /path/to/masks \
        --metadata /path/to/metadata.csv \
        --channel-stats /path/to/channel_stats.json \
        -o results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


# ================================================================
# Dataset
# ================================================================

class EvalDataset(Dataset):
    def __init__(self, metadata, images_dir, masks_dir, mean, std):
        self.meta = metadata.reset_index(drop=True)
        self.img_dir = Path(images_dir)
        self.msk_dir = Path(masks_dir)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid = row["patch_id"]

        with rasterio.open(self.img_dir / f"{pid}.tif") as src:
            img = src.read().astype(np.float32)
        img = np.moveaxis(img, 0, -1)
        if img.max() > 1.0:
            img /= 255.0

        with rasterio.open(self.msk_dir / f"{pid}.tif") as src:
            mask = (src.read(1) > 0).astype(np.uint8)

        img_t = torch.from_numpy(img).permute(2, 0, 1)
        img_t = (img_t - self.mean) / self.std

        return {
            "image": img_t,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "patch_id": pid,
            "target": int(row.get("target", int(row["label"] == "positive"))),
        }


# ================================================================
# Inference
# ================================================================

@torch.no_grad()
def run_inference(model, loader, device, threshold=0.5):
    model.eval()
    results = []
    for batch in tqdm(loader, desc="Inference"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        for i in range(images.size(0)):
            p = preds[i, 0].cpu().numpy()
            m = masks[i, 0].cpu().numpy()
            prob = probs[i, 0].cpu().numpy()

            inter = (p * m).sum()
            ps, ms = p.sum(), m.sum()
            union = ps + ms - inter

            tp, fp, fn = inter, ps - inter, ms - inter
            flat = prob.flatten()
            k = max(1, int(len(flat) * 0.01))
            patch_score = np.sort(flat)[-k:].mean()

            results.append({
                "patch_id": batch["patch_id"][i],
                "target": batch["target"][i].item(),
                "iou": float((inter + 1e-6) / (union + 1e-6)),
                "dice": float((2 * inter + 1e-6) / (ps + ms + 1e-6)),
                "pixel_precision": float(tp / (tp + fp + 1e-6)),
                "pixel_recall": float(tp / (tp + fn + 1e-6)),
                "pred_positive_px": int(ps),
                "gt_positive_px": int(ms),
                "patch_score": float(patch_score),
                "mean_prob": float(prob.mean()),
                "max_prob": float(prob.max()),
            })

    return pd.DataFrame(results)


# ================================================================
# Breakdown analyses
# ================================================================

def grouped_metrics(results, metadata, col, min_n=3):
    merged = results.merge(metadata[["patch_id", col]], on="patch_id", how="left")
    merged[col] = merged[col].fillna("(untagged)").astype(str).replace("", "(untagged)")
    pos = merged[merged["target"] == 1]
    rows = []
    for val, g in pos.groupby(col):
        if len(g) < min_n:
            continue
        rows.append({
            col: val, "count": len(g),
            "mean_iou": g["iou"].mean(), "std_iou": g["iou"].std(),
            "mean_dice": g["dice"].mean(),
            "mean_precision": g["pixel_precision"].mean(),
            "mean_recall": g["pixel_recall"].mean(),
            "median_gt_px": g["gt_positive_px"].median(),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def size_binned_metrics(results, metadata):
    merged = results.merge(metadata[["patch_id", "positive_pixels"]], on="patch_id", how="left")
    pos = merged[merged["target"] == 1].copy()
    bins = [0, 200, 1000, 5000, 20000, 200000]
    labels = ["tiny (<200)", "small (200–1k)", "medium (1k–5k)", "large (5k–20k)", "very large (>20k)"]
    pos["size_bin"] = pd.cut(pos["positive_pixels"], bins=bins, labels=labels)
    rows = []
    for b, g in pos.groupby("size_bin", observed=True):
        if len(g) == 0:
            continue
        rows.append({
            "size_bin": b, "count": len(g),
            "mean_iou": g["iou"].mean(), "std_iou": g["iou"].std(),
            "mean_dice": g["dice"].mean(),
            "mean_precision": g["pixel_precision"].mean(),
            "mean_recall": g["pixel_recall"].mean(),
        })
    return pd.DataFrame(rows)


def error_analysis(results, metadata, n=20):
    merged = results.merge(metadata, on="patch_id", how="left", suffixes=("", "_m"))
    pos = merged[merged["target"] == 1]
    neg = merged[merged["target"] == 0]
    cols_fn = ["patch_id", "iou", "dice", "gt_positive_px", "pred_positive_px",
               "county_name", "substation_type", "voltage", "geom_source"]
    cols_fp = ["patch_id", "patch_score", "pred_positive_px", "county_name",
               "mean_prob", "max_prob"]
    worst_fn = pos.nsmallest(n, "iou")[[c for c in cols_fn if c in pos.columns]]
    worst_fp = neg.nlargest(n, "patch_score")[[c for c in cols_fp if c in neg.columns]]
    return worst_fn, worst_fp


# ================================================================
# Plotting
# ================================================================

def bar_chart(df, val_col, label_col, count_col, title, xlabel, path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.45)))
    labels = [f"{r[label_col]}  (n={r[count_col]})" for _, r in df.iterrows()]
    vals = df[val_col].values
    bars = ax.barh(range(len(labels)), vals, color="#2E5D8A", alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def iou_histogram(results, path):
    pos = results[results["target"] == 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos["iou"], bins=50, color="#2E5D8A", alpha=0.8, edgecolor="white")
    ax.axvline(pos["iou"].mean(), color="red", ls="--", label=f'Mean: {pos["iou"].mean():.3f}')
    ax.axvline(pos["iou"].median(), color="orange", ls="--", label=f'Median: {pos["iou"].median():.3f}')
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("Per-Patch IoU Distribution (Positive Patches)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def pr_scatter(type_df, path):
    if type_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, r in type_df.iterrows():
        ax.scatter(r["mean_recall"], r["mean_precision"], s=r["count"] * 2, alpha=0.7)
        ax.annotate(r["substation_type"], (r["mean_recall"], r["mean_precision"]),
                    fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Mean Pixel Recall")
    ax.set_ylabel("Mean Pixel Precision")
    ax.set_title("Precision vs Recall by Substation Type")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser(description="Detailed model evaluation.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--images-dir", required=True, help="Directory of image GeoTIFFs")
    p.add_argument("--masks-dir", required=True, help="Directory of mask GeoTIFFs")
    p.add_argument("--metadata", required=True, help="CSV with patch attributes")
    p.add_argument("--channel-stats", default=None, help="JSON with mean/std arrays")
    p.add_argument("--threshold", type=float, default=0.5)
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
    img_dir = Path(args.images_dir)
    msk_dir = Path(args.masks_dir)

    # Metadata
    meta = pd.read_csv(args.metadata)
    meta["target"] = (meta["label"] == "positive").astype(int)
    for col in ["substation_type", "county_name", "geom_source", "voltage"]:
        if col not in meta.columns:
            meta[col] = ""
        meta[col] = meta[col].fillna("").astype(str)
    if "positive_pixels" not in meta.columns:
        meta["positive_pixels"] = 0

    existing = {f.stem for f in img_dir.glob("*.tif")}
    meta = meta[meta["patch_id"].isin(existing)].reset_index(drop=True)
    logger.info("Patches on disk: %d  (%d pos, %d neg)",
                len(meta), meta["target"].sum(), (1 - meta["target"]).sum())

    # Channel stats
    if args.channel_stats and Path(args.channel_stats).exists():
        with open(args.channel_stats) as f:
            s = json.load(f)
        mean = np.array(s["mean"], dtype=np.float32)
        std = np.array(s["std"], dtype=np.float32)
    else:
        mean = np.array([0.485, 0.456, 0.406, 0.456], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225, 0.224], dtype=np.float32)
        logger.warning("No channel stats provided; using ImageNet defaults for 4ch")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=4, classes=1, activation=None)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    logger.info("Model loaded → %s", device)

    # Inference
    ds = EvalDataset(meta, img_dir, msk_dir, mean, std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    results = run_inference(model, dl, device, args.threshold)

    # Overall
    pos = results[results["target"] == 1]
    overall = {
        "n_patches": len(results), "n_positive": len(pos),
        "mean_iou": float(pos["iou"].mean()), "std_iou": float(pos["iou"].std()),
        "median_iou": float(pos["iou"].median()),
        "mean_dice": float(pos["dice"].mean()),
        "mean_pixel_precision": float(pos["pixel_precision"].mean()),
        "mean_pixel_recall": float(pos["pixel_recall"].mean()),
    }
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
        yt, ys = results["target"].values, results["patch_score"].values
        yp = (ys >= args.threshold).astype(int)
        overall["patch_auc"] = float(roc_auc_score(yt, ys))
        overall["patch_ap"] = float(average_precision_score(yt, ys))
        overall["patch_f1"] = float(f1_score(yt, yp))
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        overall.update({"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
    except Exception:
        pass

    # Breakdowns
    type_m = grouped_metrics(results, meta, "substation_type")
    county_m = grouped_metrics(results, meta, "county_name")
    geom_m = grouped_metrics(results, meta, "geom_source")
    size_m = size_binned_metrics(results, meta)
    worst_fn, worst_fp = error_analysis(results, meta)

    # Save CSVs
    results.to_csv(out / "per_patch_results.csv", index=False)
    type_m.to_csv(out / "metrics_by_type.csv", index=False)
    county_m.to_csv(out / "metrics_by_county.csv", index=False)
    geom_m.to_csv(out / "metrics_by_geom_source.csv", index=False)
    size_m.to_csv(out / "metrics_by_size.csv", index=False)
    worst_fn.to_csv(out / "worst_false_negatives.csv", index=False)
    worst_fp.to_csv(out / "worst_false_positives.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Print
    logger.info("=" * 65)
    logger.info("OVERALL  IoU=%.3f±%.3f  Dice=%.3f  P=%.3f  R=%.3f",
                overall["mean_iou"], overall["std_iou"], overall["mean_dice"],
                overall["mean_pixel_precision"], overall["mean_pixel_recall"])
    logger.info("-" * 65)
    logger.info("BY TYPE:")
    for _, r in type_m.iterrows():
        logger.info("  %-22s n=%-4d IoU=%.3f  Dice=%.3f  P=%.3f  R=%.3f",
                    r["substation_type"], r["count"], r["mean_iou"],
                    r["mean_dice"], r["mean_precision"], r["mean_recall"])
    logger.info("-" * 65)
    logger.info("BY COUNTY:")
    for _, r in county_m.iterrows():
        logger.info("  %-25s n=%-4d IoU=%.3f±%.3f",
                    r["county_name"], r["count"], r["mean_iou"], r["std_iou"])
    logger.info("-" * 65)
    logger.info("BY SIZE:")
    for _, r in size_m.iterrows():
        logger.info("  %-20s n=%-4d IoU=%.3f±%.3f",
                    r["size_bin"], r["count"], r["mean_iou"], r["std_iou"])
    logger.info("=" * 65)

    # Plots
    bar_chart(type_m, "mean_iou", "substation_type", "count",
              "IoU by Substation Type", "Mean IoU", out / "iou_by_type.png")
    bar_chart(county_m, "mean_iou", "county_name", "count",
              "IoU by County", "Mean IoU", out / "iou_by_county.png")
    iou_histogram(results, out / "iou_distribution.png")
    pr_scatter(type_m, out / "precision_recall_by_type.png")

    logger.info("All outputs saved to %s/", out)


if __name__ == "__main__":
    main()
