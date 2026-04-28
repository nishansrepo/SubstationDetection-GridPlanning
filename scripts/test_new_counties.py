#!/usr/bin/env python3
"""
Evaluate ensemble model on pre-extracted test counties in the new/ folder.

Expects:
    new/
    ├── <county_name_or_id>/
    │   ├── images/
    │   ├── masks/
    │   └── metadata.csv
    ├── ...

Or flat layout:
    new/
    ├── images/
    ├── masks/
    └── metadata.csv

Produces per-county IoU/Dice breakdowns and combined summary with
separate visualizations for every metric and category.

Usage:
    python scripts/test_new_counties.py \\
        --data-dir new \\
        --model-path model/ensemble_model.pt \\
        -o results/new_counties -v
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

logger = logging.getLogger(__name__)


# ================================================================
# Ensemble loader (same as run_evaluation.py)
# ================================================================

def load_ensemble(model_path, device):
    import segmentation_models_pytorch as smp
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    models = []
    fold_keys = [k for k in checkpoint.keys()
                 if isinstance(k, str) and "fold" in k.lower()]
    if "folds" in checkpoint:
        for fd in checkpoint["folds"]:
            m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=4, classes=1)
            m.load_state_dict(fd.get("state_dict", fd.get("model_state_dict", fd)))
            m.to(device).eval()
            models.append(m)
    elif fold_keys:
        for k in sorted(fold_keys):
            m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=4, classes=1)
            m.load_state_dict(checkpoint[k].get("state_dict",
                              checkpoint[k].get("model_state_dict", checkpoint[k])))
            m.to(device).eval()
            models.append(m)
    else:
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=4, classes=1)
        m.load_state_dict(checkpoint.get("state_dict",
                          checkpoint.get("model_state_dict", checkpoint)))
        m.to(device).eval()
        models.append(m)
    logger.info("Loaded %d model(s) from %s", len(models), model_path)
    return models


# ================================================================
# Dataset
# ================================================================

class EvalDataset(Dataset):
    def __init__(self, metadata, images_dir, masks_dir, mean, std):
        self.meta = metadata.reset_index(drop=True)
        self.img_dir = Path(images_dir)
        self.msk_dir = Path(masks_dir)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid = row["patch_id"]
        with rasterio.open(self.img_dir / f"{pid}.tif") as src:
            img = src.read().astype(np.float32)
        if img.shape[0] < 4:
            img = np.concatenate([img, np.zeros((4 - img.shape[0],
                                  img.shape[1], img.shape[2]), dtype=np.float32)])
        img = img[:4, :512, :512]
        if img.max() > 1.0:
            img /= 255.0
        with rasterio.open(self.msk_dir / f"{pid}.tif") as src:
            mask = (src.read(1)[:512, :512] > 0).astype(np.uint8)
        img_t = (torch.from_numpy(img) - self.mean) / self.std
        return {
            "image": img_t,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "patch_id": pid,
            "target": int(row.get("target", int(row["label"] == "positive"))),
        }


# ================================================================
# Ensemble inference
# ================================================================

@torch.no_grad()
def run_inference(models, loader, device, threshold=0.5):
    results = []
    for batch in tqdm(loader, desc="  Inference", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        ensemble_probs = torch.zeros_like(masks)
        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu"):
            for m in models:
                ensemble_probs += torch.sigmoid(m(images))
        probs = ensemble_probs / len(models)
        preds = (probs >= threshold).float()

        for i in range(images.size(0)):
            p = preds[i, 0].cpu().numpy()
            m_arr = masks[i, 0].cpu().numpy()
            prob = probs[i, 0].cpu().float().numpy()
            inter = (p * m_arr).sum()
            ps, ms = p.sum(), m_arr.sum()
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
            })
    return pd.DataFrame(results)


# ================================================================
# Discovery
# ================================================================

def discover_datasets(data_dir):
    data_dir = Path(data_dir)
    datasets = []
    # Per-county subdirs
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        images = subdir / "images"
        masks = subdir / "masks"
        meta_candidates = [subdir / "metadata.csv", subdir / "raw" / "metadata.csv"]
        meta = next((p for p in meta_candidates if p.exists()), None)
        if images.exists() and masks.exists() and meta:
            datasets.append((subdir.name, images, masks, meta))
    # Flat
    if not datasets:
        images = data_dir / "images"
        masks = data_dir / "masks"
        meta = next((p for p in [data_dir / "metadata.csv",
                                  data_dir / "raw" / "metadata.csv"] if p.exists()), None)
        if images.exists() and masks.exists() and meta:
            datasets.append((data_dir.name, images, masks, meta))
    return datasets


# ================================================================
# Visualization
# ================================================================

PALETTE = {"iou": "#2E5D8A", "dice": "#8B3A62"}


def _dual_bar(df, label_col, count_col, title_prefix, path_prefix):
    if df.empty:
        return
    n = len(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(3.5, n * 0.55)),
                                    sharey=True)
    labels = [f"{r[label_col]}  (n={r[count_col]})" for _, r in df.iterrows()]
    y = range(n)
    for ax, metric, color in [(ax1, "mean_iou", PALETTE["iou"]),
                               (ax2, "mean_dice", PALETTE["dice"])]:
        vals = df[metric].values
        std_col = "std_iou" if "iou" in metric else "std_dice"
        stds = df.get(std_col, pd.Series([0]*n)).fillna(0).values
        bars = ax.barh(y, vals, xerr=stds, color=color, alpha=0.85, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        label_name = "IoU" if "iou" in metric else "Dice"
        ax.set_xlabel(f"Mean {label_name}")
        ax.set_title(f"{title_prefix} — {label_name}", fontweight="bold")
        ax.set_xlim(0, 1)
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{path_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _distribution_plot(results, path, title_suffix=""):
    pos = results[results["target"] == 1]
    if pos.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, col, color, label in [(ax1, "iou", PALETTE["iou"], "IoU"),
                                   (ax2, "dice", PALETTE["dice"], "Dice")]:
        vals = pos[col]
        ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(vals.mean(), color="red", ls="--", label=f"Mean: {vals.mean():.3f}")
        ax.axvline(vals.median(), color="orange", ls="--", label=f"Median: {vals.median():.3f}")
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.set_title(f"{label} Distribution {title_suffix}", fontweight="bold")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _summary_bar(summary_df, output_dir):
    """IoU + Dice bar chart for all counties across all datasets."""
    if summary_df.empty:
        return
    n = len(summary_df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n * 0.6)), sharey=True)
    labels = [f"{r['name']}  (n={r['n_positive']})" for _, r in summary_df.iterrows()]
    y = range(n)

    for ax, metric, color, label_name in [
        (ax1, "mean_iou", PALETTE["iou"], "IoU"),
        (ax2, "mean_dice", PALETTE["dice"], "Dice"),
    ]:
        vals = summary_df[metric].values
        stds = summary_df.get(f"std_{label_name.lower()}", pd.Series([0]*n)).fillna(0).values
        colors = ["#3fb950" if v >= 0.5 else "#d29922" if v >= 0.3 else "#f85149"
                  for v in vals] if metric == "mean_iou" else [color] * n
        bars = ax.barh(y, vals, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f"Mean {label_name}")
        ax.set_xlim(0, 1)
        ax.set_title(f"Test Counties — {label_name}", fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "county_iou_dice.png", dpi=150, bbox_inches="tight")
    plt.close()


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser(
        description="Evaluate ensemble model on new test counties.")
    p.add_argument("--data-dir", default="new")
    p.add_argument("--model-path", default="model/ensemble_model.pt")
    p.add_argument("--channel-stats", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("-o", "--output", default="results/new_counties")
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

    datasets = discover_datasets(args.data_dir)
    if not datasets:
        logger.error("No datasets found in %s", args.data_dir)
        sys.exit(1)

    logger.info("Found %d dataset(s):", len(datasets))
    for name, img, msk, meta in datasets:
        logger.info("  %-20s  %d images", name, len(list(img.glob("*.tif"))))

    # Channel stats
    if args.channel_stats and Path(args.channel_stats).exists():
        with open(args.channel_stats) as f:
            s = json.load(f)
        mean = np.array(s["mean"], dtype=np.float32)
        std  = np.array(s["std"],  dtype=np.float32)
    else:
        mean = np.array([0.485, 0.456, 0.406, 0.456], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225, 0.224], dtype=np.float32)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    models = load_ensemble(args.model_path, device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_county_rows = []
    all_type_rows = []
    all_size_rows = []
    all_results_frames = []

    for ds_name, images_dir, masks_dir, metadata_path in datasets:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  Evaluating: %s", ds_name)
        logger.info("=" * 70)

        meta = pd.read_csv(metadata_path)
        meta["target"] = (meta["label"] == "positive").astype(int)
        for col in ["substation_type", "county_name", "geom_source", "voltage"]:
            if col not in meta.columns:
                meta[col] = ""
            meta[col] = meta[col].fillna("").astype(str)
        if "positive_pixels" not in meta.columns:
            meta["positive_pixels"] = 0

        existing = {f.stem for f in images_dir.glob("*.tif")}
        meta = meta[meta["patch_id"].isin(existing)].reset_index(drop=True)
        n_pos = meta["target"].sum()
        n_neg = (1 - meta["target"]).sum()
        logger.info("  %d patches (%d pos, %d neg)", len(meta), n_pos, n_neg)

        if len(meta) == 0:
            continue

        ds = EvalDataset(meta, images_dir, masks_dir, mean, std)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
        results = run_inference(models, dl, device, args.threshold)

        # Merge metadata columns
        results = results.merge(
            meta[["patch_id", "county_name", "substation_type",
                  "positive_pixels", "geom_source"]],
            on="patch_id", how="left")

        out = output_dir / ds_name
        out.mkdir(parents=True, exist_ok=True)
        results.to_csv(out / "per_patch_results.csv", index=False)
        all_results_frames.append(results)

        pos = results[results["target"] == 1]
        summary = {
            "name": ds_name,
            "n_patches": len(results), "n_positive": len(pos),
            "n_negative": len(results) - len(pos),
            "mean_iou": float(pos["iou"].mean()) if len(pos) else 0,
            "std_iou": float(pos["iou"].std()) if len(pos) else 0,
            "mean_dice": float(pos["dice"].mean()) if len(pos) else 0,
            "std_dice": float(pos["dice"].std()) if len(pos) else 0,
            "mean_precision": float(pos["pixel_precision"].mean()) if len(pos) else 0,
            "mean_recall": float(pos["pixel_recall"].mean()) if len(pos) else 0,
        }

        # Patch-level
        try:
            from sklearn.metrics import roc_auc_score, f1_score
            yt = results["target"].values
            ys = results["patch_score"].values
            yp = (ys >= args.threshold).astype(int)
            if len(np.unique(yt)) > 1:
                summary["patch_auc"] = float(roc_auc_score(yt, ys))
            summary["patch_f1"] = float(f1_score(yt, yp))
        except Exception:
            pass

        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        all_summaries.append(summary)

        # County breakdown
        for county, g in pos.groupby("county_name"):
            if not county or len(g) < 1:
                continue
            all_county_rows.append({
                "dataset": ds_name, "county_name": county, "count": len(g),
                "mean_iou": g["iou"].mean(), "std_iou": g["iou"].std(),
                "mean_dice": g["dice"].mean(), "std_dice": g["dice"].std(),
                "mean_precision": g["pixel_precision"].mean(),
                "mean_recall": g["pixel_recall"].mean(),
            })

        # Type breakdown
        for stype, g in pos.groupby("substation_type"):
            if not stype or len(g) < 1:
                continue
            all_type_rows.append({
                "dataset": ds_name, "substation_type": stype, "count": len(g),
                "mean_iou": g["iou"].mean(), "std_iou": g["iou"].std(),
                "mean_dice": g["dice"].mean(), "std_dice": g["dice"].std(),
            })

        # Size breakdown
        bins_def = [0, 200, 1000, 5000, 20000, 300000]
        labels_s = ["tiny (<200)", "small (200-1k)", "medium (1k-5k)",
                     "large (5k-20k)", "very large (>20k)"]
        pos_sz = pos.copy()
        pos_sz["size_bin"] = pd.cut(pos_sz["positive_pixels"], bins=bins_def, labels=labels_s)
        for b, g in pos_sz.groupby("size_bin", observed=True):
            if len(g) == 0:
                continue
            all_size_rows.append({
                "dataset": ds_name, "size_bin": str(b), "count": len(g),
                "mean_iou": g["iou"].mean(), "std_iou": g["iou"].std(),
                "mean_dice": g["dice"].mean(), "std_dice": g["dice"].std(),
            })

        # Per-dataset visualizations
        _distribution_plot(results, out / "iou_dice_distribution.png", f"({ds_name})")

        logger.info("  RESULT: IoU=%.3f±%.3f  Dice=%.3f±%.3f  P=%.3f  R=%.3f",
                    summary["mean_iou"], summary["std_iou"],
                    summary["mean_dice"], summary["std_dice"],
                    summary["mean_precision"], summary["mean_recall"])
        if not all_county_rows:
            continue
        county_df_ds = pd.DataFrame([r for r in all_county_rows if r["dataset"] == ds_name])
        if not county_df_ds.empty:
            county_df_ds.to_csv(out / "metrics_by_county.csv", index=False)
            for _, r in county_df_ds.iterrows():
                logger.info("    %-28s n=%-4d IoU=%.3f  Dice=%.3f",
                            r["county_name"], r["count"], r["mean_iou"], r["mean_dice"])

    if not all_summaries:
        logger.error("No datasets evaluated.")
        sys.exit(1)

    # ---- Combined outputs ----
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / "combined_summary.csv", index=False)
    with open(output_dir / "combined_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    county_df = pd.DataFrame(all_county_rows)
    if not county_df.empty:
        county_df.to_csv(output_dir / "combined_county_summary.csv", index=False)
        _dual_bar(county_df, "county_name", "count",
                  "All Counties", str(output_dir / "all_counties_iou_dice"))

    type_df = pd.DataFrame(all_type_rows)
    if not type_df.empty:
        type_df.to_csv(output_dir / "combined_type_summary.csv", index=False)
        _dual_bar(type_df, "substation_type", "count",
                  "Substation Types", str(output_dir / "all_types_iou_dice"))

    size_df = pd.DataFrame(all_size_rows)
    if not size_df.empty:
        size_df.to_csv(output_dir / "combined_size_summary.csv", index=False)
        _dual_bar(size_df, "size_bin", "count",
                  "Size Bins", str(output_dir / "all_sizes_iou_dice"))

    _summary_bar(summary_df, output_dir)

    # Combined distribution plot from all results
    if all_results_frames:
        all_results = pd.concat(all_results_frames, ignore_index=True)
        _distribution_plot(all_results, output_dir / "combined_iou_dice_distribution.png",
                           "(all new counties)")

    # Final console
    logger.info("")
    logger.info("=" * 70)
    logger.info("  COMBINED RESULTS (%d datasets)", len(all_summaries))
    logger.info("=" * 70)
    for s in all_summaries:
        logger.info("  %-20s  IoU=%.3f±%.3f  Dice=%.3f±%.3f  %d pos  %d neg",
                    s["name"], s["mean_iou"], s["std_iou"],
                    s["mean_dice"], s["std_dice"], s["n_positive"], s["n_negative"])
    if len(all_summaries) > 1:
        avg_iou = np.mean([s["mean_iou"] for s in all_summaries])
        avg_dice = np.mean([s["mean_dice"] for s in all_summaries])
        logger.info("  AVERAGE:  IoU=%.3f  Dice=%.3f", avg_iou, avg_dice)
    logger.info("=" * 70)
    logger.info("All results → %s/", output_dir)


if __name__ == "__main__":
    main()
