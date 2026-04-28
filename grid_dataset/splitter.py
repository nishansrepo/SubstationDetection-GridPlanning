"""
Train / validation / test splitting.

Two methods:
    'geographic' — All patches from a county/region go to one split.
                   Tests true spatial generalization.
    'random'     — Patches shuffled and split regardless of origin.
                   Higher data efficiency but weaker generalization signal.

No counties are hardcoded to any split. Users can optionally assign
specific counties via SplitConfig.test_geoids / val_geoids; otherwise
counties are assigned automatically.
"""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def assign_splits(
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Assign each patch to train/val/test.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must contain 'county_geoid' column.
    config : PipelineConfig
        Split configuration.

    Returns
    -------
    pd.DataFrame with added 'split' column.
    """
    df = metadata.copy()
    rng = np.random.default_rng(config.split.seed)

    if config.split.method == "random":
        return _random_split(df, config, rng)
    else:
        return _geographic_split(df, config, rng)


def _geographic_split(
    df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Split by county/region — all patches from one source go together."""
    geoids = sorted(df["county_geoid"].unique())

    # Build assignment map
    assignment: dict[str, str] = {}

    # Explicit assignments first
    for g in config.split.test_geoids:
        if g in geoids:
            assignment[g] = "test"
    for g in config.split.val_geoids:
        if g in geoids:
            assignment[g] = "val"

    # Auto-assign remaining
    unassigned = [g for g in geoids if g not in assignment]
    rng.shuffle(unassigned)

    n_total = len(geoids)
    n_test_needed = max(1, int(n_total * config.split.test_fraction))
    n_val_needed = max(1, int(n_total * config.split.val_fraction))

    # Subtract already-assigned
    n_test_have = sum(1 for v in assignment.values() if v == "test")
    n_val_have = sum(1 for v in assignment.values() if v == "val")
    n_test_auto = max(0, n_test_needed - n_test_have)
    n_val_auto = max(0, n_val_needed - n_val_have)

    idx = 0
    for _ in range(n_test_auto):
        if idx < len(unassigned):
            assignment[unassigned[idx]] = "test"
            idx += 1
    for _ in range(n_val_auto):
        if idx < len(unassigned):
            assignment[unassigned[idx]] = "val"
            idx += 1
    for i in range(idx, len(unassigned)):
        assignment[unassigned[i]] = "train"

    df["split"] = df["county_geoid"].map(assignment).fillna("train")

    for split_name in ["train", "val", "test"]:
        count = (df["split"] == split_name).sum()
        split_geoids = [g for g, s in assignment.items() if s == split_name]
        logger.info("Split '%s': %d patches from %s", split_name, count, split_geoids)

    return df


def _random_split(
    df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Random shuffle split."""
    n = len(df)
    indices = rng.permutation(n)
    n_test = max(1, int(n * config.split.test_fraction))
    n_val = max(1, int(n * config.split.val_fraction))

    splits = np.full(n, "train", dtype=object)
    splits[indices[:n_test]] = "test"
    splits[indices[n_test:n_test + n_val]] = "val"
    df["split"] = splits

    for s in ["train", "val", "test"]:
        logger.info("Split '%s': %d patches", s, (df["split"] == s).sum())

    return df


def organize_by_split(
    metadata: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
) -> None:
    """Copy patches into split-organized subdirectories and save metadata."""
    has_distances = (source_dir / "distances").exists()

    for split_name in ["train", "val", "test"]:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "masks").mkdir(parents=True, exist_ok=True)
        if has_distances:
            (output_dir / split_name / "distances").mkdir(parents=True, exist_ok=True)

    for _, row in metadata.iterrows():
        pid = row["patch_id"]
        split = row["split"]

        src_img = source_dir / "images" / f"{pid}.tif"
        src_mask = source_dir / "masks" / f"{pid}.tif"

        if src_img.exists():
            shutil.copy2(src_img, output_dir / split / "images" / f"{pid}.tif")
        if src_mask.exists():
            shutil.copy2(src_mask, output_dir / split / "masks" / f"{pid}.tif")

        if has_distances:
            src_dist = source_dir / "distances" / f"{pid}.tif"
            if src_dist.exists():
                shutil.copy2(src_dist, output_dir / split / "distances" / f"{pid}.tif")

    # Per-split metadata
    for split_name, group in metadata.groupby("split"):
        group.to_csv(output_dir / split_name / "metadata.csv", index=False)

    metadata.to_csv(output_dir / "metadata.csv", index=False)

    # Summary
    summary = {"total_patches": len(metadata), "splits": {}}
    for split_name, group in metadata.groupby("split"):
        summary["splits"][split_name] = {
            "n_patches": len(group),
            "n_positive": int((group["label"] == "positive").sum()),
            "n_negative": int((group["label"] == "negative").sum()),
            "counties": sorted(group["county_geoid"].unique().tolist()),
        }
    with open(output_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
