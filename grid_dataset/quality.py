"""
Quality assurance and dataset validation.

Post-extraction checks: shape consistency, binary masks, spatial alignment,
empty patches, balance statistics.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)


def validate_patch_pair(
    image_path: Path, mask_path: Path,
    expected_bands: int = 4, expected_size: int = 512,
) -> list[str]:
    """Validate one image-mask pair. Returns list of error strings."""
    errors = []
    if not image_path.exists():
        return [f"Image not found: {image_path}"]
    if not mask_path.exists():
        return [f"Mask not found: {mask_path}"]

    try:
        with rasterio.open(image_path) as src:
            if src.count != expected_bands:
                errors.append(f"Image bands={src.count}, expected {expected_bands}")
            if src.width != expected_size or src.height != expected_size:
                errors.append(f"Image size {src.width}x{src.height}")
            data = src.read()
            if data.max() == data.min():
                errors.append("Image is constant")
            img_transform, img_crs = src.transform, src.crs
    except Exception as e:
        return [f"Image read error: {e}"]

    try:
        with rasterio.open(mask_path) as src:
            if src.count != 1:
                errors.append(f"Mask bands={src.count}")
            if src.width != expected_size or src.height != expected_size:
                errors.append(f"Mask size {src.width}x{src.height}")
            vals = np.unique(src.read(1))
            if not np.all(np.isin(vals, [0, 1])):
                errors.append(f"Mask non-binary: {vals}")
            if src.transform != img_transform:
                errors.append("Transform mismatch")
            if src.crs != img_crs:
                errors.append("CRS mismatch")
    except Exception as e:
        errors.append(f"Mask read error: {e}")

    return errors


def validate_dataset(
    dataset_dir: Path, metadata: pd.DataFrame,
    expected_bands: int = 4, expected_size: int = 512,
    sample_fraction: float = 1.0,
) -> pd.DataFrame:
    """Validate all or a sample of patches."""
    sample = (metadata.sample(frac=sample_fraction, random_state=42)
              if sample_fraction < 1.0 else metadata)
    results = []
    n_bad = 0
    for _, row in sample.iterrows():
        pid = row["patch_id"]
        errs = validate_patch_pair(
            dataset_dir / "images" / f"{pid}.tif",
            dataset_dir / "masks" / f"{pid}.tif",
            expected_bands, expected_size,
        )
        if errs:
            n_bad += 1
            logger.warning("Invalid %s: %s", pid, "; ".join(errs))
        results.append({"patch_id": pid, "is_valid": not errs,
                        "errors": "; ".join(errs)})
    df = pd.DataFrame(results)
    logger.info("Validated %d: %d valid, %d invalid", len(df), len(df) - n_bad, n_bad)
    return df


def compute_dataset_statistics(metadata: pd.DataFrame) -> dict:
    """Compute summary statistics for JSON serialization."""
    stats = {
        "total_patches": len(metadata),
        "positive_patches": int((metadata["label"] == "positive").sum()),
        "negative_patches": int((metadata["label"] == "negative").sum()),
        "balance_ratio": float((metadata["label"] == "positive").mean()),
    }
    if "region" in metadata.columns:
        stats["per_region"] = (
            metadata.groupby(["region", "label"]).size()
            .unstack(fill_value=0).to_dict(orient="index")
        )
    if "county_geoid" in metadata.columns:
        stats["per_county"] = (
            metadata.groupby(["county_geoid", "label"]).size()
            .unstack(fill_value=0).to_dict(orient="index")
        )
    if "substation_type" in metadata.columns:
        type_counts = metadata[metadata["label"] == "positive"].groupby(
            "substation_type").size().to_dict()
        stats["per_substation_type"] = type_counts
    if "positive_pixels" in metadata.columns:
        pos = metadata[metadata["label"] == "positive"]["positive_pixels"]
        if len(pos) > 0:
            stats["positive_pixel_stats"] = {
                "mean": float(pos.mean()), "median": float(pos.median()),
                "std": float(pos.std()), "min": int(pos.min()),
                "max": int(pos.max()),
            }
    return stats
