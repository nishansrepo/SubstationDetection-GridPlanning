"""
Label noise audit (Feature 4).

Diagnoses likely labeling errors by computing spectral indices within
labeled substation regions. Substations are man-made structures on cleared
lots — high vegetation (NDVI) or water (NDWI) within labeled pixels
indicates the label is probably wrong.

Spectral indices are computed directly from the NAIP 4-band image:
    NDVI = (NIR - Red) / (NIR + Red)
    NDWI = (Green - NIR) / (Green + NIR)

No additional data downloads are needed.

Output: a summary CSV per patch with columns:
    patch_id, n_labeled_pixels, pct_high_ndvi, pct_water,
    mean_ndvi_in_label, suspect_label

Called by dataset_builder after extraction (Step 5), before splitting.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def compute_ndvi(image: np.ndarray, red_idx: int, nir_idx: int) -> np.ndarray:
    """Compute NDVI from a multi-band image.

    Parameters
    ----------
    image : np.ndarray
        Shape (C, H, W), dtype uint8 or float.
    red_idx, nir_idx : int
        0-based band indices.

    Returns
    -------
    np.ndarray of float32, shape (H, W). Range [-1, 1].
    """
    red = image[red_idx].astype(np.float32)
    nir = image[nir_idx].astype(np.float32)
    denom = nir + red
    # Avoid division by zero
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    return ndvi.astype(np.float32)


def compute_ndwi(image: np.ndarray, green_idx: int, nir_idx: int) -> np.ndarray:
    """Compute NDWI (Normalized Difference Water Index).

    Parameters
    ----------
    image : np.ndarray
        Shape (C, H, W).
    green_idx, nir_idx : int
        0-based band indices.

    Returns
    -------
    np.ndarray of float32, shape (H, W). Range [-1, 1].
    """
    green = image[green_idx].astype(np.float32)
    nir = image[nir_idx].astype(np.float32)
    denom = green + nir
    ndwi = np.where(denom > 0, (green - nir) / denom, 0.0)
    return ndwi.astype(np.float32)


def audit_single_patch(
    image_path: Path,
    mask_path: Path,
    config: PipelineConfig,
) -> dict:
    """Audit one patch for label noise.

    Returns a dict with diagnostic values.
    """
    cfg = config.label_audit

    try:
        with rasterio.open(image_path) as img_src:
            image = img_src.read()  # (C, H, W)
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)  # (H, W)
    except Exception as e:
        return {
            "error": str(e),
            "n_labeled_pixels": 0,
            "pct_high_ndvi": np.nan,
            "pct_water": np.nan,
            "mean_ndvi_in_label": np.nan,
            "mean_ndwi_in_label": np.nan,
            "suspect_label": False,
        }

    n_labeled = int(mask.sum())
    if n_labeled == 0:
        return {
            "error": "",
            "n_labeled_pixels": 0,
            "pct_high_ndvi": 0.0,
            "pct_water": 0.0,
            "mean_ndvi_in_label": 0.0,
            "mean_ndwi_in_label": 0.0,
            "suspect_label": False,
        }

    # Compute indices
    ndvi = compute_ndvi(image, cfg.red_band_index, cfg.nir_band_index)
    ndwi = compute_ndwi(image, cfg.green_band_index, cfg.nir_band_index)

    # Extract values within the labeled region
    label_mask = mask > 0
    ndvi_in_label = ndvi[label_mask]
    ndwi_in_label = ndwi[label_mask]

    high_ndvi_count = int((ndvi_in_label > cfg.ndvi_threshold).sum())
    water_count = int((ndwi_in_label > cfg.ndwi_threshold).sum())

    pct_high_ndvi = 100.0 * high_ndvi_count / n_labeled
    pct_water = 100.0 * water_count / n_labeled

    # A label is suspect if >50% of its pixels are vegetation or water
    suspect = (pct_high_ndvi > 50.0) or (pct_water > 50.0)

    return {
        "error": "",
        "n_labeled_pixels": n_labeled,
        "pct_high_ndvi": round(pct_high_ndvi, 2),
        "pct_water": round(pct_water, 2),
        "mean_ndvi_in_label": round(float(ndvi_in_label.mean()), 4),
        "mean_ndwi_in_label": round(float(ndwi_in_label.mean()), 4),
        "suspect_label": suspect,
    }


def run_label_audit(
    metadata: pd.DataFrame,
    raw_dir: Path,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Run label noise audit on all positive patches.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must contain 'patch_id' and 'label' columns.
    raw_dir : Path
        Directory with images/ and masks/ subdirectories.
    config : PipelineConfig
        Label audit config.

    Returns
    -------
    pd.DataFrame with one row per audited patch, containing diagnostic columns.
    Also saved as label_audit.csv in the dataset output directory.
    """
    if not config.label_audit.enabled:
        logger.info("Label audit disabled, skipping.")
        return pd.DataFrame()

    positive = metadata[metadata["label"] == "positive"]
    logger.info("Running label noise audit on %d positive patches...",
                len(positive))

    records = []
    n_suspect = 0

    for _, row in positive.iterrows():
        pid = row["patch_id"]
        result = audit_single_patch(
            raw_dir / "images" / f"{pid}.tif",
            raw_dir / "masks" / f"{pid}.tif",
            config,
        )
        result["patch_id"] = pid
        result["county_geoid"] = row.get("county_geoid", "")
        result["osm_id"] = row.get("osm_id", "")
        result["substation_type"] = row.get("substation_type", "")
        records.append(result)

        if result["suspect_label"]:
            n_suspect += 1

    audit_df = pd.DataFrame(records)

    # Summary statistics
    logger.info(
        "Label audit complete: %d patches audited, %d (%.1f%%) flagged suspect",
        len(audit_df), n_suspect,
        100.0 * n_suspect / max(len(audit_df), 1),
    )

    if not audit_df.empty:
        logger.info(
            "  Mean NDVI in labels: %.3f",
            audit_df["mean_ndvi_in_label"].mean(),
        )
        logger.info(
            "  Mean high-NDVI percentage: %.1f%%",
            audit_df["pct_high_ndvi"].mean(),
        )
        logger.info(
            "  Mean water percentage: %.1f%%",
            audit_df["pct_water"].mean(),
        )

    # Save to disk
    out_path = config.output_path / "label_audit.csv"
    audit_df.to_csv(out_path, index=False)
    logger.info("Audit results saved to %s", out_path)

    return audit_df
