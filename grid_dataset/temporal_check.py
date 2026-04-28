"""
Temporal alignment between NAIP imagery and OSM labels (Feature 1).

Checks whether the OSM label edit timestamp is within a configurable
window of the NAIP tile acquisition date. Produces a temporal_mismatch
flag per patch that can be used to filter or stratify the training set.

Implementation approach:
    - NAIP acquisition date comes from the STAC item metadata, already
      captured in the enriched metadata by naip_source._read_stac().
    - OSM edit timestamps come from the Overpass API 'meta' output mode,
      which returns the last-modified timestamp for each element.
    - Checking happens post-extraction on the metadata DataFrame,
      so it doesn't slow down the fetch loop.

Called by dataset_builder after metadata compilation (Step 5).
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def fetch_osm_timestamps(osm_ids: list[int]) -> dict[int, str]:
    """Fetch last-edit timestamps for a batch of OSM element IDs.

    Uses the Overpass API with [out:json][meta] to get the 'timestamp'
    field. Batches into groups of 200 to avoid URL length limits.

    Parameters
    ----------
    osm_ids : list[int]
        OSM node/way IDs (positive integers).

    Returns
    -------
    dict mapping osm_id → ISO 8601 timestamp string, or empty string
    if not found.
    """
    import requests
    import time

    timestamps: dict[int, str] = {}
    batch_size = 200

    # Filter out invalid IDs (negatives are our synthetic IDs for negatives)
    valid_ids = [i for i in osm_ids if i > 0]

    for start in range(0, len(valid_ids), batch_size):
        batch = valid_ids[start:start + batch_size]
        id_union = "".join(f"node({i});" for i in batch)
        id_union += "".join(f"way({i});" for i in batch)

        query = f"[out:json][timeout:60];({id_union});out meta;"

        for attempt in range(3):
            try:
                resp = requests.post(
                    "https://overpass-api.de/api/interpreter",
                    data={"data": query},
                    timeout=120,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 ** attempt * 5)
                    continue
                resp.raise_for_status()

                for el in resp.json().get("elements", []):
                    ts = el.get("timestamp", "")
                    timestamps[el["id"]] = ts
                break

            except Exception as e:
                if attempt == 2:
                    logger.warning("Failed to fetch timestamps for batch: %s", e)
                time.sleep(2 ** attempt * 5)

        # Rate limit between batches
        time.sleep(2)

    return timestamps


def check_temporal_alignment(
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Add temporal alignment flags to the metadata DataFrame.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must contain 'osm_id' and 'acquisition_date' columns.
    config : PipelineConfig
        Temporal config with max_gap_years and exclude_mismatched.

    Returns
    -------
    pd.DataFrame with added columns:
        - osm_edit_date: ISO timestamp of last OSM edit
        - temporal_gap_years: absolute gap in years
        - temporal_mismatch: bool flag (True if gap > threshold)
    """
    if not config.temporal.enabled:
        metadata["temporal_mismatch"] = False
        return metadata

    logger.info("Checking temporal alignment (max gap: %.1f years)...",
                config.temporal.max_gap_years)

    # Collect unique OSM IDs from positive patches
    positive = metadata[metadata["label"] == "positive"]
    osm_ids = positive["osm_id"].dropna().astype(int).unique().tolist()

    if not osm_ids:
        metadata["osm_edit_date"] = ""
        metadata["temporal_gap_years"] = np.nan
        metadata["temporal_mismatch"] = False
        return metadata

    # Fetch OSM edit timestamps
    ts_map = fetch_osm_timestamps(osm_ids)
    logger.info("Fetched timestamps for %d/%d OSM features",
                len(ts_map), len(osm_ids))

    # Map timestamps back to metadata
    metadata = metadata.copy()
    metadata["osm_edit_date"] = metadata["osm_id"].map(
        lambda x: ts_map.get(int(x), "") if pd.notna(x) and int(x) > 0 else ""
    )

    # Compute gap
    def _compute_gap(row):
        acq = row.get("acquisition_date", "")
        osm = row.get("osm_edit_date", "")
        if not acq or not osm:
            return np.nan
        try:
            acq_dt = datetime.fromisoformat(acq.replace("Z", "+00:00"))
            osm_dt = datetime.fromisoformat(osm.replace("Z", "+00:00"))
            gap_days = abs((acq_dt - osm_dt).days)
            return gap_days / 365.25
        except (ValueError, TypeError):
            return np.nan

    metadata["temporal_gap_years"] = metadata.apply(_compute_gap, axis=1)
    metadata["temporal_mismatch"] = (
        metadata["temporal_gap_years"] > config.temporal.max_gap_years
    )

    # For negative patches, no mismatch by definition
    metadata.loc[metadata["label"] == "negative", "temporal_mismatch"] = False

    n_mismatch = metadata["temporal_mismatch"].sum()
    n_checked = metadata["temporal_gap_years"].notna().sum()
    logger.info(
        "Temporal check: %d/%d patches checked, %d flagged as mismatched",
        n_checked, len(metadata), n_mismatch,
    )

    if config.temporal.exclude_mismatched and n_mismatch > 0:
        before = len(metadata)
        metadata = metadata[~metadata["temporal_mismatch"]].copy()
        logger.info(
            "Excluded %d temporally mismatched patches (%d → %d)",
            before - len(metadata), before, len(metadata),
        )

    return metadata
