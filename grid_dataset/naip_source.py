"""
NAIP imagery access layer.

Abstracts multiple data sources behind a single read_patch() call:
    1. Local GeoTIFF files (for pre-downloaded state/county mosaics)
    2. Microsoft Planetary Computer STAC API (remote COG reads)

Tries local files in order first; falls back to STAC.
"""

import logging
import time
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class _LocalFile:
    """Wrapper around a rasterio DatasetReader with bounds caching."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.src = rasterio.open(path)
        self.bounds = self.src.bounds
        self.crs = self.src.crs

    def contains(self, lon: float, lat: float) -> bool:
        if self.crs and not self.crs.is_geographic:
            from pyproj import Transformer
            t = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
            x, y = t.transform(lon, lat)
        else:
            x, y = lon, lat
        b = self.bounds
        return b.left <= x <= b.right and b.bottom <= y <= b.top

    def read(self, lon: float, lat: float, patch_size: int, bands: tuple):
        if self.crs and not self.crs.is_geographic:
            from pyproj import Transformer
            t = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
            x, y = t.transform(lon, lat)
        else:
            x, y = lon, lat

        row, col = self.src.index(x, y)
        half = patch_size // 2
        rs, cs = row - half, col - half

        if rs < 0 or cs < 0 or rs + patch_size > self.src.height or cs + patch_size > self.src.width:
            return None

        window = Window(cs, rs, patch_size, patch_size)
        patch = self.src.read(list(bands), window=window)
        meta = {
            "transform": self.src.window_transform(window),
            "crs": self.src.crs,
            "source": "local",
            "width": patch_size,
            "height": patch_size,
            "count": len(bands),
            "dtype": patch.dtype,
        }
        return patch, meta

    def close(self):
        self.src.close()


class NaipSource:
    """Unified NAIP reader: local files → STAC fallback."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._locals: list[_LocalFile] = []
        self._stac_catalog = None

        for path in config.naip.local_naip_paths:
            try:
                lf = _LocalFile(path)
                self._locals.append(lf)
                logger.info("Opened local NAIP: %s  CRS=%s", path, lf.crs)
            except Exception:
                logger.exception("Failed to open local NAIP: %s", path)

    def _get_stac_catalog(self):
        if self._stac_catalog is None:
            import planetary_computer
            import pystac_client
            self._stac_catalog = pystac_client.Client.open(
                self._config.naip.stac_api_url,
                modifier=planetary_computer.sign_inplace,
            )
        return self._stac_catalog

    def _read_stac(self, lon, lat, patch_size, bands):
        """Read a patch from STAC with retry logic and metadata enrichment."""
        import hashlib

        catalog = self._get_stac_catalog()
        yr_lo, yr_hi = self._config.naip.year_range
        max_retries = self._config.naip.max_retries
        verify = self._config.naip.verify_checksum

        search = catalog.search(
            collections=[self._config.naip.collection],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=f"{yr_lo}/{yr_hi}",
            sortby=[{"field": "datetime", "direction": "desc"}],
            max_items=1,
        )
        items = list(search.items())
        if not items:
            return None

        item = items[0]
        href = item.assets["image"].href

        # Extract acquisition datetime from STAC item
        acq_datetime = None
        if item.datetime:
            acq_datetime = item.datetime.isoformat()
        elif item.properties.get("datetime"):
            acq_datetime = item.properties["datetime"]

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                with rasterio.open(href) as src:
                    from pyproj import Transformer
                    t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    x, y = t.transform(lon, lat)
                    row, col = src.index(x, y)
                    half = patch_size // 2
                    rs, cs = row - half, col - half

                    if (rs < 0 or cs < 0
                            or rs + patch_size > src.height
                            or cs + patch_size > src.width):
                        return None

                    window = Window(cs, rs, patch_size, patch_size)
                    patch = src.read(list(bands), window=window)

                    # Resolution logging — actual pixel size from transform
                    res_x = abs(src.transform.a)
                    res_y = abs(src.transform.e)

                    meta = {
                        "transform": src.window_transform(window),
                        "crs": src.crs,
                        "source": "stac",
                        "stac_item_id": item.id,
                        "width": patch_size,
                        "height": patch_size,
                        "count": src.count,
                        "dtype": patch.dtype,
                        "acquisition_date": acq_datetime,
                        "resolution_x": float(res_x),
                        "resolution_y": float(res_y),
                        "actual_band_count": src.count,
                    }

                    # Checksum verification
                    if verify:
                        raw_bytes = patch.tobytes()
                        sha = hashlib.sha256(raw_bytes).hexdigest()
                        meta["sha256"] = sha
                        # Corruption heuristic: all-zero or all-same-value
                        if patch.max() == patch.min():
                            logger.warning(
                                "Patch at (%.4f,%.4f) appears corrupt "
                                "(constant value %d), sha=%s",
                                lat, lon, int(patch.max()), sha,
                            )
                            meta["possibly_corrupt"] = True

                    return patch, meta

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt * 2  # 2s, 4s, 8s
                    logger.warning(
                        "STAC read attempt %d/%d failed for (%.4f,%.4f): %s. "
                        "Retrying in %ds...",
                        attempt + 1, max_retries + 1, lat, lon, e, wait,
                    )
                    time.sleep(wait)

        logger.error(
            "STAC read failed after %d attempts for (%.4f,%.4f): %s",
            max_retries + 1, lat, lon, last_error,
        )
        return None

    def read_patch(self, lon: float, lat: float):
        """Read a multi-band patch centered on (lon, lat).

        Returns (ndarray shape (C,H,W), metadata dict) or None.
        """
        ps = self._config.patch.patch_size
        bands = self._config.naip.bands

        # Try local files in order
        for lf in self._locals:
            if lf.contains(lon, lat):
                result = lf.read(lon, lat, ps, bands)
                if result is not None:
                    return result

        # STAC fallback
        time.sleep(self._config.naip.request_delay_s)
        return self._read_stac(lon, lat, ps, bands)

    def close(self):
        for lf in self._locals:
            lf.close()
        self._locals.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
