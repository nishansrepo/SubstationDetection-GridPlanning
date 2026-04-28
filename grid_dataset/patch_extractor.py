"""
Patch extraction and binary mask generation.

Extracts image patches from NAIP and creates corresponding segmentation
masks from substation geometries. Optionally generates a distance-to-substation
raster using EDT on the binary mask.

Output: 4-band GeoTIFF + binary mask GeoTIFF + optional distance GeoTIFF.
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, mapping

from .config import PipelineConfig
from .naip_source import NaipSource

logger = logging.getLogger(__name__)


def _reproject_geometries_to_patch_crs(
    substations: gpd.GeoDataFrame,
    patch_bounds: tuple,
    patch_crs,
) -> gpd.GeoDataFrame:
    """Clip substations to patch extent and reproject to the patch CRS."""
    from pyproj import Transformer

    if not patch_crs.is_geographic:
        transformer = Transformer.from_crs(patch_crs, "EPSG:4326", always_xy=True)
        w, s = transformer.transform(patch_bounds[0], patch_bounds[1])
        e, n = transformer.transform(patch_bounds[2], patch_bounds[3])
    else:
        w, s, e, n = patch_bounds

    bbox_geom = box(w, s, e, n)
    mask = substations.geometry.intersects(bbox_geom)
    nearby = substations.loc[mask].copy()
    if nearby.empty:
        return nearby
    return nearby.to_crs(patch_crs)


def create_mask(
    substations: gpd.GeoDataFrame,
    patch_meta: dict,
    patch_size: int,
) -> np.ndarray:
    """Rasterize substation geometries into a binary mask."""
    if substations.empty:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    shapes = [(mapping(geom), 1) for geom in substations.geometry]
    return rasterize(
        shapes, out_shape=(patch_size, patch_size),
        transform=patch_meta["transform"], fill=0,
        dtype=np.uint8, all_touched=True,
    )


def create_distance_raster(
    mask: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """Compute Euclidean distance (meters) to nearest substation pixel.

    Uses scipy's distance_transform_edt on the inverted binary mask.
    Each pixel value = distance in meters to the nearest mask=1 pixel edge.

    For negative patches (all-zero masks), every pixel gets the maximum
    distance value (diagonal of the patch).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8, values 0/1.
    resolution_m : float
        Pixel size in meters for converting pixel distance to meters.

    Returns
    -------
    np.ndarray of float32, shape (H, W). Distance in meters.
    """
    from scipy.ndimage import distance_transform_edt

    # EDT computes distance from 0-pixels to nearest 1-pixel
    # We want distance from each pixel to the nearest substation (mask=1)
    # So we invert: 1 where no substation, 0 where substation
    inverted = (mask == 0).astype(np.uint8)
    dist_pixels = distance_transform_edt(inverted)
    dist_meters = (dist_pixels * resolution_m).astype(np.float32)
    return dist_meters


def save_geotiff(array: np.ndarray, path: Path, transform, crs, dtype="uint8"):
    """Write array as GeoTIFF with spatial reference."""
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    count, height, width = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype=dtype, crs=crs, transform=transform, compress="lzw",
    ) as dst:
        dst.write(array.astype(dtype))


def _build_patch_metadata(
    patch_id: str,
    label: str,
    lon: float,
    lat: float,
    positive_pixels: int,
    patch_size: int,
    meta: dict,
    mask: np.ndarray = None,
    resolution: float = 0.6,
) -> dict:
    """Build the metadata dict for a patch, including enriched STAC fields.

    When a mask is provided, computes the bounding box of the substation
    within the patch (pixel coordinates and geographic coordinates).
    """
    transform = meta.get("transform")
    crs = meta.get("crs")

    # Compute patch geographic extent from transform
    patch_west = transform.c if transform else ""
    patch_north = transform.f if transform else ""
    patch_east = (transform.c + transform.a * patch_size) if transform else ""
    patch_south = (transform.f + transform.e * patch_size) if transform else ""

    d = {
        "patch_id": patch_id,
        "label": label,
        "center_lon": lon,
        "center_lat": lat,
        "positive_pixels": positive_pixels,
        "positive_fraction": round(positive_pixels / (patch_size * patch_size), 6),
        "total_pixels": patch_size * patch_size,
        "source": meta.get("source", "unknown"),
        "stac_item_id": meta.get("stac_item_id", ""),
        "crs": str(crs) if crs else "",
        # Patch geographic extent (in native CRS units — meters for UTM)
        "patch_west": patch_west,
        "patch_east": patch_east,
        "patch_north": patch_north,
        "patch_south": patch_south,
        "patch_width_m": round(abs(patch_east - patch_west), 2) if isinstance(patch_east, float) else "",
        "patch_height_m": round(abs(patch_north - patch_south), 2) if isinstance(patch_north, float) else "",
        # Enriched STAC fields
        "acquisition_date": meta.get("acquisition_date", ""),
        "resolution_x": meta.get("resolution_x", ""),
        "resolution_y": meta.get("resolution_y", ""),
        "actual_band_count": meta.get("actual_band_count", ""),
        "sha256": meta.get("sha256", ""),
        "possibly_corrupt": meta.get("possibly_corrupt", False),
    }

    # Substation location within patch (from mask)
    if mask is not None and positive_pixels > 0:
        rows_with_mask = np.where(mask.max(axis=1) > 0)[0]
        cols_with_mask = np.where(mask.max(axis=0) > 0)[0]

        if len(rows_with_mask) > 0 and len(cols_with_mask) > 0:
            # Pixel-space bounding box of the substation within the patch
            px_row_min = int(rows_with_mask[0])
            px_row_max = int(rows_with_mask[-1])
            px_col_min = int(cols_with_mask[0])
            px_col_max = int(cols_with_mask[-1])

            d["mask_bbox_row_min"] = px_row_min
            d["mask_bbox_row_max"] = px_row_max
            d["mask_bbox_col_min"] = px_col_min
            d["mask_bbox_col_max"] = px_col_max
            d["mask_bbox_width_px"] = px_col_max - px_col_min + 1
            d["mask_bbox_height_px"] = px_row_max - px_row_min + 1
            d["mask_bbox_width_m"] = round((px_col_max - px_col_min + 1) * resolution, 1)
            d["mask_bbox_height_m"] = round((px_row_max - px_row_min + 1) * resolution, 1)

            # Geographic centroid of the mask (in native CRS)
            if transform:
                mask_center_col = (px_col_min + px_col_max) / 2
                mask_center_row = (px_row_min + px_row_max) / 2
                mask_geo_x = transform.c + mask_center_col * transform.a
                mask_geo_y = transform.f + mask_center_row * transform.e
                d["mask_centroid_x"] = round(mask_geo_x, 2)
                d["mask_centroid_y"] = round(mask_geo_y, 2)

    return d


def extract_positive_patch(
    lon: float, lat: float,
    all_substations: gpd.GeoDataFrame,
    naip: NaipSource,
    config: PipelineConfig,
    output_dir: Path,
    patch_id: str,
) -> Optional[dict]:
    """Extract one positive patch, mask, and optional distance raster."""
    result = naip.read_patch(lon, lat)
    if result is None:
        return None

    image, meta = result
    ps = config.patch.patch_size
    t = meta["transform"]
    crs = meta["crs"]

    west, north = t.c, t.f
    east = west + t.a * ps
    south = north + t.e * ps
    patch_bounds = (min(west, east), min(south, north),
                    max(west, east), max(south, north))

    nearby = _reproject_geometries_to_patch_crs(all_substations, patch_bounds, crs)
    mask = create_mask(nearby, meta, ps)

    positive_pixels = int(mask.sum())
    if positive_pixels < config.patch.min_substation_pixels:
        return None

    # Save image and mask
    save_geotiff(image, output_dir / "images" / f"{patch_id}.tif", t, crs)
    save_geotiff(mask, output_dir / "masks" / f"{patch_id}.tif", t, crs)

    # Feature 2: distance raster
    if config.distance_raster.enabled:
        dist = create_distance_raster(mask, config.patch.resolution)
        save_geotiff(
            dist,
            output_dir / "distances" / f"{patch_id}.tif",
            t, crs,
            dtype=config.distance_raster.output_dtype,
        )

    return _build_patch_metadata(
        patch_id, "positive", lon, lat, positive_pixels, ps, meta,
        mask=mask, resolution=config.patch.resolution,
    )


def extract_negative_patch(
    lon: float, lat: float,
    naip: NaipSource,
    config: PipelineConfig,
    output_dir: Path,
    patch_id: str,
) -> Optional[dict]:
    """Extract one negative patch."""
    result = naip.read_patch(lon, lat)
    if result is None:
        return None

    image, meta = result
    ps = config.patch.patch_size
    t = meta["transform"]
    crs = meta["crs"]
    mask = np.zeros((ps, ps), dtype=np.uint8)

    save_geotiff(image, output_dir / "images" / f"{patch_id}.tif", t, crs)
    save_geotiff(mask, output_dir / "masks" / f"{patch_id}.tif", t, crs)

    if config.distance_raster.enabled:
        dist = create_distance_raster(mask, config.patch.resolution)
        save_geotiff(
            dist,
            output_dir / "distances" / f"{patch_id}.tif",
            t, crs,
            dtype=config.distance_raster.output_dtype,
        )

    return _build_patch_metadata(
        patch_id, "negative", lon, lat, 0, ps, meta,
        mask=None, resolution=config.patch.resolution,
    )
