"""Build a regular demand grid by areal interpolation of block group data."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from .config import OptimizerConfig
from .containers import DemandGrid, InputData


def build_demand_grid(config: OptimizerConfig, data: InputData) -> DemandGrid:
    """Overlay a regular grid and areal-interpolate block group population
    (and building footprint area, when available) into per-cell demand."""
    bg = data.census_blockgroups.copy()
    bg["building_area_m2"] = 0.0
    if data.building_footprints is not None:
        bg = bg.merge(data.building_footprints, on="GEOID", how="left",
                      suffixes=("", "_bldg"))
        if "building_area_m2_bldg" in bg.columns:
            bg["building_area_m2"] = bg["building_area_m2_bldg"].fillna(0)
            bg.drop(columns=["building_area_m2_bldg"], inplace=True)

    grid_cells = _make_regular_grid(bg.total_bounds,
                                    config.grid_cell_size_m,
                                    config.crs_proj)
    populated = _areal_interpolate(grid_cells, bg)

    has_buildings = data.building_footprints is not None
    populated["demand_kw"] = populated["population"] * config.pop_to_peak_kw
    proxy_desc = ["population"]
    if has_buildings:
        populated["demand_kw"] += (populated["building_area_m2"]
                                   * config.residential_kw_per_m2)
        proxy_desc.append("buildings")
    print(f"  Demand proxy: {' + '.join(proxy_desc)}")

    # Drop empty cells
    populated = populated[populated["demand_kw"] > 0].reset_index(drop=True)

    populated["cell_id"] = populated.index
    populated = populated.set_index("cell_id", drop=False)

    print(f"  Demand grid: {len(populated)} cells total "
          f"({config.grid_cell_size_m:.0f}m resolution, "
          f"total: {populated['demand_kw'].sum():,.0f} kW)")
    return DemandGrid(cells=populated)


def _make_regular_grid(bounds: tuple[float, float, float, float],
                       cell_size_m: float,
                       crs: str) -> gpd.GeoDataFrame:
    """Create a grid of square polygons covering the bounding box."""
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)
    cells = []
    for x in xs:
        for y in ys:
            cells.append(Polygon([
                (x, y), (x + cell_size_m, y),
                (x + cell_size_m, y + cell_size_m), (x, y + cell_size_m),
            ]))
    return gpd.GeoDataFrame(geometry=cells, crs=crs)


def _areal_interpolate(grid: gpd.GeoDataFrame,
                       blockgroups: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Distribute block group attributes (population, building_area_m2) into
    grid cells proportionally by the fraction of each block group's area that
    falls in each grid cell."""
    blockgroups = blockgroups.copy()
    blockgroups["bg_area"] = blockgroups.geometry.area

    intersections = gpd.overlay(grid.reset_index(), blockgroups, how="intersection")
    intersections["overlap_frac"] = (
        intersections.geometry.area / intersections["bg_area"]
    )

    value_cols = ["population", "building_area_m2"]
    for col in value_cols:
        if col in intersections.columns:
            intersections[f"{col}_share"] = intersections["overlap_frac"] * intersections[col]

    grid = grid.copy()
    share_cols = [f"{col}_share" for col in value_cols
                  if f"{col}_share" in intersections.columns]
    agg = intersections.groupby("index")[share_cols].sum()

    for col in value_cols:
        share_col = f"{col}_share"
        grid[col] = 0.0
        if share_col in agg.columns:
            grid.loc[agg.index, col] = agg[share_col].values
    return grid
