"""Candidate site generation and filtering."""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import OptimizerConfig
from .containers import Candidates, DemandGrid, InputData


def generate_candidates(config: OptimizerConfig,
                        grid: DemandGrid,
                        data: InputData,
                        data_centers: Optional[gpd.GeoDataFrame] = None) -> Candidates:
    """Populated cell centroids above demand threshold are candidates; filter by feasibility.

    Data centers (existing facilities already served by the grid) are NOT treated as
    unmet demand.  Instead their locations act as a signal that nearby grid corridors
    carry high load density.  Any data center that has no regular candidate site within
    dc_candidate_radius_m is injected as an additional priority candidate, bypassing
    the demand-threshold and transmission-proximity filters (but still subject to the
    existing-substation exclusion zone).
    """
    sites = grid.cells[["cell_id", "geometry", "demand_kw"]].copy()
    sites["geometry"] = sites.geometry.centroid

    # Filter: minimum demand threshold — skip low-demand cells as candidate sites
    before = len(sites)
    sites = sites[sites["demand_kw"] >= config.min_candidate_demand_kw].copy()
    print(f"  Demand threshold filter: {before} -> {len(sites)} candidates "
          f"(>= {config.min_candidate_demand_kw:.0f} kW)")

    # Filter: transmission line proximity (skip if no line data)
    if data.transmission_lines is not None:
        before = len(sites)
        sites = _filter_by_transmission_proximity(
            sites, data.transmission_lines, config.max_service_radius_m
        )
        print(f"  Transmission filter: {before} -> {len(sites)} candidates")

    # Filter: drop candidates very close to an existing substation
    before = len(sites)
    sites = _filter_near_existing(sites, data.existing_substations, min_dist_m=500.0)
    print(f"  Existing-proximity filter: {before} -> {len(sites)} candidates")

    # Inject priority candidates near data centers
    if data_centers is not None and len(data_centers):
        extra = _inject_dc_priority_candidates(
            sites, data_centers, data.existing_substations,
            config.dc_candidate_radius_m, config.dc_min_sqft, config.crs_proj
        )
        if len(extra):
            sites = pd.concat([sites, extra], ignore_index=True)
            print(f"  DC-priority injection: +{len(extra)} candidates "
                  f"(DCs with no coverage within {config.dc_candidate_radius_m/1000:.0f}km)")

    sites = sites.reset_index(drop=True)
    sites["candidate_id"] = sites.index
    sites = sites.set_index("candidate_id", drop=False)
    return Candidates(sites=sites[["candidate_id", "geometry"]])


def _inject_dc_priority_candidates(
        existing_sites: gpd.GeoDataFrame,
        data_centers: gpd.GeoDataFrame,
        existing_substations: gpd.GeoDataFrame,
        radius_m: float,
        min_sqft: float,
        crs: str) -> gpd.GeoDataFrame:
    """
    For each data center not already covered by a nearby candidate site,
    add the data center's location as an extra candidate.

    Skips DCs smaller than min_sqft and DCs already within 500 m of an
    existing substation (same exclusion used for regular candidates).
    """
    from scipy.spatial import KDTree

    dcs = data_centers.copy()
    if min_sqft > 0 and "sqft" in dcs.columns:
        dcs = dcs[dcs["sqft"] >= min_sqft]
    if dcs.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    dc_coords = np.column_stack([dcs.geometry.x.values, dcs.geometry.y.values])

    # Check which DCs already have a candidate site within radius_m
    if len(existing_sites):
        site_coords = np.column_stack([
            existing_sites.geometry.x.values,
            existing_sites.geometry.y.values,
        ])
        tree = KDTree(site_coords)
        uncovered_mask = np.array([
            len(tree.query_ball_point(dc_coords[i], r=radius_m)) == 0
            for i in range(len(dc_coords))
        ])
    else:
        uncovered_mask = np.ones(len(dcs), dtype=bool)

    uncovered_dcs = dcs[uncovered_mask].copy()
    if uncovered_dcs.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    # Also exclude if within 500 m of an existing substation
    sub_union = existing_substations.geometry.union_all()
    too_close = uncovered_dcs.geometry.apply(lambda pt: pt.distance(sub_union) < 500.0)
    uncovered_dcs = uncovered_dcs[~too_close]

    if uncovered_dcs.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    extra = gpd.GeoDataFrame(
        {"geometry": uncovered_dcs.geometry.values,
         "demand_kw": 0.0},
        crs=crs,
    )
    return extra


def _filter_by_transmission_proximity(sites: gpd.GeoDataFrame,
                                      lines: gpd.GeoDataFrame,
                                      max_dist_m: float) -> gpd.GeoDataFrame:
    """Keep only candidates within max_dist_m of a transmission line."""
    merged_lines = lines.geometry.union_all()
    dists = sites.geometry.apply(lambda pt: pt.distance(merged_lines))
    return sites[dists <= max_dist_m].copy()


def _filter_near_existing(sites: gpd.GeoDataFrame,
                          existing: gpd.GeoDataFrame,
                          min_dist_m: float) -> gpd.GeoDataFrame:
    """Drop candidates that are within min_dist_m of any existing substation."""
    merged_existing = existing.geometry.union_all()
    dists = sites.geometry.apply(lambda pt: pt.distance(merged_existing))
    return sites[dists >= min_dist_m].copy()
