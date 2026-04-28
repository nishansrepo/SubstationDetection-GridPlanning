"""Sparse (demand cell, substation) distance matrix."""

from __future__ import annotations

import geopandas as gpd
import numpy as np

from .containers import Candidates, DemandGrid, SparseDistances


def build_sparse_distances(grid: DemandGrid,
                           candidates: Candidates,
                           existing: gpd.GeoDataFrame,
                           max_radius_m: float) -> SparseDistances:
    """
    Compute distance from each demand cell to reachable substations.
    Only keeps pairs within max_radius_m (sparsification).

    Uses scipy KDTree for O(D log S) radius queries instead of O(D × S)
    brute-force Shapely calls.
    """
    from scipy.spatial import KDTree

    # Extract coordinate arrays
    demand_coords = np.column_stack([
        grid.cells.geometry.centroid.x.values,
        grid.cells.geometry.centroid.y.values,
    ])
    demand_ids = np.array(grid.cell_ids)

    existing_coords = np.column_stack([existing.geometry.x.values,
                                       existing.geometry.y.values])
    existing_ids = existing.index.values

    candidate_coords = np.column_stack([candidates.sites.geometry.x.values,
                                        candidates.sites.geometry.y.values])
    candidate_ids_arr = candidates.sites["candidate_id"].values

    # Build KD-trees for existing and candidate substations
    pairs: dict[tuple[int, tuple[str, int]], float] = {}

    if len(existing_coords):
        tree_ex = KDTree(existing_coords)
        for idx, neighbors in enumerate(
            tree_ex.query_ball_point(demand_coords, r=max_radius_m)
        ):
            d = int(demand_ids[idx])
            for j in neighbors:
                dist = float(np.linalg.norm(demand_coords[idx] - existing_coords[j]))
                pairs[(d, ("existing", int(existing_ids[j])))] = dist

    if len(candidate_coords):
        tree_cand = KDTree(candidate_coords)
        for idx, neighbors in enumerate(
            tree_cand.query_ball_point(demand_coords, r=max_radius_m)
        ):
            d = int(demand_ids[idx])
            for j in neighbors:
                dist = float(np.linalg.norm(demand_coords[idx] - candidate_coords[j]))
                pairs[(d, ("candidate", int(candidate_ids_arr[j])))] = dist

    # Feasibility check — drop unreachable cells if their total demand is negligible
    reachable = {d for (d, _) in pairs}
    unreachable = set(grid.cell_ids) - reachable
    if unreachable:
        unreachable_demand = sum(grid.demand(d) for d in unreachable)
        total_demand = sum(grid.demand(d) for d in grid.cell_ids)
        frac = unreachable_demand / total_demand if total_demand else 0.0
        if frac < 0.01:
            print(f"  WARNING: {len(unreachable)} demand cells unreachable within "
                  f"{max_radius_m/1000:.0f}km — dropping "
                  f"({unreachable_demand:,.0f} kW, {frac:.2%} of total demand)")
            for d in unreachable:
                grid.cells.drop(index=d, inplace=True)
        else:
            raise ValueError(
                f"{len(unreachable)} demand cells have no substation within "
                f"{max_radius_m}m ({frac:.1%} of demand). "
                f"Increase max_service_radius_m or inspect data."
            )

    sd = SparseDistances(pairs=pairs)
    sd.build_index()
    print(f"  Sparse distances: {len(pairs)} pairs "
          f"({len(grid.cell_ids)} demand × "
          f"{len(existing) + len(candidates.candidate_ids)} subs, "
          f"radius={max_radius_m/1000:.0f}km)")
    return sd
