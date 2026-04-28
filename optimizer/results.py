"""Results extraction, metrics, and CSV output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .config import BASE_DIR, OptimizerConfig
from .containers import Candidates, DemandGrid, SolveResult, SparseDistances


@dataclass
class ResultsSummary:
    county: str
    n_existing_substations: int
    n_new_substations: int
    total_new_load_served_kw: float
    new_coverage_area_km2: float
    avg_service_dist_before_m: float
    avg_service_dist_after_m: float
    max_service_dist_before_m: float
    max_service_dist_after_m: float
    coverage_gap_before_pct: float
    coverage_gap_after_pct: float
    n_transmission_constrained: int
    transmission_constrained_demand_kw: float
    transmission_expansion_candidates: pd.DataFrame
    assignment_cost: float
    build_cost: float
    total_objective: float
    mip_gap: float
    per_substation: pd.DataFrame

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Optimization Results — {self.county}")
        print(f"{'='*60}")
        print(f"  Existing substations:      {self.n_existing_substations}")
        print(f"  New substations built:     {self.n_new_substations}")
        print(f"  Load shifted to new (kW):  {self.total_new_load_served_kw:,.0f}")
        print(f"  New coverage area (km²):   {self.new_coverage_area_km2:,.1f}")
        print(f"  Avg service distance:      "
              f"{self.avg_service_dist_before_m:,.0f}m -> "
              f"{self.avg_service_dist_after_m:,.0f}m")
        print(f"  Max service distance:      "
              f"{self.max_service_dist_before_m:,.0f}m -> "
              f"{self.max_service_dist_after_m:,.0f}m")
        print()
        print(f"  Coverage analysis (reachable cells only):")
        print(f"    Gap before optimization: {self.coverage_gap_before_pct:.1f}%  (cells > 10km from any substation)")
        print(f"    Gap after optimization:  {self.coverage_gap_after_pct:.1f}%")
        print()
        print(f"  Transmission-constrained:  {self.n_transmission_constrained} cells "
              f"({self.transmission_constrained_demand_kw:,.0f} kW)")
        print(f"    These cells have no candidate site within the coverage radius.")
        print(f"    Closing this gap requires transmission expansion, not substations.")
        print()
        print(f"  Objective cost breakdown:")
        print(f"    Assignment cost:         {self.assignment_cost:,.0f}")
        print(f"    Build cost:              {self.build_cost:,.0f}")
        print(f"    Total:                   {self.total_objective:,.0f}")
        print(f"  MIP gap:                   {self.mip_gap:.2%}")
        if len(self.per_substation):
            print(f"\n  Per-substation details:")
            print(self.per_substation.to_string(index=False))

        if len(self.transmission_expansion_candidates):
            print(f"\n{'-'*60}")
            n_shown = len(self.transmission_expansion_candidates)
            print(f"  Transmission Expansion Planning Spots ({n_shown} clusters >= 300 kW)")
            print(f"  Ranked by demand / distance-to-nearest-transmission-line")
            print(f"  (load unlocked per km of required grid extension; >= 200 kW cells only)")
            print(f"{'-'*60}")
            print(self.transmission_expansion_candidates.to_string(index=False))
            print(f"\n  Note: these cells lie beyond the reachable grid footprint.")
            print(f"  They are candidates for a transmission expansion study,")
            print(f"  not direct substation build recommendations.")
        print()


def rank_transmission_constrained(
        constrained_ids: set[int],
        grid: DemandGrid,
        transmission_lines: Optional[gpd.GeoDataFrame],
        existing: gpd.GeoDataFrame,
        config: OptimizerConfig,
        top_n: int = 10) -> pd.DataFrame:
    """Cluster transmission-constrained cells and rank clusters by
    demand/distance-to-nearest-transmission-line."""
    from scipy.spatial import KDTree

    if not constrained_ids:
        return pd.DataFrame()

    cluster_radius_m = config.max_coverage_dist_m

    ids_arr = list(constrained_ids)
    centroids = np.array([
        [grid.cells.loc[d, "geometry"].centroid.x,
         grid.cells.loc[d, "geometry"].centroid.y]
        for d in ids_arr
    ])
    demands = np.array([grid.demand(d) for d in ids_arr])

    if transmission_lines is not None:
        tx_union = transmission_lines.geometry.union_all()
    else:
        tx_union = None
    sub_union = existing.geometry.union_all()

    tree = KDTree(centroids)

    min_cluster_demand_kw = 300.0

    remaining = set(range(len(ids_arr)))
    clusters = []
    while remaining:
        seed = max(remaining, key=lambda i: demands[i])
        members = [j for j in tree.query_ball_point(centroids[seed], r=cluster_radius_m)
                   if j in remaining]
        for j in members:
            remaining.discard(j)

        member_demands = demands[members]
        total_demand = member_demands.sum()
        member_coords = centroids[members]

        if total_demand > 0:
            cx = np.average(member_coords[:, 0], weights=member_demands)
            cy = np.average(member_coords[:, 1], weights=member_demands)
        else:
            cx, cy = member_coords.mean(axis=0)

        cluster_pt = Point(cx, cy)

        if tx_union is not None:
            d_tx = cluster_pt.distance(tx_union)
        else:
            d_tx = cluster_pt.distance(sub_union)

        d_sub = cluster_pt.distance(sub_union)

        d_tx_km = max(d_tx / 1000, 1.0)
        score = total_demand / d_tx_km

        clusters.append({
            "_cx": cx, "_cy": cy,
            "n_cells": len(members),
            "cluster_demand_kw": round(total_demand, 0),
            "dist_to_tx_km": round(d_tx / 1000, 1),
            "dist_to_sub_km": round(d_sub / 1000, 1),
            "score": round(score, 1),
        })

    if not clusters:
        return pd.DataFrame()

    df = pd.DataFrame(clusters)
    df = df[df["cluster_demand_kw"] >= min_cluster_demand_kw]
    df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    pts_geo = gpd.GeoSeries(
        [Point(r["_cx"], r["_cy"]) for _, r in df.iterrows()],
        crs=config.crs_proj,
    ).to_crs(config.crs_geo)
    df["lat"] = pts_geo.apply(lambda p: round(p.y, 5))
    df["lon"] = pts_geo.apply(lambda p: round(p.x, 5))

    return df[["lat", "lon", "n_cells", "cluster_demand_kw",
               "dist_to_tx_km", "dist_to_sub_km", "score"]]


def extract_results(result: SolveResult,
                    config: OptimizerConfig,
                    grid: DemandGrid,
                    candidates: Candidates,
                    existing: gpd.GeoDataFrame,
                    distances: SparseDistances,
                    transmission_lines: Optional[gpd.GeoDataFrame] = None) -> ResultsSummary:
    """Compute all output metrics from the solver result."""
    built_ids = [c for c, v in result.build.items() if v == 1]

    constrained_ids = {
        d for d in grid.cell_ids
        if not any(
            distances.pairs[(d, s)] <= config.max_coverage_dist_m
            for s in distances.neighbors_of(d)
        )
    }
    constrained_demand_kw = sum(grid.demand(d) for d in constrained_ids)
    reachable_ids = [d for d in grid.cell_ids if d not in constrained_ids]

    tx_expansion = rank_transmission_constrained(
        constrained_ids, grid, transmission_lines, existing, config, top_n=10
    )

    before = _compute_assignment_metrics(grid, existing, reachable_ids)
    after = _compute_solved_metrics(result, grid, distances, reachable_ids)

    per_sub = _build_per_substation_table(
        result, built_ids, candidates, grid, distances, config
    )

    n_cells_served_by_new = 0
    for d in grid.cell_ids:
        best_s = max(
            ((s, result.assign.get((d, s), 0.0)) for s in distances.neighbors_of(d)),
            key=lambda x: x[1],
        )
        if best_s[0][0] == "candidate" and best_s[1] > 0.5:
            n_cells_served_by_new += 1
    new_area_km2 = n_cells_served_by_new * (config.grid_cell_size_m / 1000) ** 2

    return ResultsSummary(
        county=config.county_name,
        n_existing_substations=len(existing),
        n_new_substations=len(built_ids),
        total_new_load_served_kw=per_sub["total_load_kw"].sum() if len(per_sub) else 0.0,
        new_coverage_area_km2=new_area_km2,
        avg_service_dist_before_m=before["avg_dist"],
        avg_service_dist_after_m=after["avg_dist"],
        max_service_dist_before_m=before["max_dist"],
        max_service_dist_after_m=after["max_dist"],
        coverage_gap_before_pct=before["gap_pct"],
        coverage_gap_after_pct=after["gap_pct"],
        n_transmission_constrained=len(constrained_ids),
        transmission_constrained_demand_kw=constrained_demand_kw,
        transmission_expansion_candidates=tx_expansion,
        assignment_cost=after["assignment_cost"],
        build_cost=len(built_ids) * config.fixed_build_cost,
        total_objective=after["assignment_cost"] + len(built_ids) * config.fixed_build_cost,
        mip_gap=result.mip_gap,
        per_substation=per_sub,
    )


def _compute_assignment_metrics(grid: DemandGrid,
                                existing: gpd.GeoDataFrame,
                                cell_ids: list[int]) -> dict:
    """Baseline: assign each cell to nearest existing substation."""
    gap_threshold_m = 10_000.0
    cell_centroids = grid.cells.geometry.centroid
    dists = []
    demands = []

    for d in cell_ids:
        centroid = cell_centroids.loc[d]
        min_dist = min(centroid.distance(pt) for pt in existing.geometry)
        dists.append(min_dist)
        demands.append(grid.demand(d))

    dists = np.array(dists)
    demands = np.array(demands)
    total_demand = demands.sum()

    return {
        "avg_dist": float(np.average(dists, weights=demands)) if total_demand else 0.0,
        "max_dist": float(dists.max()) if len(dists) else 0.0,
        "gap_pct": float((dists > gap_threshold_m).sum() / len(dists) * 100)
                   if len(dists) else 0.0,
    }


def _compute_solved_metrics(result: SolveResult,
                            grid: DemandGrid,
                            distances: SparseDistances,
                            cell_ids: list[int]) -> dict:
    """Metrics from the solved assignment, measured over cell_ids only."""
    gap_threshold_m = 10_000.0
    weighted_dists = []
    demands = []
    max_dist = 0.0
    n_gap = 0
    assignment_cost = 0.0

    for d in cell_ids:
        demand_d = grid.demand(d)
        cell_dist = 0.0
        for s in distances.neighbors_of(d):
            frac = result.assign.get((d, s), 0.0)
            if frac > 0:
                dist = distances.pairs[(d, s)]
                cell_dist += frac * dist
                assignment_cost += demand_d * dist * frac
        weighted_dists.append(cell_dist)
        demands.append(demand_d)
        max_dist = max(max_dist, cell_dist)
        if cell_dist > gap_threshold_m:
            n_gap += 1

    weighted_dists = np.array(weighted_dists)
    demands = np.array(demands)
    total_demand = demands.sum()

    return {
        "avg_dist": float(np.average(weighted_dists, weights=demands))
                    if total_demand else 0.0,
        "max_dist": float(max_dist),
        "gap_pct": float(n_gap / len(cell_ids) * 100) if cell_ids else 0.0,
        "assignment_cost": assignment_cost,
    }


def _build_per_substation_table(result: SolveResult,
                                built_ids: list[int],
                                candidates: Candidates,
                                grid: DemandGrid,
                                distances: SparseDistances,
                                config: OptimizerConfig) -> pd.DataFrame:
    """One row per newly built substation with summary stats."""
    if not built_ids:
        return pd.DataFrame(columns=[
            "candidate_id", "lat", "lon", "n_cells_served",
            "total_load_kw", "avg_service_dist_m",
        ])

    rows = []
    for c in built_ids:
        loc = candidates.location(c)
        loc_geo = (
            gpd.GeoSeries([loc], crs=config.crs_proj)
            .to_crs(config.crs_geo)
            .iloc[0]
        )

        n_cells = 0
        total_load = 0.0
        weighted_dist_sum = 0.0

        for (d, s), frac in result.assign.items():
            if s == ("candidate", c) and frac > 0.01:
                demand_d = grid.demand(d)
                n_cells += 1
                total_load += demand_d * frac
                weighted_dist_sum += distances.pairs[(d, s)] * frac

        avg_dist = weighted_dist_sum / n_cells if n_cells else 0.0

        rows.append({
            "candidate_id": c,
            "lat": loc_geo.y,
            "lon": loc_geo.x,
            "n_cells_served": n_cells,
            "total_load_kw": round(total_load, 1),
            "avg_service_dist_m": round(avg_dist, 0),
        })

    return pd.DataFrame(rows)


def save_csv_outputs(summary: ResultsSummary, config: OptimizerConfig) -> None:
    """Save per-substation recommendations and transmission expansion table as CSVs."""
    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    county = config.county

    sub_path = out_dir / f"{county}_new_substations.csv"
    summary.per_substation.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path.relative_to(BASE_DIR)}")

    if len(summary.transmission_expansion_candidates):
        tx_path = out_dir / f"{county}_tx_expansion.csv"
        summary.transmission_expansion_candidates.to_csv(tx_path, index=False)
        print(f"  Saved: {tx_path.relative_to(BASE_DIR)}")
