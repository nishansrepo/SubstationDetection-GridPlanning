"""Data containers used across the optimizer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


@dataclass
class InputData:
    """Everything loaded from disk or APIs, projected to CRS_PROJ."""
    existing_substations: gpd.GeoDataFrame  # Points
    transmission_lines: Optional[gpd.GeoDataFrame]  # LineStrings, or None
    census_blockgroups: gpd.GeoDataFrame    # Polygons with population
    building_footprints: Optional[pd.DataFrame] = None  # GEOID -> building_area_m2


@dataclass
class DemandGrid:
    """Regular grid of demand cells covering the study region."""
    cells: gpd.GeoDataFrame  # columns: cell_id, geometry, population, demand_kw

    @property
    def cell_ids(self) -> list[int]:
        return self.cells["cell_id"].tolist()

    def demand(self, cell_id: int) -> float:
        return float(self.cells.loc[cell_id, "demand_kw"])

    def centroid(self, cell_id: int) -> Point:
        return self.cells.loc[cell_id, "geometry"].centroid


@dataclass
class Candidates:
    """Filtered candidate sites for new substations."""
    sites: gpd.GeoDataFrame  # columns: candidate_id, geometry (point)

    @property
    def candidate_ids(self) -> list[int]:
        return self.sites["candidate_id"].tolist()

    def location(self, candidate_id: int) -> Point:
        return self.sites.loc[candidate_id, "geometry"]


@dataclass
class SparseDistances:
    """
    Sparse (demand_cell, substation) distance map.
    Keys: (demand_cell_id, ("existing"|"candidate", sub_id)) -> distance_m.
    """
    pairs: dict[tuple[int, tuple[str, int]], float] = field(default_factory=dict)

    # Pre-built index for fast neighbor lookups
    _by_demand: dict[int, list[tuple[str, int]]] = field(
        default_factory=dict, repr=False
    )

    def build_index(self) -> None:
        self._by_demand.clear()
        for d, s in self.pairs:
            self._by_demand.setdefault(d, []).append(s)

    def neighbors_of(self, demand_id: int) -> list[tuple[str, int]]:
        return self._by_demand.get(demand_id, [])


@dataclass
class SolveResult:
    status: str
    objective: float
    mip_gap: float
    build: dict[int, int]                             # candidate_id -> 0/1
    assign: dict[tuple[int, tuple[str, int]], float]  # (d, s) -> fraction
