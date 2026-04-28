"""Configuration: area presets, paths, and OptimizerConfig dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / Path("data")
CENSUS_DIR = DATA_DIR / "census"
TIGER_YEAR = 2025

# Each preset defines a manageable analysis area.
# bbox = [west, south, east, north] in WGS-84.
# Use city/metro-level bounding boxes, NOT full counties.
# Large counties (Maricopa: 24,000 km²) are impractical at NAIP resolution.

COUNTY_PRESETS: dict[str, dict] = {
    # ---------- Phoenix Metro (practical subsets of Maricopa) ----------
    "maricopa": {
        "geoid": "04013", "state_fips": "04", "state_name": "Arizona",
        "county_fips": "013", "crs_proj": "EPSG:32612",
        "county_name": "Maricopa County",
        "bbox": [-112.15, 33.35, -111.85, 33.55],  # Central Phoenix (~25×22 km, ~5k patches)
        "note": "Central Phoenix metro. Full county is 24,000 km² and impractical at NAIP resolution.",
    },
    "phoenix_east": {
        "geoid": "04013", "state_fips": "04", "state_name": "Arizona",
        "county_fips": "013", "crs_proj": "EPSG:32612",
        "county_name": "Maricopa County",
        "bbox": [-111.90, 33.35, -111.70, 33.50],  # Mesa-Tempe (~17×17 km, ~2.7k patches)
        "note": "Mesa-Tempe area. High-growth corridor east of Phoenix.",
    },
    "phoenix_west": {
        "geoid": "04013", "state_fips": "04", "state_name": "Arizona",
        "county_fips": "013", "crs_proj": "EPSG:32612",
        "county_name": "Maricopa County",
        "bbox": [-112.40, 33.40, -112.20, 33.55],  # Goodyear-Avondale (~17×17 km)
        "note": "West Valley growth corridor. Fastest-growing area in AZ.",
    },
    # ---------- Other counties (most are small enough as-is) ----------
    "fresno": {
        "geoid": "06019", "state_fips": "06", "state_name": "California",
        "county_fips": "019", "crs_proj": "EPSG:32611",
        "county_name": "Fresno County",
        "bbox": [-119.87, 36.70, -119.68, 36.83],  # Fresno city (~16×14 km)
    },
    "allegheny": {
        "geoid": "42003", "state_fips": "42", "state_name": "Pennsylvania",
        "county_fips": "003", "crs_proj": "EPSG:32617",
        "county_name": "Allegheny County",
        # Allegheny is small enough (~1,900 km²) to use full county
    },
    "harris": {
        "geoid": "48201", "state_fips": "48", "state_name": "Texas",
        "county_fips": "201", "crs_proj": "EPSG:32615",
        "county_name": "Harris County",
        "bbox": [-95.55, 29.65, -95.25, 29.85],  # Houston core (~25×22 km)
    },
    "king": {
        "geoid": "53033", "state_fips": "53", "state_name": "Washington",
        "county_fips": "033", "crs_proj": "EPSG:32610",
        "county_name": "King County",
        "bbox": [-122.40, 47.50, -122.20, 47.68],  # Seattle-Bellevue (~17×20 km)
    },
    "lancaster": {
        "geoid": "42071", "state_fips": "42", "state_name": "Pennsylvania",
        "county_fips": "071", "crs_proj": "EPSG:32618",
        "county_name": "Lancaster County",
    },
    "hennepin": {
        "geoid": "27053", "state_fips": "27", "state_name": "Minnesota",
        "county_fips": "053", "crs_proj": "EPSG:32615",
        "county_name": "Hennepin County",
    },
    "wake": {
        "geoid": "37183", "state_fips": "37", "state_name": "North Carolina",
        "county_fips": "183", "crs_proj": "EPSG:32617",
        "county_name": "Wake County",
    },
    "boulder": {
        "geoid": "08013", "state_fips": "08", "state_name": "Colorado",
        "county_fips": "013", "crs_proj": "EPSG:32613",
        "county_name": "Boulder County",
    },
    "sedgwick": {
        "geoid": "20173", "state_fips": "20", "state_name": "Kansas",
        "county_fips": "173", "crs_proj": "EPSG:32614",
        "county_name": "Sedgwick County",
    },
    "bernalillo": {
        "geoid": "35001", "state_fips": "35", "state_name": "New Mexico",
        "county_fips": "001", "crs_proj": "EPSG:32613",
        "county_name": "Bernalillo County",
    },
}


@dataclass
class OptimizerConfig:
    county: str = "maricopa"
    crs_geo: str = "EPSG:4326"

    # Analysis bounding box [west, south, east, north] in WGS-84.
    # If None, uses the full county extent from Census geometries.
    # For large counties, ALWAYS set a bbox to keep data sizes manageable.
    bbox: Optional[list[float]] = None

    # Demand grid
    grid_cell_size_m: float = 3000.0
    pop_to_peak_kw: float = 2.0
    residential_kw_per_m2: float = 0.03

    # Candidate generation
    min_candidate_demand_kw: float = 200.0

    # MILP parameters
    fixed_build_cost: float = 1.5e6
    max_service_radius_m: float = 20000.0
    max_coverage_dist_m: float = 15000.0
    max_new_substations: Optional[int] = 50

    # Substation capacity
    default_candidate_capacity_kw: float = 200_000.0
    default_existing_capacity_kw: float = 200_000.0
    skip_capacity: bool = False

    # Input paths
    metadata_path: Path = DATA_DIR / "metadata.csv"
    dc_atlas_path: Path = DATA_DIR / "im3_open_source_data_center_atlas/im3_open_source_data_center_atlas.csv"

    # Data center
    dc_candidate_radius_m: float = 10_000.0
    dc_min_sqft: float = 0.0

    # Census
    census_year: int = 2020

    # Solver
    solver_time_limit_s: int = 300
    solver_gap_rel: float = 0.01

    @property
    def preset(self) -> dict:
        return COUNTY_PRESETS[self.county]

    @property
    def crs_proj(self) -> str:
        return self.preset["crs_proj"]

    @property
    def county_geoid(self) -> str:
        return self.preset["geoid"]

    @property
    def state_fips(self) -> str:
        return self.preset["state_fips"]

    @property
    def state_name(self) -> str:
        return self.preset["state_name"]

    @property
    def county_fips(self) -> str:
        return self.preset["county_fips"]

    @property
    def county_name(self) -> str:
        return self.preset["county_name"]

    @property
    def effective_bbox(self) -> Optional[list[float]]:
        """Returns the bbox to use: explicit override > preset > None (full county)."""
        if self.bbox is not None:
            return self.bbox
        return self.preset.get("bbox")
