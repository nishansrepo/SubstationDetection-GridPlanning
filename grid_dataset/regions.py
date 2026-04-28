"""
Geographic region definitions.

Provides two mechanisms for defining where patches are sampled:

1. COUNTY_REGISTRY: A static lookup of US counties by FIPS GEOID, used by
   the 'curated' strategy. Users select counties by adding GEOIDs to
   CuratedStrategyConfig.county_geoids.

2. generate_random_regions(): Creates random square sampling regions across
   the continental US, used by the 'randomized' strategy.

The registry is intentionally larger than the default 10-county set so
users can swap counties without modifying this file.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CountySpec:
    """A single US county used for data sampling."""

    geoid: str
    """Five-digit FIPS GEOID (state_fips + county_fips)."""

    name: str
    """Human-readable name."""

    state: str
    """Two-letter state abbreviation."""

    region: str
    """Landscape/climate label for diversity tracking."""

    osm_area_name: str
    """Name string used in Overpass area queries."""

    admin_level: str = "6"
    """OSM admin_level for the boundary (6 = US county)."""

    @property
    def state_fips(self) -> str:
        return self.geoid[:2]

    @property
    def county_fips(self) -> str:
        return self.geoid[2:]


# ------------------------------------------------------------------
# County registry — a broad palette of US landscapes.
# The default curated set uses 10 of these; users can select others.
# ------------------------------------------------------------------
COUNTY_REGISTRY: dict[str, CountySpec] = {spec.geoid: spec for spec in [
    # West
    CountySpec("06019", "Fresno County",     "CA", "central_valley",    "Fresno County"),
    CountySpec("06037", "Los Angeles County", "CA", "coastal_urban",     "Los Angeles County"),
    CountySpec("06071", "San Bernardino Co.", "CA", "inland_desert",     "San Bernardino County"),
    CountySpec("04013", "Maricopa County",    "AZ", "southwest_arid",    "Maricopa County"),
    CountySpec("35001", "Bernalillo County",  "NM", "southern_desert",   "Bernalillo County"),
    CountySpec("32003", "Clark County",       "NV", "basin_desert",      "Clark County"),

    # Pacific Northwest / Mountain
    CountySpec("53033", "King County",     "WA", "pacific_northwest", "King County"),
    CountySpec("41051", "Multnomah County","OR", "pacific_northwest", "Multnomah County"),
    CountySpec("08013", "Boulder County",  "CO", "mountain_front",    "Boulder County"),
    CountySpec("16001", "Ada County",      "ID", "mountain_valley",   "Ada County"),

    # Central / Plains
    CountySpec("48201", "Harris County",   "TX", "gulf_coast",    "Harris County"),
    CountySpec("48453", "Travis County",   "TX", "hill_country",  "Travis County"),
    CountySpec("20173", "Sedgwick County", "KS", "great_plains",  "Sedgwick County"),
    CountySpec("40109", "Oklahoma County", "OK", "southern_plains","Oklahoma County"),
    CountySpec("29189", "St. Louis County","MO", "river_valley",  "St. Louis County"),

    # Midwest
    CountySpec("27053", "Hennepin County", "MN", "upper_midwest",  "Hennepin County"),
    CountySpec("17031", "Cook County",     "IL", "great_lakes",    "Cook County"),
    CountySpec("39049", "Franklin County", "OH", "ohio_valley",    "Franklin County"),
    CountySpec("26163", "Wayne County",    "MI", "great_lakes",    "Wayne County"),

    # Southeast
    CountySpec("37183", "Wake County",     "NC", "southeast",       "Wake County"),
    CountySpec("13121", "Fulton County",   "GA", "deep_south",      "Fulton County"),
    CountySpec("12086", "Miami-Dade County","FL","subtropical",      "Miami-Dade County"),
    CountySpec("47037", "Davidson County", "TN", "mid_south",       "Davidson County"),

    # Mid-Atlantic / Northeast
    CountySpec("42071", "Lancaster County","PA", "mid_atlantic",    "Lancaster County"),
    CountySpec("42003", "Allegheny County","PA", "appalachian",     "Allegheny County"),
    CountySpec("36061", "New York County", "NY", "northeast_urban", "New York County"),
    CountySpec("25017", "Middlesex County","MA", "new_england",     "Middlesex County"),
    CountySpec("24031", "Montgomery Co.",  "MD", "mid_atlantic",    "Montgomery County"),
]}


def get_counties_for_geoids(geoids: list[str]) -> list[CountySpec]:
    """Look up CountySpec objects for a list of FIPS GEOIDs.

    Raises KeyError for unknown GEOIDs with a helpful message.
    """
    result = []
    for geoid in geoids:
        if geoid not in COUNTY_REGISTRY:
            known = ", ".join(sorted(COUNTY_REGISTRY.keys()))
            raise KeyError(
                f"Unknown county GEOID '{geoid}'. "
                f"Known GEOIDs: {known}. "
                f"Add it to COUNTY_REGISTRY in regions.py to use it."
            )
        result.append(COUNTY_REGISTRY[geoid])
    return result


@dataclass
class RandomRegion:
    """A random square region for the 'randomized' strategy."""

    region_id: int
    west: float
    south: float
    east: float
    north: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.south, self.west, self.north, self.east)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.south + self.north) / 2,
            (self.west + self.east) / 2,
        )


def generate_random_regions(
    n_regions: int,
    region_size_km: float,
    us_bounds: tuple[float, float, float, float],
    rng: np.random.Generator,
    min_region_distance_km: float = 100.0,
) -> list[RandomRegion]:
    """Generate spatially dispersed random sampling regions.

    Parameters
    ----------
    n_regions : int
        Number of regions to generate.
    region_size_km : float
        Side length of each square region in km.
    us_bounds : tuple
        (west, south, east, north) in WGS-84.
    rng : np.random.Generator
        Random number generator.
    min_region_distance_km : float
        Minimum distance between region centers to prevent clustering.

    Returns
    -------
    list of RandomRegion
    """
    w, s, e, n = us_bounds
    half_deg_lat = (region_size_km / 2) / 111.0
    half_deg_lon = (region_size_km / 2) / 85.0  # approx at 40°N

    min_dist_deg = min_region_distance_km / 111.0

    regions = []
    attempts = 0
    max_attempts = n_regions * 50

    while len(regions) < n_regions and attempts < max_attempts:
        attempts += 1
        cx = rng.uniform(w + half_deg_lon, e - half_deg_lon)
        cy = rng.uniform(s + half_deg_lat, n - half_deg_lat)

        # Check distance from existing regions
        too_close = False
        for existing in regions:
            ec = existing.center
            dist = np.sqrt((cy - ec[0]) ** 2 + (cx - ec[1]) ** 2)
            if dist < min_dist_deg:
                too_close = True
                break

        if too_close:
            continue

        regions.append(RandomRegion(
            region_id=len(regions),
            west=cx - half_deg_lon,
            south=cy - half_deg_lat,
            east=cx + half_deg_lon,
            north=cy + half_deg_lat,
        ))

    return regions
