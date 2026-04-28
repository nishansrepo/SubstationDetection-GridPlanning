"""
Pipeline configuration.

Flexible, declarative configuration for the full dataset extraction pipeline.
All parameters are exposed so that the user controls exactly what gets built.

Two sampling strategies are supported:
    - "curated": Pre-selected counties ensuring landscape diversity.
    - "randomized": Patches drawn from randomly sampled US locations,
      avoiding spatial clustering via minimum inter-patch distance.

Substation type budgets allow fine-grained control, e.g.:
    type_budgets = {"transmission": 500, "distribution": 300, "": 200}
means 500 patches from substations tagged substation=transmission in OSM,
300 from distribution, and 200 from substations with no subtype tag.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PatchConfig:
    """Parameters governing patch extraction geometry."""

    patch_size: int = 512
    """Side length of extracted patches in pixels."""

    resolution: float = 0.6
    """NAIP pixel size in meters."""

    point_buffer_m: float = 75.0
    """Buffer radius (meters) for point-only OSM labels."""

    min_substation_pixels: int = 50
    """Minimum mask-positive pixels for a valid positive sample."""


@dataclass
class SamplingConfig:
    """Parameters governing how patches are sampled."""

    strategy: str = "curated"
    """Sampling strategy: 'curated' or 'randomized'.
    
    'curated'    — Draw from the counties listed in CuratedStrategyConfig.
    'randomized' — Draw substation locations from randomly selected
                   US regions, enforcing spatial dispersion.
    """

    n_positive_total: int = 2000
    """Total positive patches across all substation types.
    Used as fallback when type_budgets is empty."""

    n_negative_total: int = 2000
    """Total negative (background) patches."""

    type_budgets: dict[str, int] = field(default_factory=dict)
    """Per-substation-type budgets. Keys are OSM substation=* tag values.
    Use '' (empty string) for substations with no subtype tag.

    Example:
        {"transmission": 500, "distribution": 300, "": 200}

    When set, n_positive_total is ignored and the sum of type_budgets
    becomes the effective positive count.
    """

    negative_min_distance_m: float = 500.0
    """Minimum distance (meters) from any substation for negative samples."""

    negative_max_attempts: int = 20
    """Max random attempts per negative sample before giving up."""

    jitter_m: float = 50.0
    """Max random offset (meters) from substation centroid for positives."""

    augmentation_copies: int = 1
    """Jittered copies per substation (1 = original only, no extras)."""

    @property
    def n_positive_effective(self) -> int:
        """Effective positive budget: sum of type_budgets if set, else total."""
        if self.type_budgets:
            return sum(self.type_budgets.values())
        return self.n_positive_total


@dataclass
class CuratedStrategyConfig:
    """Parameters specific to the 'curated' sampling strategy."""

    county_geoids: list[str] = field(default_factory=lambda: [
        "06019",  # Fresno, CA — irrigated agriculture, solar farms
        "04013",  # Maricopa, AZ — desert urban
        "48201",  # Harris, TX — humid subtropical, dense suburban
        "53033",  # King, WA — forested urban
        "27053",  # Hennepin, MN — continental, mixed suburban-rural
        "37183",  # Wake, NC — piedmont, varied vegetation
        "08013",  # Boulder, CO — elevation gradients
        "42071",  # Lancaster, PA — eastern deciduous, mixed rural
        "20173",  # Sedgwick, KS — open agricultural
        "35001",  # Bernalillo, NM — sparse vegetation
    ])
    """FIPS GEOIDs of counties to sample from. Edit to add/remove."""

    balance_across_counties: bool = True
    """Allocate roughly equal sample counts per county."""


@dataclass
class RandomizedStrategyConfig:
    """Parameters specific to the 'randomized' sampling strategy."""

    n_random_regions: int = 30
    """Number of random square regions to sample across the US."""

    region_size_km: float = 50.0
    """Side length (km) of each random sampling region."""

    min_inter_patch_distance_m: float = 2000.0
    """Minimum distance between any two patch centers."""

    us_bounds: tuple = (-125.0, 24.5, -66.5, 49.5)
    """Continental US bounding box (west, south, east, north)."""

    exclude_geoids: list[str] = field(default_factory=list)
    """County GEOIDs to exclude (e.g., for a held-out test set).
    Only effective when strategy='randomized'."""


@dataclass
class SplitConfig:
    """Train/val/test split configuration."""

    method: str = "geographic"
    """Split method:
        'geographic' — by county, all patches from a county go to one split.
        'random'     — patches shuffled and split regardless of origin.
    """

    test_fraction: float = 0.15
    """Fraction of counties (geographic) or patches (random) for test."""

    val_fraction: float = 0.15
    """Fraction for validation."""

    test_geoids: list[str] = field(default_factory=list)
    """Explicit county GEOIDs for test. Overrides automatic selection."""

    val_geoids: list[str] = field(default_factory=list)
    """Explicit county GEOIDs for validation."""

    seed: int = 42
    """Separate seed for split randomness."""


@dataclass
class NaipConfig:
    """Parameters for NAIP imagery access."""

    stac_api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection: str = "naip"
    year_range: tuple = (2022, 2024)
    bands: tuple = (1, 2, 3, 4)
    request_delay_s: float = 0.3

    local_naip_paths: list[str] = field(default_factory=list)
    """Paths to local NAIP GeoTIFFs. Tried in order for each coordinate;
    falls back to STAC if none cover the point."""

    max_retries: int = 3
    """Maximum retry attempts for failed STAC tile fetches."""

    verify_checksum: bool = False
    """If True, compute SHA-256 of each fetched patch and log it.
    Catches corrupted or partially downloaded tiles."""


@dataclass
class TemporalConfig:
    """Parameters for imagery-label temporal alignment (Feature 1)."""

    enabled: bool = False
    """If True, check temporal alignment between NAIP and OSM labels."""

    max_gap_years: float = 3.0
    """Maximum acceptable gap (years) between imagery acquisition date
    and OSM label last-edit timestamp. Pairs outside this window get flagged."""

    exclude_mismatched: bool = False
    """If True, exclude temporally mismatched pairs from the dataset.
    If False, include them but set temporal_mismatch=True in metadata."""


@dataclass
class DistanceRasterConfig:
    """Parameters for distance-to-substation raster output (Feature 2)."""

    enabled: bool = False
    """If True, generate a distance raster alongside each binary mask."""

    output_dtype: str = "float32"
    """Data type for the distance raster. float32 is sufficient for
    sub-meter precision within a 512x512 patch."""


@dataclass
class LabelAuditConfig:
    """Parameters for label noise auditing (Feature 4)."""

    enabled: bool = False
    """If True, run the label noise audit after extraction."""

    ndvi_threshold: float = 0.4
    """Pixels with NDVI above this value are flagged as likely vegetation.
    Substation labels overlapping high-NDVI areas indicate labeling errors."""

    ndwi_threshold: float = 0.3
    """Pixels with NDWI above this value are flagged as likely water.
    Default 0.3 based on calibration against Allegheny County substations
    where gravel/concrete surfaces produce NDWI ~0.03."""

    red_band_index: int = 0
    """Index of the red band in the image array (0-based). NAIP R=band1 → index 0."""

    green_band_index: int = 1
    """Index of the green band. NAIP G=band2 → index 1."""

    nir_band_index: int = 3
    """Index of the NIR band. NAIP NIR=band4 → index 3."""


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""

    output_dir: str = "dataset"
    seed: int = 42

    patch: PatchConfig = field(default_factory=PatchConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    curated: CuratedStrategyConfig = field(default_factory=CuratedStrategyConfig)
    randomized: RandomizedStrategyConfig = field(default_factory=RandomizedStrategyConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    naip: NaipConfig = field(default_factory=NaipConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    distance_raster: DistanceRasterConfig = field(default_factory=DistanceRasterConfig)
    label_audit: LabelAuditConfig = field(default_factory=LabelAuditConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def ground_meters(self) -> float:
        return self.patch.patch_size * self.patch.resolution

    def validate(self) -> list[str]:
        """Return list of configuration errors (empty if valid)."""
        errors = []
        if self.sampling.strategy not in ("curated", "randomized"):
            errors.append(
                f"Unknown strategy '{self.sampling.strategy}'. "
                "Must be 'curated' or 'randomized'."
            )
        if self.sampling.strategy == "curated" and not self.curated.county_geoids:
            errors.append("Curated strategy requires at least one county GEOID.")
        if self.sampling.n_positive_effective <= 0:
            errors.append("Total positive budget must be > 0.")
        if self.sampling.n_negative_total < 0:
            errors.append("Negative budget cannot be negative.")
        if self.split.method not in ("geographic", "random"):
            errors.append(
                f"Unknown split method '{self.split.method}'. "
                "Must be 'geographic' or 'random'."
            )
        for k, v in self.sampling.type_budgets.items():
            if v < 0:
                errors.append(f"Negative budget for type '{k}': {v}")
        return errors
