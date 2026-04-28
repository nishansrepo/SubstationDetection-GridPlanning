# Grid Infrastructure Expansion Optimizer (p-median variant)

Recommends where to build new electrical substations by solving a Mixed Integer Linear Program (MILP). Takes detected substation locations from a satellite imagery model plus public datasets, then identifies underserved areas where new infrastructure would most reduce service distances.

## Data

### Required

- **Model detections** (`data/metadata.csv`) — output of the substation detection pipeline. Uses `label`, `center_lon`, `center_lat`, `county_geoid` columns. Each positive detection is treated as an existing substation location.
- **Census population** — block group population from the 2020 Decennial Census, read from `data/census/pop_<county>.json`.
- **TIGER shapefiles** — block group geometries for spatial joins and areal interpolation, read from `data/census/tl_2025_<state_fips>_bg/`.
- **Building footprints** — total footprint area per block group from Microsoft USBuildingFootprints. Better spatial proxy than population alone. **Manual download required** (multi-GB state files). Place in `data/buildings/{state_name}_footprints.parquet` or `.geojson` (e.g. `Arizona_footprints.geojson`). See [USBuildingFootprints](https://github.com/microsoft/USBuildingFootprints).
- **Transmission lines** — OSM `power=line` / `power=minor_line` geometries read from `data/osm/{county}/power_line.geojson`. Used both for candidate proximity filtering and for ranking transmission-expansion planning spots. The filter is skipped when unavailable.
- **Data centers** — locations from the IM3 open-source data center atlas (`data/im3_open_source_data_center_atlas/im3_open_source_data_center_atlas.csv`). Any data center with no candidate site within `dc_candidate_radius_m` is injected as a priority candidate, bypassing the demand-threshold and transmission-proximity filters.

### Demand proxy

Demand per grid cell is the sum of all available signals:

```
demand_kw = population × 2.0
          + building_area_m2 × 0.03
```

Population is always included. Building area adds ~30 W/m² of residential/commercial load and sharpens the signal in areas where population alone underestimates load density.

## How to Run

```bash
# From the project root:
python -m new_optimizer                           # Maricopa County (default)
python -m new_optimizer --county fresno
python -m new_optimizer --max-new 20 --time-limit 300
python -m new_optimizer --sensitivity             # sweep max_new ∈ {10,25,50,75,100}
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--county` | `maricopa` | County key (fresno, maricopa, allegheny, harris, king, lancaster, hennepin, wake, boulder, sedgwick, bernalillo) |
| `--grid-cell-size` | `3000` | Demand grid cell size in meters |
| `--build-cost` | `1.5e6` | Fixed cost per new substation (dollars) |
| `--max-new` | `50` | Maximum number of new substations |
| `--max-radius` | `20000` | Max service radius in meters (sparsification cutoff) |
| `--max-coverage-dist` | `15000` | Hard coverage constraint — cells with any option within this radius must be served within it |
| `--time-limit` | `300` | Solver time limit in seconds |
| `--metadata` | `data/metadata.csv` | Path to model detection CSV |
| `--sensitivity` | off | Run sensitivity sweep over `max_new` values |

### Dependencies

```
geopandas, shapely, pandas, numpy, scipy, pulp, matplotlib
```

Solver: HiGHS (bundled with PuLP).

## Objective Function

Classic p-median with a fixed build cost:

```
minimize:  Σ_{d,s} demand_d × dist(d,s) × assign[d,s]   (assignment cost)
         + fixed_build_cost × Σ_c build[c]              (build cost)
```

**Assignment cost** — the demand-weighted total service distance across all (cell, substation) pairs. Pairs outside `max_service_radius_m` are pruned from the sparse distance matrix and never enter the model.

**Build cost** — a flat dollar cost (`--build-cost`, default $1.5M) per new substation. Trading a dollar-valued build term against a demand×meters assignment term isn't unit-homogeneous by default; tune `--build-cost` to shift the count of recommended builds.

## Constraints

1. **Full coverage** — every demand cell must be fully assigned: `Σ_s assign[d,s] = 1`.
2. **Linking** — a demand cell can only be assigned to a candidate if that candidate is built: `assign[d,c] ≤ build[c]`.
3. **Capacity** — total demand assigned to each substation must not exceed its capacity: `Σ_d demand_d × assign[d,s] ≤ cap[s]`. Existing substations use voltage-derived capacity (500 kV → 500 MW, 230 kV → 200 MW, …, distribution → 20 MW); candidates use `default_candidate_capacity_kw` (150 MW).
4. **Coverage distance** — for every demand cell that has at least one substation within `max_coverage_dist_m`, the cell must be fully served from within that radius: `Σ_{s: dist(d,s) ≤ threshold} assign[d,s] = 1`. Cells with no option inside the threshold are skipped (they fall back to the wider `max_service_radius_m`) and reported as transmission-constrained.
5. **Substation cap** (optional) — `Σ build[c] ≤ N` when `--max-new` is set.

## Results

Output is written to `output/` (at the project root):

### `{county}_new_substations.csv`

One row per newly built substation:

| Column | Description |
|--------|-------------|
| `candidate_id` | Grid cell ID |
| `lat`, `lon` | Location (WGS84) |
| `n_cells_served` | Number of demand cells primarily assigned |
| `total_load_kw` | Total demand assigned |
| `avg_service_dist_m` | Demand-weighted average distance to assigned cells |

### `{county}_tx_expansion.csv`

Transmission-constrained clusters — demand clusters with no candidate within the coverage radius. These are **not** substation build recommendations; they are planning spots where transmission expansion would unlock the most load per km of required grid extension:

| Column | Description |
|--------|-------------|
| `lat`, `lon` | Demand-weighted centroid of the cluster |
| `n_cells` | Number of constrained cells in cluster |
| `cluster_demand_kw` | Total demand in cluster |
| `dist_to_tx_km` | Distance to nearest transmission line |
| `dist_to_sub_km` | Distance to nearest existing substation |
| `score` | `cluster_demand_kw / dist_to_tx_km` — load unlocked per km of extension |

Clusters below 300 kW are dropped as noise; top 10 by score are reported.

### `{county}_results_map.png`

Overview map: county outline, demand grid colored by load, existing substations (grey dots), new substations (red stars sized by load served), transmission-expansion clusters (orange diamonds), and transmission-constrained cells shaded grey.

### `{county}_coverage_map.png`

Service-distance heatmap. Each reachable cell is colored by its post-optimization distance to its assigned substation (0–5 km green → 15–20 km red). Transmission-constrained cells appear in neutral grey.

### `{county}_sensitivity.csv` / `{county}_sensitivity.png`

Only written when `--sensitivity` is passed. Shows how `n_built`, coverage gap %, avg/max service distance, load shifted, and new-coverage area respond to the build budget `max_new ∈ {10, 25, 50, 75, 100}`.

## Console summary

Printed at the end of every run:

- Existing / new substation counts
- Load shifted to new substations (kW)
- New coverage area (km²)
- Avg and max service distance, before vs. after
- Coverage gap % (>10 km) on reachable cells, before vs. after
- Transmission-constrained cell count and demand
- Objective cost breakdown (assignment, build, total) and MIP gap
- Per-substation table
- Top transmission-expansion planning clusters
