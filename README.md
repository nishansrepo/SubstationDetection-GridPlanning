# Grid Infrastructure Detection & Planning Pipeline

Detect existing electrical substations from aerial imagery using deep learning, then recommend where to build new ones using mathematical optimization.

**Input**: A geographic bounding box  
**Output**: Ranked substation site recommendations with street addresses, load estimates, and land-use suitability flags — plotted on an interactive satellite map.

> **Model Training Code:** See [`substation_training.ipynb`](substation_training.ipynb) — a self-contained notebook covering dataset loading, 5-fold cross-validation training, ensemble construction, calibration, and evaluation. It is the single source of truth for how the detection model was produced.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Who This Is For](#who-this-is-for)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Working with Bounding Boxes](#working-with-bounding-boxes)
- [Available Presets](#available-presets)
- [Step-by-Step Walkthrough](#step-by-step-walkthrough)
- [Output Files](#output-files)
- [Preloaded Demo Output](#preloaded-demo-output)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Training Notebook](#training-notebook)
- [Optimizer Details](#optimizer-details)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Authors](#authors)

---

## What This Does

The pipeline answers two questions:

1. **Where are electrical substations right now?**
   A 5-fold ensemble U-Net/ResNet-34 model scans NAIP aerial imagery (0.6m resolution, 4-band R/G/B/NIR) and segments substation footprints at pixel level.

2. **Where should new substations be built?**
   A p-median MILP optimizer combines detected substations with Census population, Microsoft Building Footprints, and OpenStreetMap transmission lines to recommend sites that minimize demand-weighted service distance. Each recommendation is reverse-geocoded with a street address and checked against nearby parks, wetlands, and other restricted land uses.

The entire process runs from a single command and completes in 10–30 minutes depending on area size.

---

## Who This Is For

Utilities typically know their own assets. This tool is for everyone else:

- **Regional planners** coordinating across utility boundaries where records are siloed or inconsistent
- **Regulators** assessing service equity — which communities are farthest from existing substations?
- **Infrastructure developers** identifying underserved markets for distributed energy, EV charging, or data centers
- **Disaster response teams** who need infrastructure assessment from recent imagery without utility cooperation
- **Researchers** studying grid equity, urban growth, or climate adaptation at regional scale

> **This is a planning support tool, not an engineering design tool.** Every recommendation means "this location warrants a detailed feasibility study" — not "build here."

---

## How It Works

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    DETECTION PHASE                              │
 │                                                                 │
 │  Bounding Box ──► NAIP Tiles ──► 512×512 Patches ──► U-Net      │
 │                   (from STAC)    (0.6m, 4-band)      Ensemble   │
 │                                                        │        │
 │                                              ┌─────────▼──────┐ │
 │                                              │  metadata.csv  │ │
 │                                              │  (detections)  │ │
 │                                              └─────────┬──────┘ │
 ├────────────────────────────────────────────────────────│────────┤
 │                    OPTIMIZATION PHASE                  │        │
 │                                                        ▼        │
 │  Census Pop ─────┐                            ┌──────────────┐  │
 │  MS Buildings ───┼──► Demand Grid ──► MILP ──►│ Recommended  │  │
 │  OSM TX Lines ───┘    (kW per cell)   Solver  │ Sites + Map  │  │
 │                                               └──────────────┘  │
 ├─────────────────────────────────────────────────────────────────┤
 │                    SITE ANALYSIS PHASE                          │
 │                                                                 │
 │  Each site ──► Reverse Geocode ──► Land-Use Check ──► Map       │
 │                (street address)    (parks/wetlands)   (Folium)  │
 └─────────────────────────────────────────────────────────────────┘
```

The detection and optimization phases are connected by a single file: `metadata.csv`. This means you can swap in your own substation locations (from any source) and still use the optimizer, or use the detector standalone for asset mapping.

---

## Quick Start

### Prerequisites

- Python 3.10+
- ~7 GB disk space for demo imagery
- ~500 MB for model weights
- Internet connection (for NAIP download and Census/OSM queries)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_REPO/grid-siting-pipeline.git
cd grid-siting-pipeline
pip install -r requirements.txt
```

### 2. Download model weights

The 5-fold ensemble model (~500 MB) is hosted on Google Drive:

```bash
python scripts/download_assets.py --model
```

This places the checkpoint at `model/ensemble_model.pt`. You can also download manually and place it there.

### 3. Extract image patches for your area

```bash
# Central Phoenix (~26×22 km, ~7k patches, takes ~30 min)
python scripts/extract_region.py --county maricopa -o demo_data -v
```

This downloads NAIP tiles from Microsoft Planetary Computer, extracts 512×512 patches at 0.6m resolution (native 0.3m downsampled via Lanczos), deduplicates overlapping tiles, and builds `demo_data/metadata.csv`. No pre-existing data is needed.

**Tip:** Run `--dry-run` first to see how many patches and how much disk space your area will need before committing to the download.

### 4. Fetch auxiliary data

```bash
python scripts/fetch_data.py --county maricopa
```

This downloads Microsoft Building Footprints (state-level GeoJSON, filtered to county bbox, saved as Parquet) and OSM power transmission lines (via Overpass API). Without this step, the optimizer falls back to population-only demand estimation, which still works but is less accurate in commercial/industrial areas.

### 5. Run the full pipeline

```bash
python scripts/local_pipeline.py \
    --county maricopa \
    --eval-csv demo_data/metadata.csv \
    --img-dir demo_data/images \
    --model-path model/ensemble_model.pt \
    --no-capacity \
    -o output/maricopa -v
```

Open `output/maricopa_satellite_map.html` in a browser to see the results on satellite imagery.

### Skip to the results

If you want to see what the output looks like before running anything, the repo ships with a `preloaded_demo_output/` folder containing a complete Maricopa County (Central Phoenix) run. Open `preloaded_demo_output/maricopa_satellite_map.html` directly.

---

## Working with Bounding Boxes

### Why bounding boxes, not counties?

NAIP imagery at 0.6m resolution is enormous. A full county like Maricopa (24,000 km²) would require ~260,000 patches and over 100 GB of storage. The MILP solver also cannot handle the resulting problem size in reasonable time. Instead, work at **city or metro scale** (20–80 km per side).

This mirrors how infrastructure planning actually works — utilities plan by service territory and growth corridor, not by arbitrary county boundaries.

### How to choose a bounding box

1. Go to [bboxfinder.com](http://bboxfinder.com) or Google Maps
2. Navigate to your area of interest
3. Draw a rectangle (~25–50 km per side for best results)
4. Note the coordinates as **west, south, east, north** in decimal degrees
5. Pass via `--bbox`:

```bash
python scripts/extract_region.py \
    --county maricopa \
    --bbox -112.10 33.38 -111.90 33.52 \
    -o demo_data -v
```

The `--county` flag is still required because it tells the pipeline which state FIPS code and UTM zone to use. The `--bbox` overrides the preset's default bounding box. For disk use intuition, bounding box area sizes up to 30×30 km are feasibly run on a machine with 10 GB disk space.

---

## Available Presets

Each preset packages a state FIPS code, county GEOID, UTM zone, and a default bounding box. Pass `--county <key>` to any script that needs geographic context.

| Key | Area | Approx. Size | Notes |
|-----|------|-------------|-------|
| `maricopa` | Central Phoenix, AZ | 26×22 km | Default demo area |
| `phoenix_east` | Mesa-Tempe, AZ | 17×17 km | High-growth corridor |
| `phoenix_west` | Goodyear-Avondale, AZ | 17×17 km | Fastest-growing area in AZ |
| `fresno` | Fresno city, CA | 16×14 km | Central Valley |
| `allegheny` | Pittsburgh metro, PA | Full county | Small enough as-is |
| `harris` | Houston core, TX | 25×22 km | Dense urban |
| `king` | Seattle-Bellevue, WA | 17×20 km | Pacific NW |
| `lancaster` | Lancaster, PA | Full county | Rural mid-Atlantic |
| `hennepin` | Minneapolis, MN | Full county | Twin Cities |
| `wake` | Raleigh, NC | Full county | Research Triangle |
| `boulder` | Boulder, CO | Full county | Front Range |
| `sedgwick` | Wichita, KS | Full county | Great Plains |
| `bernalillo` | Albuquerque, NM | Full county | Rio Grande Valley |

To add a new area, add an entry to `COUNTY_PRESETS` in `optimizer/config.py`. You need the county GEOID, state FIPS, UTM zone EPSG, and an optional bounding box in `[west, south, east, north]` format.

---

## Step-by-Step Walkthrough

### Extracting patches for a new area

```bash
# 1. Dry run — see how many patches and estimated disk usage
python scripts/extract_region.py --county maricopa --dry-run

# 2. Full extraction (downloads from Planetary Computer)
python scripts/extract_region.py --county maricopa -o demo_data -v

# 3. Resume after interruption (skips already-downloaded patches)
python scripts/extract_region.py --county maricopa -o demo_data --skip-existing -v

# 4. Custom bounding box within a preset county
python scripts/extract_region.py \
    --county maricopa \
    --bbox -112.10 33.38 -111.90 33.52 \
    -o my_custom_area -v

# 5. Restrict to specific NAIP years
python scripts/extract_region.py \
    --county maricopa \
    --year-min 2022 --year-max 2023 \
    -o demo_data -v
```

### Fetching auxiliary data

```bash
# Both buildings and power lines (default)
python scripts/fetch_data.py --county maricopa -v

# Buildings only (skips Overpass API query)
python scripts/fetch_data.py --county maricopa --skip-osm

# Power lines only (skips ~800 MB state download)
python scripts/fetch_data.py --county maricopa --skip-buildings
```

### Running the full pipeline

```bash
# Standard run (recommended flags)
python scripts/local_pipeline.py \
    --county maricopa \
    --eval-csv demo_data/metadata.csv \
    --img-dir demo_data/images \
    --model-path model/ensemble_model.pt \
    --no-capacity \
    -o output/maricopa -v

# Custom bounding box (optimizer only analyzes this subarea)
python scripts/local_pipeline.py \
    --county maricopa \
    --bbox -112.05 33.40 -111.90 33.50 \
    --eval-csv demo_data/metadata.csv \
    --img-dir demo_data/images \
    --model-path model/ensemble_model.pt \
    --no-capacity \
    -o output/central_phx -v

# Skip detection, reuse existing detections
python scripts/local_pipeline.py \
    --county maricopa \
    --skip-detection \
    --metadata output/maricopa/metadata.csv \
    --no-capacity \
    -o output/maricopa -v

# With sensitivity analysis (sweeps build budget from 10 to 100)
python scripts/local_pipeline.py \
    --county maricopa \
    --eval-csv demo_data/metadata.csv \
    --img-dir demo_data/images \
    --model-path model/ensemble_model.pt \
    --no-capacity --sensitivity \
    -o output/maricopa -v

# Skip site analysis (faster, no Nominatim/OSM queries)
python scripts/local_pipeline.py \
    --county maricopa \
    --eval-csv demo_data/metadata.csv \
    --img-dir demo_data/images \
    --model-path model/ensemble_model.pt \
    --no-capacity --skip-analysis \
    -o output/maricopa -v
```

---

## Output Files

A successful pipeline run produces these files in the output directory:

| File | Description |
|------|-------------|
| `maricopa/metadata.csv` | Detection results: lat, lon, confidence for each patch |
| `maricopa/detections.geojson` | Detection centroids as GeoJSON (open in QGIS/ArcGIS) |
| `maricopa/detection_map.html` | Interactive map of detections only (QA the model) |
| `maricopa_new_substations.csv` | Recommended new sites with coordinates, load, cells served |
| `maricopa_site_analysis.csv` | Sites enriched with street addresses and suitability flags |
| `maricopa_satellite_map.html` | **Interactive satellite map** — best way to review results |
| `maricopa_results_map.png` | Demand grid with existing + new substations (for reports) |
| `maricopa_coverage_map.png` | Service distance heatmap, before and after |
| `maricopa_tx_expansion.csv` | Areas needing transmission lines, not substations |
| `maricopa_sensitivity.csv/png` | Budget sweep results (only with `--sensitivity`) |

---

## Preloaded Demo Output

The repository includes a `preloaded_demo_output/` directory with a complete pipeline run on Central Phoenix, AZ so you can inspect outputs immediately without running anything:

```
preloaded_demo_output/
├── maricopa_results_map.png
├── maricopa_coverage_map.png
├── maricopa_satellite_map.html        ← open this in a browser
├── maricopa_new_substations.csv
├── maricopa_site_analysis.csv
└── maricopa/
    ├── metadata.csv
    ├── detections.geojson
    └── detection_map.html
```

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── substation_training.ipynb          # ★ Model training & evaluation notebook
│
├── scripts/                           # All runnable scripts
│   ├── local_pipeline.py              # ★ Main pipeline: detect → optimize → analyze
│   ├── extract_region.py              # Download NAIP patches for a bounding box
│   ├── fetch_data.py                  # Download MS Buildings + OSM power lines
│   ├── download_assets.py             # Download model weights from Google Drive
│   ├── build_dataset.py               # Build training datasets from NAIP + OSM
│   ├── county_pipeline.py             # Training-time county-level pipeline
│   ├── predict_county.py              # Run inference on a single county's patches
│   ├── evaluate_model.py              # Evaluate model on labeled test datasets
│   ├── test_new_counties.py           # Evaluate on new/unseen county data
│   ├── extract_eval.py                # Extract evaluation metrics from run logs
│   ├── demo_qa.py                     # QA checks on demo_data patches
│   └── config.py                      # Shared script-level configuration
│
├── optimizer/                         # Substation siting optimizer (MILP)
│   ├── config.py                      # Area presets + OptimizerConfig dataclass
│   ├── pipeline.py                    # Optimizer orchestration
│   ├── data_loader.py                 # Census, TIGER shapefiles, buildings, TX lines
│   ├── demand_grid.py                 # Areal-interpolated demand grid (kW per cell)
│   ├── candidates.py                  # Candidate site generation
│   ├── distances.py                   # Sparse distance matrix (KDTree)
│   ├── containers.py                  # Named data containers
│   ├── model.py                       # PuLP MILP formulation + HiGHS solver
│   ├── results.py                     # Metrics + TX expansion ranking
│   ├── visualize.py                   # Results map + coverage heatmap
│   ├── site_analysis.py               # Reverse geocoding + land-use suitability
│   ├── sensitivity.py                 # Budget sensitivity sweep
│   ├── cli.py                         # Standalone optimizer CLI entry point
│   ├── __main__.py                    # python -m optimizer
│   ├── __init__.py
│   └── README.md                      # Optimizer-specific documentation
│
├── grid_dataset/                      # Detection data pipeline package
│   ├── config.py                      # Pipeline configuration
│   ├── regions.py                     # County registry (27 US counties)
│   ├── osm_labels.py                  # Overpass API substation polygon fetcher
│   ├── naip_source.py                 # NAIP reader (local files + STAC COGs)
│   ├── patch_extractor.py             # 512×512 windowed reads + mask rasterization
│   ├── negative_sampler.py            # Distance-filtered negative patch sampling
│   ├── dataset_builder.py             # Training pipeline orchestrator
│   ├── splitter.py                    # Geographic train/val/test splitting
│   ├── quality.py                     # Post-extraction QA checks
│   ├── label_audit.py                 # NDVI/NDWI label noise detection
│   ├── temporal_check.py              # Imagery-vs-label temporal alignment
│   └── __init__.py
│
├── model/                             # Model weights (not in repo; download via script)
│   ├── .gitkeep
│   └── ensemble_model.pt              # 5-fold U-Net/ResNet-34 (~500 MB)
│
├── demo_data/                         # Extracted patches (not in repo; generate via script)
│   ├── metadata.csv                   # Patch locations + labels
│   └── images/                        # ~7,236 TIF patches (512×512, 4-band)
│
├── data/                              # Runtime data (not in repo; auto-fetched)
│   ├── buildings/                     # MS Building Footprints (.parquet)
│   ├── osm/                           # Power lines per county (.geojson)
│   └── census/                        # Block group population + TIGER shapefiles
│
└── preloaded_demo_output/             # Pre-computed demo results (tracked in repo)
    ├── maricopa_results_map.png
    ├── maricopa_coverage_map.png
    ├── maricopa_satellite_map.html
    ├── maricopa_new_substations.csv
    ├── maricopa_site_analysis.csv
    └── maricopa/
        ├── metadata.csv
        ├── detections.geojson
        └── detection_map.html
```

### Script roles at a glance

The `scripts/` directory contains both **user-facing pipeline scripts** and **development/evaluation utilities**:

**Pipeline scripts** (run these):

| Script | Purpose | When to use |
|--------|---------|-------------|
| `local_pipeline.py` | Full detect → optimize → analyze pipeline | Primary entry point |
| `extract_region.py` | Download NAIP patches for a bounding box | Before first pipeline run |
| `fetch_data.py` | Download MS Buildings + OSM power lines | Before first pipeline run |
| `download_assets.py` | Download model weights from Google Drive | One-time setup |

**Development/evaluation scripts** (used during model training and validation):

| Script | Purpose |
|--------|---------|
| `build_dataset.py` | Build labeled training datasets from NAIP + OSM |
| `county_pipeline.py` | Training-time county-level data pipeline |
| `predict_county.py` | Run inference on a single county's patches |
| `evaluate_model.py` | Compute IoU/Dice/AUC on labeled test sets |
| `test_new_counties.py` | Evaluate generalization on unseen counties |
| `extract_eval.py` | Extract evaluation metrics from logs |
| `demo_qa.py` | QA checks on extracted demo patches |
| `config.py` | Shared configuration for scripts |

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | U-Net with ResNet-34 encoder |
| Input | 4-band (R, G, B, NIR), 512×512 pixels at 0.6m |
| Training | 5-fold cross-validation across 10 US counties |
| Inference | Average sigmoid probabilities across all 5 folds |
| Library | segmentation-models-pytorch |
| Checkpoint format | `{fold_01: state_dict, ..., fold_05: state_dict}` |

### Why these choices?

- **U-Net**: Designed for precise boundary delineation with limited training data — both apply to substation segmentation where labeled examples are scarce.
- **ResNet-34** (not 50/101): Substations have relatively simple visual signatures (gravel pads, metal structures, clearings). Deeper encoders add inference cost without meaningful accuracy gains for this task.
- **4 bands**: The NIR channel helps discriminate metal and gravel from vegetation. Removing NIR drops IoU by approximately 0.08 in our ablation tests.
- **Ensemble of 5 folds**: Reduces variance across geographic regions and improves probability calibration, which matters because the optimizer downstream uses detection confidence.

### Performance

| Dataset | Patches | Mean IoU | Mean Dice | Patch AUC |
|---------|---------|----------|-----------|-----------|
| eval_dataset (in-distribution) | 3,086 | 0.739 | 0.801 | 0.993 |
| extra_val (out-of-distribution) | 822 | 0.650 | 0.724 | 0.987 |

Patch-level detection performance: 99.8% precision (2 false positives out of 1,717 negatives), 88.2% recall.

Best-performing county: Maricopa, AZ (0.894 IoU). Worst: King, WA (0.553 IoU). Performance degrades in heavily forested regions where tree canopy occludes infrastructure.

---

## Training Notebook

[`substation_training.ipynb`](substation_training.ipynb) is a self-contained Jupyter notebook (~2,700 lines of code across 12 code cells) that documents the complete model development workflow. It runs in Google Colab or locally and produces the `ensemble_model.pt` checkpoint used by the inference pipeline. By default it loads the cached ensemble and reproduces evaluation metrics without retraining; setting `run_full_training = True` reruns everything from scratch.

### What the notebook covers

**1. Runtime Setup** — Installs missing dependencies, mounts Google Drive (Colab), and imports all packages for raster I/O, training, and evaluation.

**2. Configuration** — Centralizes all hyperparameters, paths, and execution switches in a single config dictionary for auditability:

| Parameter | Value |
|-----------|-------|
| Encoder | ResNet-34 (ImageNet pretrained) |
| Training schedule | 2 epochs frozen encoder (lr=1e-3) → 12 epochs full fine-tune (lr=1e-4) |
| Batch size | 8 (GPU) / 4 (CPU) |
| Folds | 5 (StratifiedGroupKFold, grouped by `county_geoid`) |
| Aggregation | Max across fold probabilities |
| Patch threshold | 0.03 (top-1% pixel scoring) |
| Loss | Dice + BCE (combined) |

**3. Dataset Loading & Fold Plan** — Loads curated train/val/val2/test splits and reconstructs the 5-fold cross-validation plan from the `train + val2` pool. Folds are grouped by county so that no county appears in both training and validation within the same fold, giving a conservative estimate of geographic generalization. An optional label-audit filter removes patches flagged by NDVI/NDWI noise detection.

**4. Training Functions** — Defines raster reading (4-band GeoTIFF normalization with per-fold channel statistics), data augmentation, the U-Net model via `segmentation-models-pytorch`, a two-stage training schedule (frozen encoder then full fine-tuning with cosine annealing), and early stopping by validation Dice.

**5. Cross-Validation Loop** — Trains each fold with resumable checkpointing, tunes a patch-level detection threshold on each fold's validation split, and evaluates on the fixed test set. Each fold exports its best checkpoint.

**6. Evaluation & Calibration** — Computes pixel-level metrics (Dice, IoU) and patch-level metrics (precision, recall, F1, ROC-AUC, average precision). Patch scores are derived from the mean probability of the top 1% of predicted pixels, which avoids penalizing correct models for not predicting the exact substation boundary.

**7. Ensemble Construction** — Loads all 5 fold checkpoints, evaluates four aggregation methods (mean, median, trimmed mean, max), selects `max` aggregation as the final strategy (preserves strong evidence from any single fold), and exports the combined checkpoint.

**8. Pipeline Execution & Final Summary** — Runs exactly one branch (cached evaluation or full training), reports final metrics on validation and test splits, and generates visual prediction demonstrations showing the input image, probability heatmap, binary mask, and top-scoring pixels.

---

## Optimizer Details

### Objective function

The optimizer solves a capacitated p-median facility location problem:

```
minimize:  Σ demand(d) × distance(d,s) × assign[d,s]    (service distance)
         + build_cost × Σ build[c]                        (construction budget)
```

where `d` indexes demand cells, `s` indexes candidate sites (existing + new), and `assign[d,s]` is a binary variable assigning each demand cell to exactly one substation.

### Demand estimation

Each grid cell's demand (in kW) is computed as:

```
demand_kw = population × 2.0 + building_area_m² × 0.03
```

- **Population**: Census block group data, areally interpolated to the optimizer's grid cells
- **Building area**: Microsoft Building Footprints, spatially joined to block groups
- Falls back to population-only demand when footprints are unavailable

### Key optimizer flags

| Flag | What it does | When to use |
|------|-------------|-------------|
| `--no-capacity` | Removes substation capacity constraints | **Recommended for most runs.** Capacity values are rough estimates since voltage class cannot be determined from imagery alone. |
| `--grid-cell-size 5000` | Coarser demand grid (5 km cells instead of 3 km) | If the solver is slow or times out |
| `--max-new 30` | Cap new substations at 30 | Smaller budgets, tighter analysis |
| `--build-cost 2e6` | Set construction cost to $2M per substation | Sensitivity testing |
| `--sensitivity` | Sweep `max_new` from 10 to 100 | Shows diminishing returns curve |
| `--bbox W S E N` | Override preset bounding box for optimization | Focus on a specific corridor or neighborhood |

---

## CLI Reference

### extract_region.py

| Flag | Default | Description |
|------|---------|-------------|
| `--county` | required | Preset key (e.g., `maricopa`) |
| `--bbox W S E N` | from preset | Custom bounding box (WGS-84 decimal degrees) |
| `-o` | `demo_data` | Output directory |
| `--year-min` | `2021` | NAIP year range start |
| `--year-max` | `2024` | NAIP year range end |
| `--skip-existing` | off | Resume interrupted extraction |
| `--dry-run` | off | Print patch/tile count and estimated size without downloading |
| `-v` | off | Verbose logging |

### local_pipeline.py

| Flag | Default | Description |
|------|---------|-------------|
| `--county` | required | Preset key |
| `--eval-csv` | `eval_dataset/metadata.csv` | Path to metadata CSV |
| `--img-dir` | `eval_dataset/images` | Path to image patches directory |
| `--model-path` | `model/ensemble_model.pt` | Path to ensemble checkpoint |
| `--bbox W S E N` | from preset | Override optimizer bounding box |
| `--threshold` | `0.5` | Detection confidence threshold |
| `--batch-size` | `64` | Inference batch size (lower if GPU OOM) |
| `--max-new` | `50` | Maximum new substations to recommend |
| `--build-cost` | `1.5e6` | Cost per new substation ($) |
| `--grid-cell-size` | `3000` | Demand grid cell size in meters |
| `--no-capacity` | off | Skip capacity constraints |
| `--sensitivity` | off | Run budget sensitivity sweep |
| `--skip-detection` | off | Skip model inference, use existing metadata |
| `--skip-analysis` | off | Skip site suitability analysis |
| `--time-limit` | `300` | MILP solver time limit in seconds |
| `-o` | `output` | Output directory |
| `-v` | off | Verbose logging |

### fetch_data.py

| Flag | Default | Description |
|------|---------|-------------|
| `--county` | required | Preset key |
| `--skip-buildings` | off | Skip Microsoft Building Footprints download |
| `--skip-osm` | off | Skip OSM power line download |
| `-v` | off | Verbose logging |

### download_assets.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | off | Download ensemble model weights |
| `--demo` | off | Download prebuilt demo_data (alternative to extract_region.py) |

---

## Troubleshooting

### "Solver status: Not Solved" or all-zero results

The MILP could not find a feasible solution within the time limit. This usually means the demand grid is too fine or capacity constraints are too tight. Try:

```bash
--grid-cell-size 5000 --no-capacity --time-limit 600
```

### "HTTP response code: 403" during patch extraction

Planetary Computer signed URLs expire after approximately one hour. Re-run with `--skip-existing` to pick up where you left off with fresh tokens:

```bash
python scripts/extract_region.py --county maricopa -o demo_data --skip-existing -v
```

### "No positive detections for county_geoid ..."

The metadata CSV has no detections matching the county's GEOID. Check that `--eval-csv` points to the correct file and that the county GEOID in the preset matches the data you extracted.

### Extraction produces too many patches

Shrink the bounding box. Always use `--dry-run` first to check the estimate before downloading.

### Model loading fails with "Unexpected key(s): fold_01, fold_02..."

You are running an older script that tries to load the ensemble as a single model. Make sure `scripts/local_pipeline.py` is current — it includes the `load_ensemble()` function that loads each fold separately.

### Building footprints download is slow or fails

The state-level GeoJSON files are large (Arizona alone is ~800 MB). The script downloads the full state file, filters to your county's bounding box, saves the result as a compact Parquet file, then deletes the intermediates. This is a one-time cost per state. If the download fails partway through, delete any partial files in `data/buildings/` and re-run.

### Site analysis produces "Address not found" for some sites

Nominatim (OpenStreetMap's geocoder) has rate limits and incomplete coverage in some areas. The pipeline respects a 1-second delay between requests. Sites without addresses still get coordinates and suitability flags — only the street address field will be empty.

---

## Assumptions and Limitations

| Assumption | Reality | Impact |
|-----------|---------|--------|
| 2.0 kW per person | Actual per-capita load varies 1–4 kW by climate, income, and housing type | Demand estimates may be off by up to 2× |
| 0.03 kW/m² building area | Industrial buildings draw 0.05–0.1 kW/m², residential 0.01–0.02 | Commercial/industrial areas may be underweighted |
| 200 MW substation capacity | Real range: 20 MW (distribution) to 500 MW (transmission) | Use `--no-capacity` to avoid this assumption entirely |
| 500m restriction buffer around parks/wetlands | Some real projects are built within 200m | Results are conservative about restricted areas |
| OSM substation labels are accurate | Some labels are outdated or have imprecise boundaries | Training data noise; mitigated by ensemble averaging |
| NAIP vintage represents current conditions | NAIP imagery is typically 1–3 years old | Recently built substations may be missed |
| Building footprints proxy for load | Footprints capture area, not function or occupancy | A warehouse and a data center look the same |
| Optimizer has no ground truth | We cannot evaluate site recommendations against real planning decisions | Detection is validated (IoU/Dice); siting quality is not |

**This is a planning support tool.** Every recommendation means "investigate this site further" — not "build here."

---

## Authors

**Nishan Sah · Arthur Spirou · Jordan Gutterman · Arturo Arias**

Carnegie Mellon University, Heinz College of Information Systems and Public Policy
Introduction to Artificial Intelligence, Spring 2026
