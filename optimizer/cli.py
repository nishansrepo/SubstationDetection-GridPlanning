"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import COUNTY_PRESETS, DATA_DIR, OptimizerConfig
from .pipeline import run_optimizer
from .sensitivity import run_sensitivity_analysis


def parse_args() -> tuple[OptimizerConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Grid substation siting optimizer")
    p.add_argument("--county", default="maricopa",
                   choices=list(COUNTY_PRESETS.keys()),
                   help="County to optimize (default: maricopa)")
    p.add_argument("--metadata", type=Path, default=DATA_DIR / "metadata.csv",
                   help="Path to model detection metadata CSV")
    p.add_argument("--grid-cell-size", type=float, default=3000.0,
                   help="Demand grid cell size in meters (default: 3000)")
    p.add_argument("--build-cost", type=float, default=1.5e6,
                   help="Fixed cost per new substation (default: 1.5e6)")
    p.add_argument("--max-new", type=int, default=50,
                   help="Max number of new substations (default: 50)")
    p.add_argument("--max-radius", type=float, default=20000.0,
                   help="Max service radius in meters (default: 20000)")
    p.add_argument("--max-coverage-dist", type=float, default=15000.0,
                   help="Hard coverage distance constraint in meters (default: 15000)")
    p.add_argument("--time-limit", type=int, default=300,
                   help="Solver time limit in seconds (default: 300)")
    p.add_argument("--save-outputs", action="store_true",
                   help="Save per-substation and TX expansion tables as CSVs to output/")
    p.add_argument("--visualize", action="store_true",
                   help="Generate and save results map and coverage heatmap to output/")
    p.add_argument("--sensitivity", action="store_true",
                   help="Run sensitivity analysis over max_new values [10,25,50,75,100]")
    args = p.parse_args()

    config = OptimizerConfig(
        county=args.county,
        metadata_path=args.metadata,
        grid_cell_size_m=args.grid_cell_size,
        fixed_build_cost=args.build_cost,
        max_new_substations=args.max_new,
        max_service_radius_m=args.max_radius,
        max_coverage_dist_m=args.max_coverage_dist,
        solver_time_limit_s=args.time_limit,
    )
    return config, args


def main() -> None:
    config, args = parse_args()
    run_optimizer(config, save_outputs=args.save_outputs, visualize=args.visualize)
    if args.sensitivity:
        run_sensitivity_analysis(config)


if __name__ == "__main__":
    main()
