"""Top-level optimizer pipeline."""

from __future__ import annotations

from .candidates import generate_candidates
from .config import OptimizerConfig
from .data_loader import load_data_centers, load_input_data
from .demand_grid import build_demand_grid
from .distances import build_sparse_distances
from .model import SubstationSitingModel, build_substation_capacities
from .results import ResultsSummary, extract_results, save_csv_outputs
from .visualize import generate_visualizations


def run_optimizer(config: OptimizerConfig,
                  save_outputs: bool = False,
                  visualize: bool = True) -> ResultsSummary:
    """Execute the full optimization pipeline."""
    print(f"Optimizing substation placement for {config.county_name}")
    print(f"  CRS: {config.crs_proj}, grid cell: {config.grid_cell_size_m}m")

    print("\n[1/5] Loading data...")
    data = load_input_data(config)
    data_centers = load_data_centers(config)

    print("\n[2/5] Building demand grid...")
    grid = build_demand_grid(config, data)

    print("\n[3/5] Generating candidate sites...")
    candidates = generate_candidates(config, grid, data, data_centers)

    print("\n[4/5] Building sparse distance matrix...")
    distances = build_sparse_distances(
        grid, candidates, data.existing_substations, config.max_service_radius_m
    )

    print("\n[5/5] Solving MILP...")
    capacities = build_substation_capacities(data.existing_substations, candidates, config)
    model = SubstationSitingModel(config, grid, candidates, distances, capacities)
    model.build()
    result = model.solve()

    print("\nExtracting results...")
    summary = extract_results(result, config, grid, candidates,
                              data.existing_substations, distances,
                              transmission_lines=data.transmission_lines)
    summary.print_summary()

    print("\nSaving CSV outputs...")
    save_csv_outputs(summary, config)

    print("\nGenerating visualizations...")
    generate_visualizations(summary, grid, data, candidates, result, distances, config)

    return summary
