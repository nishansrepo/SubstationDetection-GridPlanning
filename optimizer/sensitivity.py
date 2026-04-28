"""Sensitivity analysis over max_new_substations."""

from __future__ import annotations

import dataclasses

import pandas as pd

from .candidates import generate_candidates
from .config import BASE_DIR, OptimizerConfig
from .data_loader import load_data_centers, load_input_data
from .demand_grid import build_demand_grid
from .distances import build_sparse_distances
from .model import SubstationSitingModel, build_substation_capacities
from .results import extract_results


def run_sensitivity_analysis(base_config: OptimizerConfig,
                              max_new_values: list[int] = None) -> pd.DataFrame:
    """Run the optimizer for a range of max_new_substations values and report
    how key metrics change. Returns a summary DataFrame and saves a plot."""
    import matplotlib.pyplot as plt

    if max_new_values is None:
        max_new_values = [10, 25, 50, 75, 100]

    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)

    print(f"\nSensitivity analysis: max_new in {max_new_values}")
    print("  (re-uses loaded data; only MILP re-solved for each value)")

    data = load_input_data(base_config)
    data_centers = load_data_centers(base_config)
    grid = build_demand_grid(base_config, data)
    candidates = generate_candidates(base_config, grid, data, data_centers)
    distances = build_sparse_distances(
        grid, candidates, data.existing_substations, base_config.max_service_radius_m
    )
    capacities = build_substation_capacities(
        data.existing_substations, candidates, base_config
    )

    rows = []
    for n in max_new_values:
        cfg = dataclasses.replace(base_config, max_new_substations=n)

        print(f"  Solving max_new={n}...", end=" ", flush=True)
        model = SubstationSitingModel(cfg, grid, candidates, distances, capacities)
        model.build()
        result = model.solve()

        if result.status != "Optimal":
            print(f"INFEASIBLE — coverage constraint requires more than {n} substations")
            rows.append({
                "max_new": n, "n_built": "INFEASIBLE",
                "coverage_gap_pct": None, "avg_service_dist_m": None,
                "max_service_dist_m": None, "load_shifted_kw": None,
                "new_area_km2": None,
            })
            continue

        summary = extract_results(result, cfg, grid, candidates,
                                  data.existing_substations, distances,
                                  transmission_lines=data.transmission_lines)
        print(f"gap={summary.coverage_gap_after_pct:.1f}%, "
              f"avg_dist={summary.avg_service_dist_after_m:.0f}m, "
              f"built={summary.n_new_substations}")
        rows.append({
            "max_new": n,
            "n_built": summary.n_new_substations,
            "coverage_gap_pct": round(summary.coverage_gap_after_pct, 2),
            "avg_service_dist_m": round(summary.avg_service_dist_after_m, 0),
            "max_service_dist_m": round(summary.max_service_dist_after_m, 0),
            "load_shifted_kw": round(summary.total_new_load_served_kw, 0),
            "new_area_km2": round(summary.new_coverage_area_km2, 1),
        })

    df = pd.DataFrame(rows)

    print(f"\n  Sensitivity results ({base_config.county_name}):")
    print(df.to_string(index=False))

    csv_path = out_dir / f"{base_config.county}_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.relative_to(BASE_DIR)}")

    df_plot = df[df["n_built"] != "INFEASIBLE"].copy()
    df_plot["coverage_gap_pct"] = df_plot["coverage_gap_pct"].astype(float)
    df_plot["avg_service_dist_m"] = df_plot["avg_service_dist_m"].astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(df_plot["max_new"], df_plot["coverage_gap_pct"], "o-", color="#2563eb", linewidth=2)
    ax1.axhline(y=df_plot["coverage_gap_pct"].iloc[-1], color="#94a3b8",
                linestyle="--", linewidth=1, label="Floor (TX-constrained)")
    ax1.set_xlabel("Max new substations")
    ax1.set_ylabel("Coverage gap — reachable cells > 10 km (%)")
    ax1.set_title("Coverage Gap vs. Build Budget")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(df_plot["max_new"], df_plot["avg_service_dist_m"], "o-", color="#16a34a", linewidth=2)
    ax2.set_xlabel("Max new substations")
    ax2.set_ylabel("Avg service distance (m)")
    ax2.set_title("Avg Service Distance vs. Build Budget")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{base_config.county_name} — Sensitivity: Build Budget",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    plot_path = out_dir / f"{base_config.county}_sensitivity.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path.relative_to(BASE_DIR)}")

    return df
