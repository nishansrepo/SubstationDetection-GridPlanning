"""Result visualizations: overview map and service-distance heatmap."""

from __future__ import annotations

import geopandas as gpd
import numpy as np

from .config import BASE_DIR, OptimizerConfig
from .containers import Candidates, DemandGrid, InputData, SolveResult, SparseDistances
from .results import ResultsSummary


def generate_visualizations(
        summary: ResultsSummary,
        grid: DemandGrid,
        data: InputData,
        candidates: Candidates,
        result: SolveResult,
        distances: SparseDistances,
        config: OptimizerConfig) -> None:
    """Generate and save two maps: results overview and service distance heatmap."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import BoundaryNorm

    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)

    built_ids = {c for c, v in result.build.items() if v == 1}

    constrained_ids = {
        d for d in grid.cell_ids
        if not any(
            distances.pairs[(d, s)] <= config.max_coverage_dist_m
            for s in distances.neighbors_of(d)
        )
    }

    service_dist = {}
    for d in grid.cell_ids:
        service_dist[d] = sum(
            distances.pairs[(d, s)] * result.assign.get((d, s), 0.0)
            for s in distances.neighbors_of(d)
        )

    county_outline = data.census_blockgroups.dissolve()

    # Map 1: Results overview
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    county_outline.boundary.plot(ax=ax, color="#333333", linewidth=1.0, zorder=1)

    demand_cells = grid.cells.copy()
    normal_cells = demand_cells[~demand_cells.index.isin(constrained_ids)]
    constrained_cells = demand_cells[demand_cells.index.isin(constrained_ids)]

    if len(normal_cells):
        normal_cells.plot(
            ax=ax, column="demand_kw", cmap="Blues",
            linewidth=0, alpha=0.6, zorder=2,
            legend=True,
            legend_kwds={"label": "Demand (kW)", "shrink": 0.5, "pad": 0.01},
        )
    if len(constrained_cells):
        constrained_cells.plot(
            ax=ax, color="#dddddd", linewidth=0, alpha=0.7, zorder=2
        )

    data.existing_substations.plot(
        ax=ax, color="#555555", markersize=8, zorder=3,
        marker="o", alpha=0.5, label=f"Existing substations ({len(data.existing_substations)})"
    )

    built_gdf = None
    if len(summary.per_substation):
        built_gdf = gpd.GeoDataFrame(
            summary.per_substation,
            geometry=gpd.points_from_xy(
                summary.per_substation["lon"],
                summary.per_substation["lat"],
            ),
            crs="EPSG:4326",
        ).to_crs(config.crs_proj)

        sizes = (built_gdf["total_load_kw"] / built_gdf["total_load_kw"].max() * 200 + 40)
        built_gdf.plot(
            ax=ax, color="#e63946", markersize=sizes, zorder=5,
            marker="*", label=f"New substations ({len(built_gdf)})"
        )

    tx = summary.transmission_expansion_candidates
    if len(tx):
        tx_gdf = gpd.GeoDataFrame(
            tx, geometry=gpd.points_from_xy(tx["lon"], tx["lat"]), crs="EPSG:4326"
        ).to_crs(config.crs_proj)
        tx_gdf.plot(
            ax=ax, color="#f4a261", markersize=120, zorder=4,
            marker="D", label=f"TX expansion clusters ({len(tx_gdf)})"
        )

    patches = [
        mpatches.Patch(color="#dddddd", label="Transmission-constrained cells"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + patches,
        labels=labels + [p.get_label() for p in patches],
        loc="lower left", fontsize=8, framealpha=0.9,
    )

    ax.set_title(
        f"{config.county_name} — Substation Siting Results\n"
        f"{len(built_ids)} new substations, {config.grid_cell_size_m/1000:.0f} km grid, "
        f"{config.max_coverage_dist_m/1000:.0f} km coverage constraint",
        fontsize=12,
    )
    ax.set_axis_off()
    fig.tight_layout()

    out_path = out_dir / f"{config.county}_results_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(BASE_DIR)}")

    # Map 2: Service distance heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    county_outline.boundary.plot(ax=ax, color="#333333", linewidth=1.0, zorder=1)

    dist_col = []
    for idx in demand_cells.index:
        if idx in constrained_ids:
            dist_col.append(np.nan)
        else:
            dist_col.append(service_dist.get(idx, np.nan))
    demand_cells = demand_cells.copy()
    demand_cells["service_dist_km"] = np.array(dist_col) / 1000.0

    reachable = demand_cells[~demand_cells["service_dist_km"].isna()]
    bounds = [0, 5, 10, 15, 20, 999]
    colors = ["#2dc653", "#94d82d", "#fcc419", "#ff6b35", "#c92a2a"]
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    reachable.plot(
        ax=ax, column="service_dist_km", cmap=cmap, norm=norm,
        linewidth=0, alpha=0.8, zorder=2,
    )

    if len(constrained_cells):
        constrained_cells.plot(ax=ax, color="#cccccc", linewidth=0, alpha=0.7, zorder=2)

    data.existing_substations.plot(
        ax=ax, color="#333333", markersize=5, zorder=3, marker="o", alpha=0.4
    )
    if built_gdf is not None:
        built_gdf.plot(ax=ax, color="#e63946", markersize=60, zorder=5, marker="*")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_ticks([2.5, 7.5, 12.5, 17.5])
    cbar.set_ticklabels(["0–5 km", "5–10 km", "10–15 km", "15–20 km"])
    cbar.set_label("Service distance", fontsize=9)

    legend_patches = [
        mpatches.Patch(color="#cccccc", label="Transmission-constrained"),
        mpatches.Patch(color="white", label=""),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333",
                   markersize=5, label="Existing substation"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e63946",
                   markersize=10, label="New substation"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.9)

    ax.set_title(
        f"{config.county_name} — Service Distance After Optimization\n"
        f"Coverage gap (>10 km, reachable): {summary.coverage_gap_after_pct:.1f}%",
        fontsize=12,
    )
    ax.set_axis_off()
    fig.tight_layout()

    out_path = out_dir / f"{config.county}_coverage_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(BASE_DIR)}")
