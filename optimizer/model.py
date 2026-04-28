"""MILP formulation, solve, and substation capacity helpers."""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pulp

from .config import OptimizerConfig
from .containers import Candidates, DemandGrid, SolveResult, SparseDistances


def voltage_to_capacity_kw(voltage_v) -> Optional[float]:
    """Map nominal voltage (volts) to approximate substation capacity (kW).
    Based on typical transmission/distribution substation ratings.
    Returns None if voltage is missing or unparseable (caller uses default).
    """
    try:
        # Handle multi-voltage strings like "230000;69000" — take the highest
        v = max(float(x) for x in str(voltage_v).split(";") if x.strip())
    except (ValueError, TypeError):
        return None

    if v >= 500_000: return 500_000.0  # 500 kV  -> ~500 MW
    if v >= 345_000: return 300_000.0  # 345 kV  -> ~300 MW
    if v >= 230_000: return 200_000.0  # 230 kV  -> ~200 MW
    if v >= 115_000: return 100_000.0  # 115 kV  -> ~100 MW
    if v >= 69_000:  return 100_000.0  # 69 kV   -> ~100 MW (dense metro, multiple transformers)
    if v >= 46_000:  return  30_000.0  # 46 kV   ->  ~30 MW
    return 20_000.0                    # distribution -> ~20 MW


def build_substation_capacities(existing: gpd.GeoDataFrame,
                                candidates: Candidates,
                                config: OptimizerConfig) -> dict[tuple[str, int], float]:
    """Build a capacity (kW) lookup for every substation in the model.

    Existing substations use voltage-derived capacity from metadata;
    candidates use config.default_candidate_capacity_kw.
    """
    capacities: dict[tuple[str, int], float] = {}

    for idx, row in existing.iterrows():
        cap = voltage_to_capacity_kw(row.get("voltage"))
        capacities[("existing", int(idx))] = (
            cap if cap is not None else config.default_existing_capacity_kw
        )

    for cid in candidates.candidate_ids:
        capacities[("candidate", cid)] = config.default_candidate_capacity_kw

    return capacities


class SubstationSitingModel:
    """PuLP MILP for substation siting."""

    def __init__(self,
                 config: OptimizerConfig,
                 grid: DemandGrid,
                 candidates: Candidates,
                 distances: SparseDistances,
                 capacities: dict[tuple[str, int], float]):
        self.config = config
        self.grid = grid
        self.candidates = candidates
        self.distances = distances
        self.capacities = capacities

        self.prob: Optional[pulp.LpProblem] = None
        self.build_vars: dict[int, pulp.LpVariable] = {}
        self.assign_vars: dict[tuple[int, tuple[str, int]], pulp.LpVariable] = {}

    def build(self) -> None:
        """Construct the MILP."""
        self.prob = pulp.LpProblem("substation_siting", pulp.LpMinimize)
        self._create_variables()
        self._set_objective()
        self._add_coverage_constraints()
        self._add_coverage_distance_constraints()
        self._add_linking_constraints()
        if not self.config.skip_capacity:
            self._add_capacity_constraints()
        else:
            print("  Capacity constraints: SKIPPED (--no-capacity)")
        self._add_substation_cap()
        print(f"  MILP: {len(self.build_vars)} binary + "
              f"{len(self.assign_vars)} continuous variables")

    def _create_variables(self) -> None:
        for c in self.candidates.candidate_ids:
            self.build_vars[c] = pulp.LpVariable(f"build_{c}", cat="Binary")

        for (d, s) in self.distances.pairs:
            self.assign_vars[(d, s)] = pulp.LpVariable(
                f"assign_{d}_{s[0]}_{s[1]}", lowBound=0, upBound=1
            )

    def _set_objective(self) -> None:
        # Scale factor: keeps coefficients in a solver-friendly range.
        # With building footprints, demand can be ~50,000 kW per cell and
        # distances ~20,000 m, producing coefficients of ~10^9 per term.
        # Dividing by 1000 (kW→MW) keeps coefficients under ~10^6.
        # Both terms are scaled equally so the optimal solution is unchanged.
        SCALE = 1000.0

        # Assignment cost: (demand / SCALE) × distance × assignment fraction
        assignment_cost = pulp.lpSum(
            (self.grid.demand(d) / SCALE) * dist * self.assign_vars[(d, s)]
            for (d, s), dist in self.distances.pairs.items()
        )
        # Fixed build cost (scaled by same factor)
        build_cost = pulp.lpSum(
            (self.config.fixed_build_cost / SCALE) * self.build_vars[c]
            for c in self.candidates.candidate_ids
        )
        self.prob += assignment_cost + build_cost

    def _add_coverage_constraints(self) -> None:
        """Every demand cell must be fully served."""
        for d in self.grid.cell_ids:
            servers = self.distances.neighbors_of(d)
            self.prob += (
                pulp.lpSum(self.assign_vars[(d, s)] for s in servers) == 1,
                f"coverage_{d}",
            )

    def _add_linking_constraints(self) -> None:
        """A candidate can serve load only if built."""
        for (d, s), var in self.assign_vars.items():
            if s[0] == "candidate":
                self.prob += (var <= self.build_vars[s[1]], f"link_{d}_{s[1]}")

    def _add_capacity_constraints(self) -> None:
        """Total load assigned to each substation must not exceed its capacity."""
        # Group assignment vars by substation
        by_sub: dict[tuple[str, int], list[tuple[int, pulp.LpVariable]]] = {}
        for (d, s), var in self.assign_vars.items():
            by_sub.setdefault(s, []).append((d, var))

        for s, pairs in by_sub.items():
            cap = self.capacities.get(s)
            if cap is None:
                continue
            self.prob += (
                pulp.lpSum(self.grid.demand(d) * var for d, var in pairs) <= cap,
                f"capacity_{s[0]}_{s[1]}",
            )

    def _add_coverage_distance_constraints(self) -> None:
        """Hard coverage constraint: demand cells that have at least one substation
        within max_coverage_dist_m must be served entirely within that distance.

        Cells with no option within the threshold are skipped (keeps problem feasible
        in sparse areas; those cells fall back to the wider max_service_radius_m).
        """
        threshold = self.config.max_coverage_dist_m
        n_constrained = 0
        n_skipped = 0

        for d in self.grid.cell_ids:
            # Partition this cell's neighbors into close vs. far
            close = [s for s in self.distances.neighbors_of(d)
                     if self.distances.pairs[(d, s)] <= threshold]
            if not close:
                n_skipped += 1
                continue  # no option within threshold — skip gracefully

            # Require full coverage from close neighbors
            self.prob += (
                pulp.lpSum(self.assign_vars[(d, s)] for s in close) == 1.0,
                f"coverage_dist_{d}",
            )
            n_constrained += 1

        print(f"  Coverage distance constraints: {n_constrained} cells enforced "
              f"(<= {threshold/1000:.0f}km), {n_skipped} cells skipped (no nearby option)")

    def _add_substation_cap(self) -> None:
        """Optional upper bound on number of new substations."""
        if self.config.max_new_substations is None:
            return
        self.prob += (
            pulp.lpSum(self.build_vars.values()) <= self.config.max_new_substations,
            "substation_cap",
        )

    def solve(self) -> SolveResult:
        if self.prob is None:
            raise RuntimeError("Call build() before solve()")

        solver = pulp.getSolver(
            "HiGHS",
            msg=True,
            timeLimit=self.config.solver_time_limit_s,
            gapRel=self.config.solver_gap_rel,
        )
        self.prob.solve(solver)

        status = pulp.LpStatus[self.prob.status]
        obj = pulp.value(self.prob.objective)

        # Check for solver failure
        if status not in ("Optimal", "Feasible"):
            print(f"\n  [!] Solver status: {status}")
            if status == "Infeasible":
                print("      The problem is infeasible. This usually means capacity")
                print("      constraints conflict with coverage requirements.")
                print("      Try: --grid-cell-size 5000 or increase substation capacity.")
            elif status == "Not Solved":
                print("      Solver timed out without finding a feasible solution.")
                print("      Try: --time-limit 900 --grid-cell-size 5000")
            # Return empty result so pipeline can still output partial info
            return SolveResult(
                status=status, objective=0.0, mip_gap=0.0,
                build={c: 0 for c in self.build_vars},
                assign={k: 0.0 for k, v in self.assign_vars.items()},
            )

        print(f"  Solver status: {status}, objective: {obj:,.0f}")

        # Safe extraction: handle None values from PuLP
        def _val(v):
            x = v.value()
            return x if x is not None else 0.0

        return SolveResult(
            status=status,
            objective=obj if obj is not None else 0.0,
            mip_gap=0.0,
            build={c: int(round(_val(v))) for c, v in self.build_vars.items()},
            assign={k: _val(v) for k, v in self.assign_vars.items()},
        )
