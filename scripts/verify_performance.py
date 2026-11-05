"""Reproducible full-workload solver benchmark; no GUI/startup time is hidden."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rotor_math import RotorInputs, build_rotor, slot_geometry  # noqa: E402
from solver import SolverConfig, solve, warm_up  # noqa: E402
from data_io import sample_blades  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-blades", type=int, default=120)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/benchmark_matrix.json")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 2 <= args.max_blades <= 120:
        parser.error("repeats must be 1..10 and max-blades 2..120")

    scratch = ROOT / "artifacts/local"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cold-numba-", dir=scratch) as cache:
        env = dict(os.environ, NUMBA_CACHE_DIR=cache)
        started = time.perf_counter()
        cold = subprocess.run(
            [
                sys.executable,
                "-c",
                'import json; from solver import warm_up; print(json.dumps({"jit_ms": warm_up()}))',
            ],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        cold_process_ms = (time.perf_counter() - started) * 1000
        cold_jit_ms = json.loads(cold.stdout)["jit_ms"]
    warmup_ms = warm_up()
    rows = []
    seeds = (42, 7, 2026, 1001, 23, 123, 918, 305, 71, 113)
    readings = tuple(
        0.003 * math.cos(j * math.pi / 4)
        - 0.002 * math.sin(j * math.pi / 4)
        + 0.0005 * math.cos(j * math.pi / 2)
        for j in range(8)
    )
    for n in range(2, args.max_blades + 1):
        masses = tuple(b.weight for b in sample_blades(n, seed=n))
        for mode in ("blade_only", "combined"):
            rotor = build_rotor(RotorInputs(mode=mode, cg_radius_mm=300, runout_mm=readings))
            geom = slot_geometry(n, rotor)
            timings = []
            residuals = []
            for rep in range(args.repeats):
                cfg = SolverConfig(seed=seeds[rep])
                started = time.perf_counter()
                result = solve(masses, geom.x_mm, geom.y_mm, (rotor.disk_ux_gmm, rotor.disk_uy_gmm), cfg)
                elapsed_ms = (time.perf_counter() - started) * 1000
                # Independent final full-sum and model check, outside the timer.
                ux = math.fsum(
                    [rotor.disk_ux_gmm]
                    + [
                        masses[index]
                        * (
                            300
                            + rotor.ex_mm * math.cos(2 * math.pi * k / n)
                            + rotor.ey_mm * math.sin(2 * math.pi * k / n)
                        )
                        * math.cos(2 * math.pi * k / n)
                        for k, index in enumerate(result.permutation)
                    ]
                )
                uy = math.fsum(
                    [rotor.disk_uy_gmm]
                    + [
                        masses[index]
                        * (
                            300
                            + rotor.ex_mm * math.cos(2 * math.pi * k / n)
                            + rotor.ey_mm * math.sin(2 * math.pi * k / n)
                        )
                        * math.sin(2 * math.pi * k / n)
                        for k, index in enumerate(result.permutation)
                    ]
                )
                assert result.iterations_completed == 500_000, (n, mode, result.iterations_completed)
                assert math.isclose(math.hypot(ux, uy), result.residual_gmm, abs_tol=1e-6)
                assert all(math.isfinite(v) for v in (ux, uy, result.residual_gmm, result.lower_bound_gmm))
                assert (
                    result.residual_gmm
                    <= min(result.initial_j_gmm, result.greedy_j_gmm, result.pairing_j_gmm) + 1e-6
                )
                assert result.lower_bound_gmm <= result.residual_gmm + 1e-6
                timings.append(elapsed_ms)
                residuals.append(result.residual_gmm)
            rows.append(
                {
                    "n": n,
                    "mode": mode,
                    "wall_ms": timings,
                    "median_ms": statistics.median(timings),
                    "max_ms": max(timings),
                    "residuals_gmm": residuals,
                    "proposals_per_call": 500_000,
                }
            )
    elapsed_values = [v for row in rows for v in row["wall_ms"]]
    summary = {
        "cases": len(rows),
        "solver_calls": len(elapsed_values),
        "proposals_per_call": 500_000,
        "total_proposals": len(elapsed_values) * 500_000,
        "median_wall_ms": statistics.median(elapsed_values),
        "maximum_wall_ms": max(elapsed_values),
        "calls_below_500_ms": sum(v < 500 for v in elapsed_values),
        "cold_fresh_cache_jit_ms": cold_jit_ms,
        "cold_process_import_plus_jit_ms": cold_process_ms,
        "in_process_warmup_ms": warmup_ms,
        "all_math_and_workload_assertions_passed": True,
    }
    report = {
        "host": {
            "os": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "executable": Path(sys.executable).name,
            "environment": "project-local .venv",
            "packages": {
                name: importlib.metadata.version(name) for name in ("numpy", "numba", "matplotlib", "PySide6")
            },
        },
        "summary": summary,
        "matrix": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Report:", args.out)
    return 0 if all(v < 500 for v in elapsed_values) else 1


if __name__ == "__main__":
    raise SystemExit(main())
