"""Independent acceptance checks added after reviewing the first implementation."""

import math
import json
from pathlib import Path

import numpy as np
import pytest

from rotor_math import RotorInputs, build_rotor, slot_geometry, first_harmonic
from solver import SolverConfig, solve, swap_delta


def test_reported_initial_net_is_the_input_order_not_final_result():
    rotor = build_rotor(RotorInputs(cg_radius_mm=250))
    geo = slot_geometry(7, rotor)
    masses = [99, 102, 100, 98, 101, 99.7, 100.4]
    disk = (17.0, -8.0)
    result = solve(masses, geo.x_mm, geo.y_mm, disk, SolverConfig(iterations=2000, restarts=2))
    direct = (
        disk[0] + math.fsum(m * x for m, x in zip(masses, geo.x_mm)),
        disk[1] + math.fsum(m * y for m, y in zip(masses, geo.y_mm)),
    )
    assert (result.initial_ux_gmm, result.initial_uy_gmm) == pytest.approx(direct, abs=1e-8)
    assert math.hypot(*direct) == pytest.approx(result.initial_j_gmm)


def test_convergence_is_global_best_with_initial_and_final_points():
    geo = slot_geometry(9, build_rotor(RotorInputs(cg_radius_mm=250)))
    masses = [100, 102, 99, 98, 101, 100.3, 99.4, 102.3, 99.2]
    cfg = SolverConfig(iterations=4101, restarts=3, trace_every=137)
    result = solve(masses, geo.x_mm, geo.y_mm, config=cfg)
    trace = result.trace
    assert trace[0][0] == 0
    assert trace[0][1] == pytest.approx(min(result.initial_j_gmm, result.greedy_j_gmm, result.pairing_j_gmm))
    assert trace[-1][0] == cfg.iterations * cfg.restarts
    assert trace[-1][1] == pytest.approx(result.residual_gmm, abs=1e-7)
    assert all(b[0] > a[0] for a, b in zip(trace, trace[1:]))
    assert all(b[1] <= a[1] + 1e-7 for a, b in zip(trace, trace[1:]))
    assert any(i == 137 for i, _ in trace)


@pytest.mark.parametrize(
    "config",
    [
        SolverConfig(seed=True),
        SolverConfig(seed=-1),
        SolverConfig(seed=2.3),
        SolverConfig(initial_temperature=-1),
        SolverConfig(initial_temperature=float("nan")),
        SolverConfig(initial_temperature=0),
        SolverConfig(alpha=True),
        SolverConfig(trace_every=2.5),
        SolverConfig(trace_every=True),
    ],
)
def test_invalid_config_is_rejected_before_search(config):
    with pytest.raises(ValueError):
        solve([100, 101], [1, -1], [0, 0], config=config)


@pytest.mark.parametrize("readings", [[True] * 8, ["0"] * 8, [float("nan")] * 8, [0] * 7])
def test_harmonic_rejects_nonphysical_reading_types(readings):
    with pytest.raises(ValueError):
        first_harmonic(readings)


def test_harmonic_handles_large_finite_dc_without_overflow():
    fit = first_harmonic([1e300] * 8)
    assert fit.ex_mm == 0 and fit.ey_mm == 0 and fit.residual_rms_mm == 0


@pytest.mark.parametrize(
    "masses,x,y",
    [
        ([1e308, 1e308], [1e308, -1e308], [0, 0]),
        ([True, 2], [1, -1], [0, 0]),
    ],
)
def test_solver_rejects_overflow_or_boolean_inputs(masses, x, y):
    with pytest.raises(ValueError):
        solve(masses, x, y, config=SolverConfig(iterations=1, restarts=1))


def test_independent_reachable_even_odd_fixtures():
    cases = json.loads((Path(__file__).parent / "artifacts/independent_reference_cases.json").read_text())[
        "cases"
    ]
    for c in cases:
        rotor = build_rotor(
            RotorInputs(
                mode="combined",
                diameter_mm=c["diameter_mm"],
                thickness_mm=c["thickness_mm"],
                density_kg_m3=c["density_kg_m3"],
                cg_radius_mm=c["cg_radius_mm"],
                runout_mm=tuple(c["runout_mm"]),
            )
        )
        geo = slot_geometry(c["n"], rotor)
        result = solve(c["masses_g"], geo.x_mm, geo.y_mm, (rotor.disk_ux_gmm, rotor.disk_uy_gmm))
        assert result.iterations_completed == 500000
        assert result.residual_gmm < 1e-6, (c["n"], result.residual_gmm)
        assert rotor.ex_mm == pytest.approx(c["ex_mm"], abs=1e-14)
        assert rotor.ey_mm == pytest.approx(c["ey_mm"], abs=1e-14)


@pytest.mark.parametrize("n", [2, 3, 31, 32, 119, 120])
@pytest.mark.parametrize("combined", [False, True])
def test_independent_vector_truth_and_lower_bound(n, combined):
    readings = tuple(0.003 * math.cos(j * math.pi / 4) - 0.002 * math.sin(j * math.pi / 4) for j in range(8))
    rotor = build_rotor(
        RotorInputs(mode="combined" if combined else "blade_only", cg_radius_mm=300, runout_mm=readings)
    )
    geo = slot_geometry(n, rotor)
    masses = np.random.default_rng(n).uniform(95, 105, n)
    result = solve(masses, geo.x_mm, geo.y_mm, (rotor.disk_ux_gmm, rotor.disk_uy_gmm))
    x = math.fsum(
        [rotor.disk_ux_gmm]
        + [float(masses[idx]) * float(geo.x_mm[k]) for k, idx in enumerate(result.permutation)]
    )
    y = math.fsum(
        [rotor.disk_uy_gmm]
        + [float(masses[idx]) * float(geo.y_mm[k]) for k, idx in enumerate(result.permutation)]
    )
    assert result.residual_gmm == pytest.approx(math.hypot(x, y), abs=1e-7)
    assert result.lower_bound_gmm <= result.residual_gmm + 1e-7
    assert result.residual_gmm <= min(result.initial_j_gmm, result.greedy_j_gmm, result.pairing_j_gmm) + 1e-7


def test_random_swap_deltas_match_independent_full_sums():
    rng = np.random.default_rng(123)
    n = 17
    m = rng.uniform(50, 200, n)
    x = rng.uniform(-300, 300, n)
    y = rng.uniform(-300, 300, n)
    p = rng.permutation(n).tolist()
    for _ in range(100):
        a, b = map(int, rng.choice(n, 2, replace=False))
        ux = math.fsum(m[p[k]] * x[k] for k in range(n))
        uy = math.fsum(m[p[k]] * y[k] for k in range(n))
        dx, dy = swap_delta(m, p, x, y, a, b)
        p[a], p[b] = p[b], p[a]
        assert ux + dx == pytest.approx(math.fsum(m[p[k]] * x[k] for k in range(n)), abs=1e-8)
        assert uy + dy == pytest.approx(math.fsum(m[p[k]] * y[k] for k in range(n)), abs=1e-8)
