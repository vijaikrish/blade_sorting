import itertools
import math
import numpy as np
import pytest
from solver import SolverConfig, SolverCancelled, vector_sum, swap_delta, solve


def test_sum_and_swap_delta_match_independent_direct_sum():
    m = np.array([1.0, 3.0, 2.0, 5.0])
    x = np.array([1.0, 0.0, -1.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, -1.0])
    p = (0, 1, 2, 3)
    before = vector_sum(m, p, x, y, (0.2, -0.1))
    dx, dy = swap_delta(m, p, x, y, 0, 2)
    q = (2, 1, 0, 3)
    after = vector_sum(m, q, x, y, (0.2, -0.1))
    assert (before[0] + dx, before[1] + dy) == pytest.approx(after)


def test_greedy_starts_and_hybrid_never_worse_than_baselines():
    m = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.cos(np.arange(5) * 2 * math.pi / 5)
    y = np.sin(np.arange(5) * 2 * math.pi / 5)
    result = solve(m, x, y, config=SolverConfig(iterations=40, restarts=2, seed=18))
    assert result.residual_gmm <= min(result.initial_j_gmm, result.greedy_j_gmm, result.pairing_j_gmm) + 1e-10
    assert result.iterations_completed == 80
    assert (
        solve(m, x, y, config=SolverConfig(iterations=40, restarts=2, seed=18)).permutation
        == result.permutation
    )


def test_cancel_and_invalid_inputs():
    with pytest.raises(ValueError):
        vector_sum([1.0, -2.0], [0, 1], [1, 0], [0, 1])
    with pytest.raises(ValueError):
        vector_sum([1.0, 2.0], [0, 0], [1, 0], [0, 1])
    with pytest.raises(SolverCancelled):
        solve(
            [1.0, 2.0],
            [1.0, 0.0],
            [0.0, 1.0],
            config=SolverConfig(iterations=40, restarts=1),
            cancel=lambda: True,
        )


def test_exhaustive_small_n_matches_exact_oracle():
    m = np.array([1.0, 2.0, 4.0, 7.0])
    x = np.cos(np.arange(4) * math.pi / 2)
    y = np.sin(np.arange(4) * math.pi / 2)
    truth = min(math.hypot(*vector_sum(m, p, x, y)) for p in itertools.permutations(range(4)))
    result = solve(m, x, y, config=SolverConfig(iterations=100, restarts=2, seed=3))
    assert result.residual_gmm >= truth - 1e-10
    assert result.residual_gmm <= truth + 1e-10


def test_convergence_history_is_not_backfilled_with_the_future_final_result():
    m = np.array([99.1, 101.2, 100.4, 98.8, 102.1, 99.6, 100.7, 98.3, 101.9])
    angles = np.arange(len(m)) * 2 * math.pi / len(m)
    x, y = 300 * np.cos(angles), 300 * np.sin(angles)
    short = solve(m, x, y, config=SolverConfig(iterations=2048, restarts=1, seed=123, trace_every=128))
    long = solve(m, x, y, config=SolverConfig(iterations=4096, restarts=1, seed=123, trace_every=128))
    prefix = {i: j for i, j in long.trace if i <= 2048}
    for i, j in short.trace:
        assert prefix[i] == pytest.approx(j, abs=1e-7)


def test_annealing_kernel_preserves_current_state_separately_from_best():
    import solver

    assert hasattr(solver, "_anneal_block"), (
        "a resumable kernel must expose both current and incumbent states"
    )
    m = np.array([1.0, 2.0])
    x = np.array([1.0, -1.0])
    y = np.array([0.0, 0.0])
    current = np.array([0, 1], dtype=np.int64)
    best = current.copy()
    # The equal-J swap is accepted but is not a strictly improved incumbent.
    out = solver._anneal_block(
        m,
        x,
        y,
        current,
        -1.0,
        0.0,
        1.0,
        best,
        1.0,
        1.0,
        0.9999,
        np.array([0]),
        np.array([1]),
        np.array([0.5]),
        0,
        1,
    )
    assert tuple(current) == (1, 0)
    assert tuple(best) == (0, 1)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)


def test_trace_uses_requested_interval_and_initial_net_is_input_order():
    masses = [2.0, 3.0, 5.0]
    x, y = [1.0, -0.5, -0.5], [0.0, 0.8, -0.8]
    cfg = SolverConfig(iterations=19, restarts=2, trace_every=7, seed=4)
    result = solve(masses, x, y, config=cfg)
    assert result.initial_ux_gmm == pytest.approx(sum(m * a for m, a in zip(masses, x)))
    assert result.trace[0][0] == 0
    assert result.trace[-1][0] == 38
    assert [i for i, _ in result.trace] == sorted(set(i for i, _ in result.trace))
    # Exact full-sum drift correction can raise a sampled value by <1e-12 g.mm.
    assert all(result.trace[i][1] <= result.trace[i - 1][1] + 1e-10 for i in range(1, len(result.trace)))


def test_solver_rejects_boolean_and_nonfinite_physical_inputs():
    with pytest.raises(ValueError):
        solve([True, 2], [1, -1], [0, 0], config=SolverConfig(iterations=1, restarts=1))
    with pytest.raises(ValueError):
        solve([1e308, 1], [1e308, -1e308], [0, 0], config=SolverConfig(iterations=1, restarts=1))
