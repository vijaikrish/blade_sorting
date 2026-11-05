"""Deterministic swap simulated annealing for static blade placement."""

from dataclasses import dataclass, replace
import math
import time
import numpy as np
from numba import njit


@dataclass(frozen=True)
class SolverConfig:
    seed: int = 42
    iterations: int = 100_000
    restarts: int = 5
    alpha: float = 0.9999
    initial_temperature: float | None = None
    trace_every: int = 250


@dataclass(frozen=True)
class SolverResult:
    permutation: tuple[int, ...]
    initial_ux_gmm: float
    initial_uy_gmm: float
    initial_blade_ux_gmm: float
    initial_blade_uy_gmm: float
    greedy_j_gmm: float
    pairing_j_gmm: float
    initial_j_gmm: float
    residual_gmm: float
    blade_ux_gmm: float
    blade_uy_gmm: float
    net_ux_gmm: float
    net_uy_gmm: float
    lower_bound_gmm: float
    elapsed_ms: float
    iterations_completed: int
    seed: int
    method: str
    trace: tuple[tuple[int, float], ...]
    warnings: tuple[str, ...]


class SolverCancelled(Exception):
    pass


def _inputs(masses, permutation, x_mm, y_mm, disk):
    raw_m, raw_x, raw_y = tuple(masses), tuple(x_mm), tuple(y_mm)
    if any(
        isinstance(v, (bool, str, bytes)) or not isinstance(v, (int, float, np.number))
        for v in (*raw_m, *raw_x, *raw_y)
    ):
        raise ValueError("masses and coordinates must be numeric, not bool or string")
    m = np.array(raw_m, dtype=np.float64, copy=True)
    x = np.array(raw_x, dtype=np.float64, copy=True)
    y = np.array(raw_y, dtype=np.float64, copy=True)
    if m.ndim != 1 or len(m) < 2 or len(m) > 120 or x.shape != m.shape or y.shape != m.shape:
        raise ValueError("masses, x_mm and y_mm must be equal one-dimensional arrays of length 2..120")
    if (
        not np.all(np.isfinite(m))
        or np.any(m <= 0)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("masses must be positive and all inputs finite")
    if max(float(np.max(m)), float(np.max(np.abs(x))), float(np.max(np.abs(y)))) > 1e150:
        raise ValueError("input dynamic range exceeds supported finite calculations")
    p = tuple(range(len(m))) if permutation is None else tuple(permutation)
    if (
        len(p) != len(m)
        or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in p)
        or set(p) != set(range(len(m)))
    ):
        raise ValueError("permutation must contain each blade index exactly once")
    if (
        len(disk) != 2
        or any(isinstance(v, (bool, str, bytes)) for v in disk)
        or not all(math.isfinite(float(v)) for v in disk)
    ):
        raise ValueError("disk must contain two finite values")
    return m, np.asarray(p, dtype=np.int64), x, y, (float(disk[0]), float(disk[1]))


def vector_sum(masses, permutation, x_mm, y_mm, disk=(0.0, 0.0)):
    m, p, x, y, d = _inputs(masses, permutation, x_mm, y_mm, disk)
    return math.fsum([d[0], *(float(m[p[i]] * x[i]) for i in range(len(m)))]), math.fsum(
        [d[1], *(float(m[p[i]] * y[i]) for i in range(len(m)))]
    )


def swap_delta(masses, permutation, x_mm, y_mm, a, b):
    m, p, x, y, _ = _inputs(masses, permutation, x_mm, y_mm, (0.0, 0.0))
    if any(
        isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(m) for i in (a, b)
    ):
        raise ValueError("swap slots out of range")
    dm = m[p[b]] - m[p[a]]
    return float(dm * (x[a] - x[b])), float(dm * (y[a] - y[b]))


def _objective(m, p, x, y, d):
    u = vector_sum(m, p, x, y, d)
    return math.hypot(*u)


def vector_greedy(masses, x_mm, y_mm, disk=(0.0, 0.0)):
    m, _, x, y, d = _inputs(masses, None, x_mm, y_mm, disk)
    slots = sorted(range(len(m)), key=lambda k: -(math.hypot(x[k], y[k])))
    order = sorted(range(len(m)), key=lambda i: (-m[i], i))
    p = [-1] * len(m)
    ux, uy = d
    for blade in order:
        k = min(
            (k for k in slots if p[k] < 0),
            key=lambda k: (math.hypot(ux + m[blade] * x[k], uy + m[blade] * y[k]), k),
        )
        p[k] = blade
        ux += m[blade] * x[k]
        uy += m[blade] * y[k]
    return tuple(p)


def greedy_pairing(masses, x_mm, y_mm, disk=(0.0, 0.0)):
    m, _, x, y, d = _inputs(masses, None, x_mm, y_mm, disk)
    n = len(m)
    # Opposite slots are nearest angularly; deterministic heavy/light assignment.
    p = [-1] * n
    slots = list(range(n))
    blades = sorted(range(n), key=lambda i: (-m[i], i))
    while len(blades) >= 2:
        i = blades.pop(0)
        a = slots.pop(0)
        b = min(slots, key=lambda k: (math.hypot(x[a] + x[k], y[a] + y[k]), k))
        slots.remove(b)
        j = min(blades, key=lambda q: (m[q], q))
        blades.remove(j)
        p[a] = i
        p[b] = j
    if blades:
        i = blades.pop()
        a = min(slots, key=lambda k: (math.hypot(d[0] + m[i] * x[k], d[1] + m[i] * y[k]), k))
        p[a] = i
    return tuple(p)


@njit(cache=True, nogil=True)
def _anneal_block(
    m,
    x,
    y,
    current,
    ux,
    uy,
    current_j,
    best,
    best_j,
    temperature,
    alpha,
    slots_a,
    slots_b,
    uniforms,
    offset,
    trace_every,
):
    """Continue current Markov state and RNG; never substitute the incumbent."""
    count = len(slots_a)
    trace_i = np.empty(count // trace_every + 2, dtype=np.int64)
    trace_j = np.empty(count // trace_every + 2, dtype=np.float64)
    used = 0
    for q in range(count):
        a, b = slots_a[q], slots_b[q]
        dm = m[current[b]] - m[current[a]]
        vx = ux + dm * (x[a] - x[b])
        vy = uy + dm * (y[a] - y[b])
        candidate_j = math.hypot(vx, vy)
        delta_j = candidate_j - current_j
        accept = delta_j <= 0.0
        if not accept and temperature > 0.0:
            accept = uniforms[q] < math.exp(-delta_j / temperature)
        if accept:
            current[a], current[b] = current[b], current[a]
            ux, uy, current_j = vx, vy, candidate_j
        if current_j < best_j:
            best_j = current_j
            best[:] = current
        proposal = offset + q + 1
        if proposal % trace_every == 0:
            trace_i[used], trace_j[used] = proposal, best_j
            used += 1
        temperature *= alpha
    return ux, uy, current_j, best_j, temperature, trace_i[:used], trace_j[:used]


def _check_config(c):
    if not isinstance(c, SolverConfig):
        raise ValueError("config must be a SolverConfig")
    limits = (
        (c.seed, 0, 2**32 - 1),
        (c.iterations, 1, 10_000_000),
        (c.restarts, 1, 100),
        (c.trace_every, 1, 10_000_000),
    )
    if any(isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi for v, lo, hi in limits):
        raise ValueError("invalid solver integer configuration")
    if (
        isinstance(c.alpha, bool)
        or not isinstance(c.alpha, (int, float))
        or not math.isfinite(c.alpha)
        or not 0 < c.alpha < 1
    ):
        raise ValueError("alpha must be finite and strictly between zero and one")
    t = c.initial_temperature
    if t is not None and (
        isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0
    ):
        raise ValueError("initial_temperature must be finite and positive")


def _cancelled(cancel):
    return cancel is not None and (bool(cancel()) if callable(cancel) else cancel.is_set())


def solve(masses, x_mm, y_mm, disk=(0.0, 0.0), config=None, method="hybrid", cancel=None):
    """Run every requested proposal, returning an independently summed schedule.

    Temperatures and vector magnitudes use g.mm. Five finite starts do not
    guarantee a global optimum. JIT warm-up is separate from warmed latency.
    """
    started = time.perf_counter()
    c = SolverConfig() if config is None else config
    _check_config(c)
    m, _, x, y, d = _inputs(masses, None, x_mm, y_mm, disk)
    if method not in ("hybrid", "pairing"):
        raise ValueError("method must be 'hybrid' or 'pairing'")
    if _cancelled(cancel):
        raise SolverCancelled("solver cancelled")
    n = len(m)
    original = tuple(range(n))
    greedy = vector_greedy(m, x, y, d)
    pairing = greedy_pairing(m, x, y, d)
    initial_u = vector_sum(m, original, x, y, d)
    initial_blades = vector_sum(m, original, x, y, (0.0, 0.0))
    initial_j = math.hypot(*initial_u)
    greedy_j = _objective(m, greedy, x, y, d)
    pairing_j = _objective(m, pairing, x, y, d)
    best_tuple = min((original, greedy, pairing), key=lambda p: _objective(m, p, x, y, d))
    best = np.array(best_tuple, dtype=np.int64)
    best_j = _objective(m, best, x, y, d)
    trace = [(0, best_j)]
    completed = 0
    arm = max(math.hypot(float(a), float(b)) for a, b in zip(x, y))
    if method == "pairing":
        best = np.array(pairing, dtype=np.int64)
        trace = []
    else:
        seeds = np.random.SeedSequence(c.seed).spawn(c.restarts)
        default_temp = max(float(np.std(m)) * arm * 0.25, 1e-9)
        for restart, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            current = np.array(greedy if restart == 0 else rng.permutation(n), dtype=np.int64)
            ux, uy = vector_sum(m, current, x, y, d)
            current_j = math.hypot(ux, uy)
            if current_j < best_j:
                best[:], best_j = current, current_j
            temperature = default_temp if c.initial_temperature is None else float(c.initial_temperature)
            for offset in range(0, c.iterations, 2048):
                if _cancelled(cancel):
                    raise SolverCancelled("solver cancelled")
                count = min(2048, c.iterations - offset)
                # Three draws per proposal preserve the same RNG prefix for
                # different block/run lengths, without allocating all moves.
                draws = rng.random((count, 3))
                a = (draws[:, 0] * n).astype(np.int64)
                b = (draws[:, 1] * (n - 1)).astype(np.int64)
                b += b >= a
                ux, uy, current_j, best_j, temperature, ti, tj = _anneal_block(
                    m,
                    x,
                    y,
                    current,
                    ux,
                    uy,
                    current_j,
                    best,
                    best_j,
                    temperature,
                    c.alpha,
                    a,
                    b,
                    draws[:, 2],
                    completed + offset,
                    c.trace_every,
                )
                trace.extend((int(i), float(j)) for i, j in zip(ti, tj))
                # Correct drift, retaining the CURRENT (possibly worse) state.
                ux, uy = vector_sum(m, current, x, y, d)
                current_j = math.hypot(ux, uy)
                best_j = _objective(m, best, x, y, d)
                if current_j < best_j:
                    best[:], best_j = current, current_j
            completed += c.iterations
    if _cancelled(cancel):
        raise SolverCancelled("solver cancelled")
    permutation = tuple(map(int, best))
    blade_u = vector_sum(m, permutation, x, y, (0.0, 0.0))
    net_u = vector_sum(m, permutation, x, y, d)
    residual = math.hypot(*net_u)
    mean = math.fsum(map(float, m)) / n
    bx = d[0] + mean * math.fsum(map(float, x))
    by = d[1] + mean * math.fsum(map(float, y))
    correction = arm * math.fsum(abs(float(v) - mean) for v in m)
    bound = max(0.0, math.hypot(bx, by) - correction)
    if method == "hybrid":
        if trace[-1][0] == completed:
            trace[-1] = (completed, residual)
        else:
            trace.append((completed, residual))
    warnings = ["Finite simulated annealing does not guarantee a global optimum."]
    if bound > max(1e-9, residual * 1e-10):
        warnings.append(f"Zero residual is impossible under this model; conservative floor {bound:.6g} g.mm.")
    return SolverResult(
        permutation=permutation,
        initial_ux_gmm=initial_u[0],
        initial_uy_gmm=initial_u[1],
        initial_blade_ux_gmm=initial_blades[0],
        initial_blade_uy_gmm=initial_blades[1],
        greedy_j_gmm=greedy_j,
        pairing_j_gmm=pairing_j,
        initial_j_gmm=initial_j,
        residual_gmm=residual,
        blade_ux_gmm=blade_u[0],
        blade_uy_gmm=blade_u[1],
        net_ux_gmm=net_u[0],
        net_uy_gmm=net_u[1],
        lower_bound_gmm=bound,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        iterations_completed=completed,
        seed=c.seed,
        method=method,
        trace=tuple(trace),
        warnings=tuple(warnings),
    )


def solve_fixed_closure(
    masses, x_mm, y_mm, *, fixed_blade_index: int, fixed_slot: int,
    disk=(0.0, 0.0), mode="demonstration", closure_radius_mm=None,
    measured_disk=None, config=None, method="hybrid", cancel=None,
):
    """Constrain the closing blade to a physical slot, never sorting it away.

    'measured': user-entered effective CG radius and *total* measured bare
    rotor vector replace, not augment, the runout-derived disk estimate.
    'demonstration': nominal radius and supplied simplified disk estimate.
    'pending': cannot optimize until drawings/measurements are supplied.
    """
    m, _, x, y, d = _inputs(masses, None, x_mm, y_mm, disk)
    n = len(m)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (fixed_blade_index, fixed_slot)):
        raise ValueError("closing blade index and slot must be integers")
    if not (0 <= fixed_blade_index < n and 0 <= fixed_slot < n):
        raise ValueError("closing blade index or slot out of range")
    if n < 3:
        raise ValueError("closure constraint requires at least three blades")
    if mode not in ("demonstration", "measured", "pending"):
        raise ValueError("invalid closure interpretation")
    if mode == "pending":
        raise ValueError("drawings/measurements pending; enter closure CG and measured bare-rotor vector")
    if mode == "measured":
        if (isinstance(closure_radius_mm, bool) or not isinstance(closure_radius_mm, (int, float))
                or not math.isfinite(closure_radius_mm) or closure_radius_mm <= 0):
            raise ValueError("measured closure effective CG radius must be finite and positive")
        if (measured_disk is None or len(measured_disk) != 2
                or any(isinstance(v, (bool, str, bytes)) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in measured_disk)):
            raise ValueError("enter the measured TOTAL bare-rotor X/Y vector in g·mm")
        d = tuple(map(float, measured_disk))
        angle = math.atan2(y[fixed_slot], x[fixed_slot])
        x[fixed_slot] = closure_radius_mm * math.cos(angle)
        y[fixed_slot] = closure_radius_mm * math.sin(angle)
    elif closure_radius_mm is not None or measured_disk is not None:
        raise ValueError("measured closure parameters require measured interpretation")
    remaining_blades = [i for i in range(n) if i != fixed_blade_index]
    remaining_slots = [k for k in range(n) if k != fixed_slot]
    fixed_ux, fixed_uy = float(m[fixed_blade_index] * x[fixed_slot]), float(m[fixed_blade_index] * y[fixed_slot])
    effective_disk = (d[0] + fixed_ux, d[1] + fixed_uy)
    reduced = solve(m[remaining_blades], x[remaining_slots], y[remaining_slots],
                    effective_disk, config=config, method=method, cancel=cancel)
    full = [fixed_blade_index] * n
    for k, index in zip(remaining_slots, reduced.permutation):
        full[k] = remaining_blades[index]
    p = tuple(full)
    blade_ux, blade_uy = vector_sum(m, p, x, y)
    net_ux, net_uy = vector_sum(m, p, x, y, d)
    return replace(
        reduced,
        permutation=p,
        initial_blade_ux_gmm=reduced.initial_blade_ux_gmm + fixed_ux,
        initial_blade_uy_gmm=reduced.initial_blade_uy_gmm + fixed_uy,
        blade_ux_gmm=blade_ux, blade_uy_gmm=blade_uy,
        net_ux_gmm=net_ux, net_uy_gmm=net_uy,
        residual_gmm=math.hypot(net_ux, net_uy),
        warnings=reduced.warnings + ((
            "Demonstration: closing blade uses nominal blade radius; cutout mass moment is omitted. Not a physical balance prediction."
            if mode == "demonstration" else
            "Measured total bare-rotor vector replaces runout-derived disk estimate; confirm closure CG and reference phase."
        ),),
    )


def warm_up() -> float:
    """Compile/load the production kernel off the GUI thread; return milliseconds."""
    started = time.perf_counter()
    m = np.array([1.0, 2.0])
    x = np.array([1.0, -1.0])
    y = np.array([0.0, 0.0])
    p = np.array([0, 1], dtype=np.int64)
    draws = np.array([[0.1, 0.2, 0.5], [0.2, 0.1, 0.3]])
    _anneal_block(
        m,
        x,
        y,
        p,
        -1.0,
        0.0,
        1.0,
        p.copy(),
        1.0,
        1.0,
        0.9999,
        np.array([0, 1], dtype=np.int64),
        np.array([1, 0], dtype=np.int64),
        draws[:, 2],
        0,
        250,
    )
    return (time.perf_counter() - started) * 1000
