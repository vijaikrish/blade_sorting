# Research notes — hybrid rotor blade balancer

## Decision and provenance

Reviewed before application development. The chosen baseline is **Vijai's existing `vijaikrish/blade_sorting`**, `master` commit `7d01cdf9d3256a7b7a21a59a48420b46e516538a`. Vijai explicitly chose upgrading it instead of creating a second, third-party fork. Development branch: `feature/hybrid-rotor-blade-balancer`. This research was recorded before the owner authorized publication.

| Candidate | Inspected implementation | Extensibility / decision |
|---|---|---|
| `vijaikrish/blade_sorting` | Hard-coded weights; 1,000 random permutations; full vector sums and plots coupled in one script. Trigonometry mistakenly receives degree-valued angles; phase uses quadrant-losing `atan(y/x)`.[4] | Best ownership/provenance fit; replace the active application, preserve original files under `legacy/`. |
| `AndreTeixeira1998/blade-sorting` | Permutation genetic algorithm, disk moment, Tk/Excel workflow in one large script; commit `cbe185fa0e8b4d85a197b3c07f3d990e4411c8b2`.[5] | Relevant comparator, not a reusable modular backend. No declared license found. No source copied into the application. |
| `idukkigold/TurbineBalancingProblem` | Constructive greedy placement and separate factorial-enumeration scripts; commit `1e29e903e1ff774490cff35e28d2647746388251`.[6] | Useful conceptual comparator and small-case oracle idea; not a production GUI. No declared license found. No source copied into the application. |

At the time of this research, none of these repositories declared a license. Public visibility is not an open-source license. Subsequently the owner chose MIT for the code in **his own** `blade_sorting` repository; source from the two other unlicensed projects was not incorporated. The separately licensed photographs are credited in `ASSET_LICENSES.md`.

## Literature: what is established

The turbine balancing problem assigns unequal blade masses/mass moments to fixed angular slots. It has a specialized quadratic assignment formulation, and exact enumeration is unsuitable at practical blade counts. Pitsoulis, Pardalos and Hearn (2001) describe a GRASP implementation; the accessible abstract and introduction, not the paywalled full paper, were reviewed.[12]

Opposite heavy/light pairing and interchange heuristics are established fast baselines. The ACO comparison reports sensitivity to starting placement and fast, useful exchange heuristics; this does not establish universal superiority of one algorithm.[1][2]

Sun et al. (2021) model blade mass moments and disk concentricity using a cloud-adaptive genetic algorithm. Their reported improvements are specific to their tested datasets; mass moment is preferable to mass alone when blade CGs differ.[8]

Liu and Li (2021) study simulated annealing with reheating/tabu ideas. Unterauer et al. (GECCO 2024) include bare-disk imbalance in a QUBO model and show strong results for constraint-preserving permutation-swap SA. Their results motivate a compact classical solver, not a requirement for quantum hardware.[1][3]

This is a focused, reproducible literature review, not a claim that every paper through 2026 has been exhaustively reviewed. No finite SA run, five starts or otherwise, guarantees a global optimum or zero residual.

## Mathematical basis (project derivation)

Let `v_k = r_k (cos(theta_k), sin(theta_k))`, `w_k = m_pi(k)`, and `d = U_disk`, with lengths in mm, mass in g and unbalance in g·mm. The requested objective is `J = ||d + sum_k w_k v_k||`.

Squaring the norm gives

`J² = ||d||² + 2 sum_k w_k (d·v_k) + sum_k sum_l w_k w_l (v_k·v_l)`.

This is a quadratic assignment objective with a linear disk term. Minimizing `J` and `J²` has the same minimizer, but **their SA acceptance temperatures are not interchangeable**. Implement acceptance on `delta J`, in g·mm, as requested. Store blade IDs as permutation indices so equal masses remain distinguishable.

For a swap of slots `a,b`, only two terms change:

`delta U = (w_b - w_a) (v_a - v_b)`.

The production inner loop will use this O(1) update, with independent full-sum verification of the final candidate and bounded periodic drift correction. The best-so-far result must never be worse than the original order or either constructive baseline.

## Runout model and engineering limits

The specified eight-point DFT coefficients are a least-squares **first-harmonic radial fit**. NIST documents single-trace limitations: spindle and workpiece errors cannot be separated from one trace. Published roundness work identifies the first harmonic with small center movement and higher harmonics with profile shape.[9][10]

Implement exactly the requested model:

1. Subtract the mean indicator reading for numerical stability; obtain `ex=(2/8)sum(reading*cos theta)` and `ey=(2/8)sum(reading*sin theta)`.
2. Uniform **solid** disk: `M_g = rho*pi*(D_mm/2000)^2*(t_mm/1000)*1000`.
3. Estimated mass unbalance: `(Ux,Uy) = M_g*(ex,ey)`.
4. Requested slot model: `r_k = R_CG + ex*cos(theta_k) + ey*sin(theta_k)`.
5. Mode A forces both disk vector and eccentricity to zero, ignoring inactive rotor/runout fields.

**Limits that must travel with results:**

- Radial runout does not by itself establish mass eccentricity. Bore/hub/cut-outs, density gradients, datum/shaft motion and probe effects are excluded. Mode B is a geometry-derived estimate, not a measured balancing-machine vector.
- Readings are signed radial values, not eight peak-to-peak TIR values. Positive means outward in the adopted rotor-fixed reference. Slot 1 is 0° at +x; angles increase counterclockwise viewed from a consistently declared face. Probe polarity and the mapping from instrument rotation angle to rotor-fixed position must be verified with a known eccentricity. Do not automatically negate readings.
- Eight samples can alias higher spatial harmonics into the 1x component. Display the residual RMS after subtracting mean and fitted 1x; this flags omitted sampled content but cannot prove absence of aliasing.
- The requested slot-radius expression is a **radial projection model**, not a rigid translation of all blade CG coordinates. A true translation is `p_k = e + R*(cos theta,sin theta)`. For equal masses at N>=3, the requested radial-only model contributes `M_blades*e/2`, whereas rigid translation contributes `M_blades*e`. Do not silently substitute one geometry for the other.
- A common blade CG radius is an assumption. This is single-plane static pre-balancing, not couple/modal/dynamic balancing, stress assessment, or overspeed qualification.
- The final design authority must verify actual geometry, phase/polarity, blade moment measurements, mounting constraints, and residual unbalance on appropriate equipment before service. ISO 21940-11:2016 with Amendment 1:2022 concerns rigid-behaviour balancing; flexible behaviour is outside that part. No acceptance grade/table is implemented or implied.[11]

## Solver design and performance contract

Requested default: 100,000 swaps per start, five independently seeded starts, geometric cooling `alpha=0.9999`. Start 1 uses descending-mass vector-greedy construction; additional starts use deterministic seed-derived permutations/perturbations rather than five identical states. Retain a standalone heavy/light opposite-pairing baseline; odd counts use the closest available opposing slots and are labelled approximate pairing.

A dimensionally scaled initial temperature is chosen from characteristic swap changes. Numba compiles the scalar swap loop; GUI work is asynchronous. Report compilation/cold-start time separately from warmed solve latency. The 500 ms target applies to complete warmed solver calls, including initialization and all five starts, on the measured host; it is not asserted for every computer, GUI rendering or first JIT compilation. Never shorten the iteration count silently to make a benchmark pass.

A conservative reachability bound uses the mean mass `mbar`: `B = d + mbar*sum(v_k)` and `C = max(||v_k||)*sum(|m_i-mbar|)`. Every permutation has `J >= max(0,||B||-C)`. A positive bound proves zero residual impossible under this model; a zero bound does not prove attainability. Finite-permutation small-N exact tests provide independent truth, not a general proof for large N.

## Upgrades and verification plan

- Modular `rotor_math.py`, `solver.py`, `data_io.py`, `gui.py`, `main.py`; original experiments remain inactive in `legacy/`.
- Strict numeric/finite/positive validation, unique blade IDs, transactional CSV import, safe mounting export, visible units and input/result invalidation.
- Deterministic search, O(1) swap deltas, raw initial and pairing comparisons, best-so-far convergence, lower-bound messaging and complete final recomputation.
- PySide6 workbench with threaded solver, vector plot, slot map/runout trace, convergence and exportable mounting schedule.
- Tests before implementation: analytic harmonic/mass cases, radians/quadrants, exhaustive small permutations, odd/even counts, equal masses, impossible compensation, invalid inputs, reproducibility, CSV errors, GUI interactions and stale-result prevention.
- Benchmark full default 500,000-swap solves for even and odd counts through 120 blades, with and without runout; capture real UI screenshots and independent code review before presenting for approval.

## Sources

[1] https://ar5iv.labs.arxiv.org/html/2405.06412
[2] https://www.degruyterbrill.com/de/document/doi/10.1515/tjj-2017-0060/html
[3] https://iopscience.iop.org/article/10.1088/1742-6596/1939/1/012098/pdf
[4] https://github.com/vijaikrish/blade_sorting
[5] https://github.com/AndreTeixeira1998/blade-sorting
[6] https://github.com/idukkigold/TurbineBalancingProblem
[8] https://www.mdpi.com/2073-8994/13/5/832
[9] https://acris.aalto.fi/ws/portalfiles/portal/142407979/1-s2.0-S0263224124003464-main.pdf
[10] https://www.itl.nist.gov/div898/handbook/mpc/section3/mpc344.htm
[11] https://www.iso.org/standard/54074.html
[12] https://www.sciencedirect.com/science/article/abs/pii/S0377221700000291
