# Engineering model and verification examples

> **Use:** single-plane static assembly pre-balancing. This is not a release-to-service calculation. Confirm the mounting reference, blade CG assumptions and real residual unbalance with the responsible balancing engineer.

## Units and reference

Mass inputs are **g**, diameter/thickness/radius/runout are **mm**, density is **kg/m³**, result is **g·mm**. Angle calculations use radians internally and degrees in the mounting schedule. Global **0° is +x**; slot 1 is at the installation angle `φ` (default **0°**) and slots increase **counterclockwise**, viewed from the same declared face for measurement and mounting. The eight runout readings start at slot 1 and advance at 45° intervals relative to it. A vector points towards the heavy side in that coordinate system; it is not automatically an add-weight instruction.

Use signed individual indicator readings, positive outward in this model. Do not enter total indicated runout (peak-to-peak TIR) into each angular cell. A measured probe waveform versus rotation angle may need phase/sign conversion before it is a rotor-fixed radial profile. Verify the mapping with a known eccentricity; software cannot identify your probe convention.

## 1. Eight-point first harmonic

For `theta_j = 2*pi*j/8`, `j=0..7`, first remove the mean `rbar=sum(r_j)/8` for floating-point conditioning, then compute:

```
ex = (2/8) sum((r_j-rbar) cos(theta_j))   [mm]
ey = (2/8) sum((r_j-rbar) sin(theta_j))   [mm]
e  = hypot(ex,ey)                       [mm]
phase = atan2(ey,ex), modulo 360°
fit_j = rbar + ex cos(theta_j) + ey sin(theta_j)
RMS_omitted = sqrt(sum((r_j-fit_j)^2)/8)  [mm]
```

Subtracting the mean does not change the exact harmonic coefficient at equal angular spacing. Mean subtraction reduces cancellation error when indicator readings contain an offset. The formulas above give `(ex_local, ey_local)` in the slot-1 frame; rotate both coefficients by installation angle `φ` before calculating the global bare-rotor vector or projected slot radius. Zero vector phase is undefined, not 0°. This is a first-harmonic radial least-squares estimate, not a high-precision general nonlinear geometric circle fit. NIST's single-trace equations and measurement-error discussion support this analysis.[10][13]

Eight samples are sparse: higher spatial harmonics can alias into the sampled 1x component. A small omitted RMS does not prove the actual surface is sinusoidal or the mass center has been identified.

## 2. Disk mass and estimated unbalance

Under the explicit **uniform, solid cylinder** assumption:

```
D_m = D_mm / 1000
t_m = t_mm / 1000
M_kg = rho * pi * (D_m/2)^2 * t_m
M_g  = 1000 * M_kg
U_disk = M_g * (ex,ey)                  [g·mm]
```

No bore, hub, holes, pockets, density variation or separate shaft mass is included. This formula is a volume/mass derivation; it is not an ISO residual-unbalance tolerance. Geometric 1x runout is not a measured mass-unbalance vector. Mode B results are therefore **estimated**.

Mode A deliberately sets `ex=ey=U_disk,x=U_disk,y=0`; inactive disk/runout fields do not influence blade-only results.

## 3. Blade radius and objective

```
R = explicit CG radius, or D/2 + CG offset  [mm]
theta_k = φ + 2*pi*k/N, k=0..N-1
(ex,ey) = rotation(φ) * (ex_local,ey_local)
r_k = R + ex*cos(theta_k) + ey*sin(theta_k)
v_k = (r_k*cos(theta_k), r_k*sin(theta_k))
U_blades = sum(m_pi(k) * v_k)              [g·mm]
U_net = U_disk + U_blades                 [g·mm]
J = hypot(U_net,x, U_net,y)               [g·mm]
```

Every blade must have the same nominal CG radius for this mass-only representation to be applicable. Measured blade mass moments are needed if blade CG offsets differ materially; the literature explicitly considers such moment measurements.[3][8]

### Geometry qualification

The requested formula changes **radial lengths only** at fixed shaft-centered angles. It does not translate the whole blade circle rigidly. A rigid translation would instead use:

```
x_k = ex + R*cos(theta_k)
y_k = ey + R*sin(theta_k)
```

For N>=3 equal masses, the radial-projection formulation yields the blade-set translation term `M_blades*e/2`; a rigidly translated circle yields `M_blades*e`. Example: eight 100 g blades and 0.05 mm eccentricity contribute **20 g·mm** in the requested radial model, versus **40 g·mm** under rigid translation. These values were calculated independently in Python. The application intentionally does not switch geometry without approval.

## 4. Optimization

Vector greedy assigns descending masses to the empty slot minimizing the current vector magnitude. The pairing baseline places near-equal/heavy-light selected pairs in opposing slots; odd N has no exact diametrical slot pair and uses the nearest available opposite slot. The exact implemented rule is documented in the solver docstring.

For swapping assigned blade masses `m_a,m_b` between slots `a,b`:

```
delta_U = (m_b-m_a)*(v_a-v_b)
U_candidate = U_current + delta_U
delta_J = norm(U_candidate)-norm(U_current)
```

Accept all non-worsening proposals; accept a worsening proposal with probability `exp(-delta_J/T)`. Temperature has units of **g·mm**. Cooling uses `T_next=0.9999*T`. Five seed-derived starts each perform 100,000 proposals by default. The best solution is retained even when a later proposal/start is worse. The initial input order and both heuristic seeds remain comparison candidates. Finite-budget SA provides **no global-optimum guarantee**.[1]

Delta updates avoid a complete blade summation for each proposal. The candidate ultimately returned must be independently recomputed using all blades; periodic exact recomputation controls accumulated round-off. The implementation records both iteration count and latency, separating JIT warm-up from solve time.

### Conservative correction-authority bound

With `mbar=mean(masses)`, let `B=U_disk + mbar*sum(v_k)` and `C=max(norm(v_k))*sum(abs(m_i-mbar))`. By the triangle inequality,

```
J >= max(0, norm(B)-C).
```

A positive bound proves that zero residual is impossible under this model. It does not give the exact optimum, and a zero bound does not establish that perfect compensation is possible. Equal-mass blades have no permutation authority: rearranging them cannot correct the disk's heavy side.

## 5. Independent calculation reference

Synthetic data are explicitly generated, not claimed as instrument measurements:

```
r(theta) = 0.7 + 0.03*cos(theta) - 0.04*sin(theta)
           + 0.1*cos(2*theta)                [mm]
D = 500 mm; t = 50 mm; rho = 7850 kg/m³
```

Independent Python calculation (before backend implementation):

| Quantity | Reference |
|---|---:|
| ex | +0.030000 mm |
| ey | -0.040000 mm |
| 1x amplitude | 0.050000 mm |
| Disk mass | 77,067.194783 g |
| Disk Ux | +2,312.015844 g·mm |
| Disk Uy | -3,082.687791 g·mm |
| Disk magnitude | 3,853.359739 g·mm |
| Heavy-side phase | 306.869898° |

The +0.7 mm mean and 0.1 mm 2x term must not bias the recovered 1x vector. Tests also exercise odd/even blade counts, radians, quadrants, zero phase, equal masses, impossible compensation, independent full sums and small exhaustive permutation oracles.

## 6. Fixed closing blade and measured rotor

For a selected blade ID and one-based physical slot `s`, the closing blade remains at `θ_s = φ + 2π(s−1)/N` while the other blades are permuted. In demonstration mode it has its inventory mass and nominal blade radius; **missing cutout mass moment is omitted**. That result is not a physical balance prediction. In measured mode an explicitly verified effective closure CG radius replaces its nominal radial placement, and the user-entered **total measured bare-rotor vector** replaces, rather than adds to, the runout-derived solid-disk vector. The measured bare-rotor vector must **exclude the closure blade** because the solver already includes it as one fixed inventory row; if it was installed during measurement, do not use that reading in this interface. Pending mode blocks the solver. See [closure decision note](CLOSURE_RESEARCH.md) and the [illustrated operator guide](USER_GUIDE.html).

## 7. Final engineering checks

1. Check blade identities, calibrated scale resolution and common CG/moment assumption.
2. Check slot numbering, observation face, rotation/angle convention and probe polarity against a known eccentricity.
3. Check the solid-disk estimate against real geometry and a measured balancing-machine vector before treating it as mass unbalance.
4. Check mounting constraints, blade families, aerodynamic/mistuning constraints and any prohibited permutations. These constraints are **not** represented in this unconstrained sorting model.
5. Verify the assembled rotor on suitable balancing equipment. Check residual static/couple/modal requirements, runout and assembly repeatability under the controlled specification and final design authority.

No ISO/API pass/fail grade is asserted. ISO 21940-11:2016 with Amendment 1:2022 concerns rigid-behaviour procedures/tolerances; flexible behaviour is treated separately. Only the official abstract/scope was consulted, not a controlled full standard.[11]

## Sources

[1] https://ar5iv.labs.arxiv.org/html/2405.06412
[3] https://iopscience.iop.org/article/10.1088/1742-6596/1939/1/012098/pdf
[8] https://www.mdpi.com/2073-8994/13/5/832
[10] https://www.itl.nist.gov/div898/handbook/mpc/section3/mpc344.htm
[11] https://www.iso.org/standard/54074.html
[13] https://www.itl.nist.gov/div898/handbook/mpc/section3/mpc3441.htm
