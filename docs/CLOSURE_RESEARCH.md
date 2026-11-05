# Closing-blade mass-moment treatment — research and decision note

## Purpose and evidence boundary

A circumferentially loaded rotor may have a designated closure blade/lock or other closure feature whose mass and location are not equivalent to an ordinary interchangeable blade. This note defines model choices for representing its contribution to a single-plane static mass-moment calculation. It does **not** identify the geometry of this project's rotor, authorize a particular installation, or replace the controlled drawing and balancing procedure.

The two bundled turbine photographs in the operator guide are contextual images only; neither establishes closure shape, mass, CG radius, angular location or assembly sequence. Likewise, the original accompanying groove illustration is explicitly schematic.

## What the sources establish (and do not establish)

* **EP2549060A2, “Locking of blades in a rotor tangential mounting groove,”** describes a circumferential/tangential groove, one or more locking/loading slots, blade insertion/rotation and lock engagement. Its abstract describes locking slots that provide clearance for assembly; the claims describe inserting blades and a lock and rotating the lock into engagement.[1] This establishes that such groove/slot/lock arrangements are a documented design family. It does not establish that the project rotor has this construction, identify a single “last blade,” or provide the project's closure-blade mass moment.
* **EP2253800 A2, “A balanced rotor for a turbine engine,”** describes a circumferentially slotted rotor with blades and a balance weight whose land extends beneath a blade root and whose mass-adjustment protrusion lies between adjacent blade roots.[2] Its description also discusses loading blades through a loading slot. This confirms that balancing features can be integrated with the blade row/slot system. It does not establish that this design or any particular closure part is present in the rotor under study.

Both are patent disclosures: illustrative technical prior art, not independent validation, manufacturing data, or a production drawing for this application. No copied patent figures are used here.

## Three permissible interpretations

### 1. Measured effective CG radius and total bare-rotor moment (preferred when measured)

For a known closure blade mass `m_c` and its effective CG radius `R_c` from the rotor axis, compute its fixed-slot vector separately. If its slot is `s_c` and its global azimuth is `θ_c`,

```
U_c = m_c R_c (cos θ_c, sin θ_c)              [g·mm]
U_net = U_bladed + U_bare + U_c
```

Use a measured/evidence-based effective CG radius rather than assuming all closure geometries match the nominal blades. Do not count it both as an ordinary blade and as a closure term.

If a measured **total bare-rotor** static unbalance vector is available, it represents the bare rotor/disk together with any included integral features in that measurement. Replace the modeled disk vector with this measured total vector:

```
U_net = U_bladed + U_bare,total + U_c,separate-if-not-included
```

The measurement scope must be explicit. In this version, the closure part is always included as a separate fixed inventory blade. Therefore the measured TOTAL bare-rotor vector entered into the UI must EXCLUDE that closure part; if the closure part was installed during the bare-rotor measurement, this interface cannot subtract it safely and measured mode must not be used with that reading. Re-measure bare rotor without the part or obtain an independently verified excluded-part vector. Identify the measurement plane, angular reference, direction/sign convention, calibration, instrumentation, rotor state, and uncertainty. A total bare-rotor vector is not the same thing as a uniform-solid-disk estimate. Keep the measured vector components and metadata with the run record.

For a closure part with distributed mass, a single effective radius reproduces only its first mass moment about the rotor axis: `R_eff = |∫ r dm|/m_c` for the modeled radial component when its angular spread is negligible. In full generality, use its measured/CAD centroid vector in the rotor plane; converting that vector to a radius/phase is `R_eff=√(x_c²+y_c²)` and `θ_c=atan2(y_c,x_c)`. The CAD centroid must use the actual part definition/material density and correct coordinate datum. A mass-property calculation does not establish installed position, retained hardware or production tolerances.

### 2. Demonstration-only nominal CG; no cutout correction

For a clearly labeled synthetic/demo case only, assign closure part a declared nominal radius `R_nom` and nominal mass, and place it in the selected fixed slot. Do **not** apply any loading-slot, groove cutout or missing-material correction to the disk estimate. State visibly that the closure geometry and mass moment are illustrative assumptions, not measured values. Do not use this interpretation for production optimization, disposition or acceptance.

A nominal-radius closure contribution may be represented as `U_c=m_c R_nom(cos θ_c,sin θ_c)`, but must not be silently conflated with the other ordinary blades or treated as verified total rotor moment.

### 3. Drawing/measurement pending — block solver

If closure mass/radius/geometry, the total bare-rotor vector, or inclusion scope is unknown, do not calculate or present an actionable optimized schedule. Block solver start until the responsible engineer supplies a valid documented interpretation. This is a data-readiness gate, not a warning that may be bypassed by entering zero. The UI default closure slot is the final slot, but the selected slot remains user-selectable; its angular location follows the established slot convention. Default slot placement does not imply that real assembly always closes there.

## Slot reference and angular bookkeeping

The declared global +x defines 0°; slot 1 is at global installation angle φ (default φ=0°). Positive angles are counterclockwise viewed from the agreed face. Let selected closure slot index be `s_c` (one-based) and slot pitch be `360°/N`. With global start phase `φ`,

```
θ_c = φ + (s_c − 1) * 360°/N
θ_j = φ + j * 45°                 for eight runout samples j=0…7
```

Runout readings remain indexed relative to slot 1 at 0°,45°,…315°; global orientation is supplied by `φ`. A closure-slot selection and a global phase shift are distinct inputs. Confirm their convention against a known reference mark. Preserve signed readings and outward-positive convention; TIR is not a signed eight-point radial profile.

## Measurement and CAD workflow

1. Obtain the released rotor and closure-part drawing/model revision, assembly definition and approved measurement procedure. Resolve whether the closure piece is blade, lock, filler, balance feature, or a combination; distinguish separate masses and included hardware.
2. Establish rotor axis, reference face, slot 1, rotation direction, and closure slot. Record whether coordinate values are rotor-fixed or measured in a machine frame.
3. For CAD-derived mass properties, use validated material density and the production-representative part/assembly, with all relevant holes, coatings, fasteners and inserts. Compute mass centroid in a coordinate system whose origin is the rotor axis; transform the centroid to the rotor plane and report `(m,x_c,y_c)` and source revision. Check units and coordinate handedness independently.
4. For measurement, use calibrated equipment and a documented procedure suitable for the actual component and plane. Record setup, support/fixture correction, repeatability, calibration, temperature/state where relevant, angular index and vector sign/phase. Have the responsible balancing engineer determine whether a bare-rotor vector includes the closure part and any other fixed features.
5. Compare independent CAD and measurement results within a pre-established engineering uncertainty/tolerance. Resolve discrepancies before solver use; do not average unexplained conflicts or infer an absent value.
6. Record selected interpretation, evidence, slot, phase, units and inclusion scope alongside the schedule. Independently verify assembled configuration and residual balance against controlled acceptance requirements.

A CAD centroid provides a model-based mass property; a balance measurement provides an observed vector for a defined setup. Neither by itself proves the entire assembly representation is correct.

## Relation to the application's disk estimate

The application's uniform solid-cylinder approximation uses `M=ρπ(D/2)²t` (with SI dimensions) and multiplies the inferred first-harmonic eccentricity by disk mass to estimate `U_disk`. It has no geometric cutout correction for loading/locking slots and is not a substitute for the measured total bare-rotor vector. Therefore, interpretation 1 must explicitly replace that modeled disk vector with measured total bare-rotor `U_bare,total`; do not add both. If closure is included in the bare measurement, this version cannot omit its separate inventory contribution: reject that measurement for this interface, rather than silently double-counting. Interpretation 2 is demo-only and applies no cutout correction. Interpretation 3 blocks solving.

## Limits and required engineering decisions

This is a single-plane static vector model. It does not determine couple balance, flexible-rotor/modal response, speed-dependent behavior, retention integrity, stress, blade compatibility, manufacturability, locking sequence, or allowable residual. No residual limit is supplied or implied here. The actual closing slot must be selected by the user/approved assembly plan (with last slot as software default), not inferred from a generic patent or contextual photograph.

The guide and diagrams are explanatory aids. Do not use them to machine, assemble, inspect or release a rotor. Obtain approved drawings and directions from the design authority.

## Sources

[1] European Patent Office / Google Patents, EP2549060A2, *Locking of blades in a rotor tangential mounting groove*: https://patents.google.com/patent/EP2549060A2/en

[2] European Patent Office, EP2253800 A2, *A balanced rotor for a turbine engine*, publication PDF: https://data.epo.org/publication-server/rest/v1.2/patents/EP2253800NWA2/document.pdf

Photo licenses and attribution are displayed adjacent to each unmodified bundled photograph in `USER_GUIDE.html`. For model equations and the software's present assumptions, see `ENGINEERING_MODEL.md`.
