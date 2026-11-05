# Local verification report — Blade sorting

## Scope and provenance

Local branch `feature/hybrid-rotor-blade-balancer` of `vijaikrish/blade_sorting`. Existing experimental code is preserved under `legacy/`. Research was recorded in `RESEARCH_NOTES.md` before implementing the new application. At the time of this pre-publication verification, the origin push URL was disabled and publication had not yet been authorized. **Generated `artifacts/` outputs cited below are local-only and excluded from Git; rerun the named scripts to regenerate them.** The independent reference-case JSON fixture is retained for tests.

## Numerical and workflow checks

- `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest -q --tb=short`: **85 passed**, 16 third-party matplotlib/PySide6 deprecation warnings (no failed tests).
- `.venv/Scripts/python.exe -m ruff check . --output-format concise`: **All checks passed**.
- Tests include independent harmonic fits with mean/2× interference, solid-disk mass and unbalance, zero-runout mode A, mode B, odd/even blade counts, small-case exhaustive permutation oracle, exact swap-delta checks, equal-mass and impossible-zero cases, seeded reproducibility, distinct starts, result-vector recomputation, malformed CSV and injection-safe export, mode switching, cancellation/stale-results, auto-radius invalidation, real Qt worker completion and chart axis units.
- Independent fail-closed review of the final staged code diff passed: **no reproduced high-priority logic or security defects**. The reviewer ran a focused offscreen suite (78 passed), checked solver state/restarts/trace, worker cancellation and GUI-thread rendering, CSV atomic publication, vector units, and scan results (no secret/shell/eval/pickle findings). This is a local code review, not an engineering certification.
- The auto-radius/diameter result-invalidation path was re-tested explicitly after an earlier independent reviewer raised a possible stale-export concern. Diameter is connected to `_edited` and the regression test confirms export is disabled after changing it. The earlier concern was not reproducible.
- A screenshot-capture harness initially polled `QTest.qWait(10)` from Python while the worker ran and measured ~5.5 s for a warmed solve. The cause was harness-induced GIL contention. Using a nested Qt `QEventLoop` to observe the actual worker completion measured **59 ms solve** and **146 ms warm-up** on the same synthetic 32-blade combined case. GUI painting/export are outside solver timing. The harness now uses the Qt event loop.

## Full default-workload benchmark

Command: `.venv/Scripts/python.exe scripts/verify_performance.py --repeats 3 --max-blades 120`. Machine: Windows 11, AMD64 Family 25 Model 33, CPython 3.14.9, NumPy 2.5.4, Numba 0.68.1, PySide6 6.12.0. Source of individual cases: `artifacts/benchmark_matrix.json` (synthetic blade data, both modes).

| Measured item | Result |
|---|---:|
| Cases (N=2…120, both modes) | 238 |
| Complete solver calls | 714 |
| Swaps per complete call | 500,000 (five × 100,000) |
| Median warmed wall time | 82.6 ms |
| Maximum warmed wall time | 139.0 ms |
| Calls below 500 ms | 714/714 |
| Fresh-cache JIT compilation | 1,886.9 ms |
| Fresh process import + JIT | 2,320.5 ms |

Each benchmark call independently recomputes the final unbalance from its complete mounting permutation and checks the announced work count, finite output, initial/greedy/pairing dominance and a conservative lower bound. The 500 ms target is **machine-specific, warmed, solver-only**, not a first-launch, UI-rendering, or hardware-independent guarantee. Five-start SA cannot guarantee the global optimum or near-zero residual.

## Real desktop GUI exercise

`QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe scripts/capture_preview.py` created screen grabs from the real PySide6/Matplotlib widget and exported a synthetic mounting schedule. Evidence: `artifacts/ui_1280x720.png`, `artifacts/ui_slot_map.png`, `artifacts/ui_convergence.png`, `artifacts/ui_mounting.png`, `artifacts/ui_verification.json`, `artifacts/demo_mounting_schedule.csv`. The screenshots were visually checked; the polar radial rings were reduced to three and moved away from the legend. The 32-blade synthetic combined-mode example reported **0.257386 g·mm residual**, **500,000 proposed swaps**, seed **42**, and **32 exported rows**. This synthetic number is not a physical balancing-machine measurement or acceptance criterion.

## Engineering limitation

The uniform-solid-disk mass and runout-derived eccentricity are an **assumed model**, not an independently measured bare-rotor mass unbalance. Runout polarity, face, reference, actual geometry, blade CG/moment variation and mounting constraints must be checked by the responsible balancing engineer. Use the final assembly balance measurement and controlled acceptance specification before service. The optimization is single-plane static only: not couple, flexible-rotor/modal or high-speed balancing.

## Revision verification — installation angle, closure, inventory, illustrated Help

The preceding 85-test/82.6 ms evidence refers to the **earlier baseline**. The following checks were run on the later revision (same local branch) with `QT_QPA_PLATFORM=offscreen`; they supersede the baseline for current functionality:

- `.venv/Scripts/python.exe -m pytest -q --ignore=legacy --tb=short`: **116 passed**, 17 third-party/pytest deprecation warnings, no failures. New tests cover phase-reference rotation, fixed closing slot, measured-vector replacement, pending-data blocking, manual blank inventory, seeded sample CSV, Qt worker, and Help content.
- `.venv/Scripts/python.exe -m ruff check . --exclude .venv,legacy,artifacts`: **All checks passed**.
- `.venv/Scripts/python.exe scripts/verify_performance.py`: **238 synthetic cases, 714 complete warmed solver calls**, each 500,000 proposed swaps; **357,000,000** proposals total. Median **85.51 ms**, maximum **250.63 ms**, **714/714 below 500 ms**. Fresh-cache JIT **1,956.68 ms**; cold process import plus JIT **2,436.31 ms**. Full case evidence in `artifacts/benchmark_matrix.json`. This matrix exercises baseline modes across N=2–120; it does **not** exhaustively measure every fixed-closure combination.
- `.venv/Scripts/python.exe scripts/capture_preview.py` exercised the real Qt worker with **100** synthetic blade masses (99 × 30–40 g + 50 g closing blade), fixed closing slot **100** and installation angle **25°**. It completed **500,000** proposals in **116.75 ms** (after **223 ms** warm-up), exported **100** schedule rows, and captured `artifacts/ui_inventory.png`, `artifacts/ui_installation.png`, `artifacts/ui_closure.png`, `artifacts/ui_help.png`, `artifacts/ui_mounting.png`, and plot views. `artifacts/ui_verification.json` is the exact latest run. The **0.0679104 g·mm** residual is **demonstration only**: closure CG at nominal radius, cutout moment omitted, no physical acceptance claim.
- The Help tab was loaded from local `docs/USER_GUIDE.html` in a real `QTextBrowser` and screenshot-checked. The two contextual photos are attributed in that guide with Commons source and CC BY-SA license; original schematics are labelled. They do not establish the user's actual closure construction.

At the time of this verification, no commit, push, PR or release had been made. The measured mode demands an effective closing-blade CG radius and total bare-rotor vector **excluding** that separate closing blade. The checkbox records user verification, not independent confirmation of measurements. Final design approval and assembled-rotor balancing remain outstanding.

## Guided Inputs / Outputs revision

- The workbench now separates top-level **Inputs**, **Outputs**, and **Help**. Inputs has six numbered stages: Analysis → Blades → Runout → Installation → Closure → Method; Previous/Next step buttons and a step explanation remain visible. Outputs has a scrollable four-graph Overview plus full Vectors, Slot map, Runout, Convergence, and Schedule views. The individual graph views retain Matplotlib zoom/pan controls. The Slot map 1× curve now spans from the installation angle through +360° in the same global frame as the blade angles (verified by a regression with slot 1 at 25°). Editing inputs still invalidates the result and export.
- `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest -q --ignore=legacy --tb=short`: **122 passed**, one external pytest deprecation warning. `.venv/Scripts/python.exe -m ruff check . --exclude .venv,legacy,artifacts`: **All checks passed**. New real-Qt tests cover step ordering, navigation, physical input placement, explanations, scrollable overview, rendered thumbnails, graph selection and stale-result handling. Solver mathematics were unchanged.
- The Qt worker and export harness ran with **100** synthetic blades, installation angle **25°**, fixed closure slot **100**, and **500,000** proposals in **166.79 ms** warmed solve time on this run. The result remains demonstration-only. Screenshots at **1400 × 900**, **1280 × 720**, and **900 × 600** were inspected. Key evidence: `artifacts/ui_900x600_inputs.png`, `artifacts/ui_900x600_outputs.png`, `artifacts/ui_runout_input.png`, `artifacts/ui_method.png`, `artifacts/ui_overview.png`, and `artifacts/ui_verification.json`. The 900 × 600 overview scrolls to the lower graphs.
- At the time of this guided-workflow revision, the prior 714-call, N=2–120 warmed baseline solver matrix was retained because only the UI/guide changed. It was subsequently regenerated for publication (see preflight below). Both results are machine-specific and do not cover every closing-blade geometry or total UI response time.

The owner subsequently approved MIT licensing and GitHub publication. This report records **local** verification and is not a claim of engineering certification.

## Personal-project publication preflight

- Rebranded the window/header, entry point, launcher and README to **Blade sorting**. The README includes a five-stage illustrated walkthrough with seven tracked screenshots in `docs/images/` plus the offline illustrated Help guide. The bundled photographs have distinct attributions and CC BY-SA license conditions in `ASSET_LICENSES.md`; the code license is MIT.
- `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest -q --tb=short`: **124 passed**, one external pytest deprecation warning. `.venv/Scripts/python.exe -m ruff check . --exclude .venv,legacy,artifacts`: **All checks passed**.
- `.venv/Scripts/python.exe scripts/verify_performance.py`: **238 synthetic cases, 714 complete warmed solver calls**, each with **500,000** proposals; median **99.14 ms**, maximum **214.16 ms**, **714/714 below 500 ms**. Fresh-cache JIT **2,028.32 ms** and cold-process import plus JIT **2,568.83 ms**. The recorded matrix in `artifacts/benchmark_matrix.json` was regenerated for this preflight; these are solver-only, machine-specific timings. The fixed-closure GUI example is separate.
- `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe scripts/capture_preview.py` exercised the real Qt worker and schedule export with **100** synthetic blades, fixed closing slot **100** and installation angle **25°**. It made **500,000** proposals, returned **0.0679104 g·mm** synthetic demonstration residual and exported **100** rows. The result is not a measured rotor acceptance value. The latest timing and numerical fields are in `artifacts/ui_verification.json`; the selected regenerated screenshots are included under `docs/images/`.
