# Improvement roadmap (2026-08-02, researched at b9c01b5)

Produced by a 7-agent research workflow (source-data gap inventory against
genani.F/anim_to_vtk.cpp, empirical lasso-2.0.4 writer-ceiling tests loaded
in LS-PrePost 4.13, OpenRadioss ecosystem scan, input/perf measurements,
LS-PrePost UX comparison against the Vortex reference d3plot, engineering
quality review) + a completeness critic that independently re-verified the
load-bearing claims. Everything marked "verified" was demonstrated, not
assumed. Impact is rated for THIS workflow: OpenRadioss runs → LS-PrePost
fringes/screenshots, batch automation.

## Strategic frame (critic-corrected)

- OpenRadioss still has NO native d3plot writer (checked through the 2025
  release). The community route is Vortex-Radioss (MPL-2.0, reads binary
  ANIM A-files directly, bundled as optional GUI feature since Jan 2025).
- Therefore: official-dialect feature work competes with Vortex, which reads
  the richer A-file. vtk2d3plot's UNIQUE value is (1) the pyradioss
  port-dialect path — the only route to LS-PrePost for pyradioss output —
  and (2) fidelity/robustness Vortex lacks. Weight investments accordingly.
- The single highest-leverage move may be UPSTREAM: pyradioss's Model holds
  full solid Voigt stress, per-layer shell stress/epsp and beam resultants,
  but its anim_vtk.py writer reduces everything to VONM/EPSP scalars. A
  pyradioss writer patch emitting official-named TENSORS blocks lights up
  this converter's existing tensor paths with zero converter changes.

## Tier 1 — high value, small effort

1. **EROSION_STATUS → is_alive** (official dialect). Verified today: deleted
   elements stay visible AND corrupt the auto-fringe max (TENSILE: 0.750
   from a dead element vs 0.237 true). anim_to_vtk emits it unconditionally;
   semantics 1=eroded confirmed in source + both fixtures (confirm once more
   on the TENSILE failure state in the test). Mirror the OFF path.
2. **Unmapped-array warnings.** The converter recognizes exactly 17 array
   names and silently ignores every other array (real data: zug
   2DELEM_Specific_Energy absmax 2015). One end-of-build diff of consumed
   vs present keys + aggregated warning. Recommended since the M1-5 review.
3. **Part titles.** lasso part_titles verified writable and displayed;
   Vortex shows real names ("Einspannung"), we show "Part 1". Neither VTK
   dialect carries names, so: CLI/API option for an `id: name` mapping file
   (and, later, parse from the starter .rad/.out).
4. **SPH robustness.** Official files from SPH runs contain VTK_VERTEX
   cells → map_topology raises ValueError and the whole conversion dies.
   Skip-with-warning (count masked particles) is one guard clause.
5. **Atomic writes + stale-family cleanup** (batch hazard, critic finding).
   convert() writes the final path directly and never removes pre-existing
   higher family members — overwriting a run leaves a mixed stale family
   that LS-PrePost happily reads as garbage states. Write to temp + rename,
   delete stale `NN` members, clean up on exception.
6. **CLI batch polish.** Per-state progress (a 40-state real run is minutes
   of silence today), Windows glob expansion, non-empty default title,
   lspp_check --fringe defaulting to the LAST state (t=0 renders an
   all-zero fringe that reads as success).

## Tier 2 — the fidelity jump (medium effort)

7. **Strain tensors.** 3DELEM_Strain → element_solid_strain;
   2DELEM_Strain_(upper/lower) → element_shell_strain. lasso writes both
   (ISTRN derived automatically — empirically verified incl. LS-PrePost
   load); the arrays already exist in the zug/TENSILE fixtures and are
   dropped today. Clones of the existing stress branches.
8. **Shell thickness + internal energy.** 2DELEM_Thickness carries real
   necking data in TENSILE (1.700 → 1.0168); element_shell_thickness
   verified writable. Enables %-thinning fringes.
9. **Acceleration → node_acceleration** (verified writable; present in
   official fixtures; Vortex maps it).
10. **Shell tensor local→global rotation** (Vortex parity; the real close
    of M1). Today all shell components are element-local: every raw
    component fringe (σxx…) is systematically wrong (~50% deviation on W3)
    while invariants are exact. Vortex's rotation is the reference recipe.
11. **Beam section forces** → element_beam_axial_force/shear/moments.
    Blocked on one research question: Radioss F1-F3/M1-M6 slot convention
    vs d3plot axial/shear-s/t/torsion/bending-s/t (one instrumented run).
12. **Global energy curves.** History>Global is empty today. Two routes:
    T01 time-series merge (exact) or per-element energy sums (approx).
    lasso nglbv=3/6 verified writable and loading; verify LS-PrePost
    actually PLOTS the curves before investing (open question).

## Tier 3 — performance & memory (structural, not micro)

Measured on a 65 MB synthetic official state: current parse 1.4-1.9 s
(pure-Python split floor ≈ 1.0-1.1 s — micro-optimization is dead-ended).
The real levers, in order:
13. **Flatten the cells structure** (list of 154k tiny arrays → flat
    connectivity + offsets): 2.8× memory overhead gone; prerequisite for
    multiprocessing (unlocks measured 4.0× with 8 workers).
14. **Streaming conversion.** lasso does NOT need the whole series in RAM
    (write_d3plot slices per timestep). Today: fully eager, ~2 GB for 40
    states. Streaming or per-state lazy load caps memory at ~2 states.
15. **Binary/.vtu input.** Neither producer emits binary today, so this is
    coupled to the upstream pyradioss writer question; meshio 5.3.5 was
    tested and is NOT safe as the engine (failed its own binary round-trip).
    Prefer a small hand-rolled binary-legacy reader IF upstream ever emits
    it; the in-memory pyradioss→converter path (no files at all) is better.

## Tier 4 — engineering hardening

16. **Property-based tests** (hypothesis) over the array-combination space —
    the N4/P1/P1b crash class lived exactly there; the synthetic VTK
    generator in test_m8_fixes.py is the ready-made foundation. Key
    invariants: never crash, never silently drop a recognized array,
    read-back shape/value fidelity, warning-on-fill.
17. **CI.** One blocker: conftest hardcodes `.venv\Scripts\python.exe`
    (5 sites) — parametrize to sys.executable, then a plain
    ubuntu+windows pytest matrix (lspp/pyradioss markers auto-skip)
    provides the independent "suite green" referee this project's history
    shows it needs.
18. **Golden management.** Checksum manifest + regenerate-and-diff script;
    give TENSILE a real golden (its shell count changed 208→202 across M7
    with no golden to notice).
19. **Machine-readable ConversionReport** (JSON export; warning codes
    instead of free strings) for pipeline use.
20. **lasso pitfalls to pin in tests** (all verified in 2.0.4): solid-state
    writer silently zero-fills on shape mismatch (bare try/except); node
    deletion mdlopt=1 broken for multi-state; part-energy arrays require
    n_parts == nmmat exactly.

## Open questions worth one instrumented run each

- EROSION_STATUS polarity confirmation on the TENSILE failure state.
- Beam F1-F3/M1-M6 → d3plot slot mapping convention.
- Which /ANIM keyword emits 2DELEM_Thickness (assumed /ANIM/ELEM/THIC).
- Does LS-PrePost plot History>Global from lasso-written nglbv files.
- lasso `_str_codec` override (latin-1) for umlaut titles — what does
  LS-PrePost display? (feeds the M9 P2 fix.)
- Is pyradioss solid stress global-frame (safe as 3DELEM_Stress) or
  corotational?
