# M10 + M11 verification (2026-08-03, commits 795e03a + ebf2588)

Verified by 23 agents: 8 verifiers (C1 latch, T5 atomic, T2+minors, M11
arrays, M11 infra, full roadmap inventory, test-rigor with 8 mutation
probes, fresh-eyes on both diffs) + 15 adversarial adjudications (12
in-flow, 3 overflow). Suite at verification: 108 passed / 0 skipped.

## M10 contract scoreboard

| Item | Status | Notes |
|---|---|---|
| C1 erosion latch | **FIXED** — zug 864/864 alive matching the Vortex golden (LS-PrePost render pixel-diff vs golden 0.02%, pre-fix 1.59%); TENSILE exactly {2,3} dead at 15/16+; resurrection clamped; all three families; OFF branch untouched; 4 tests fail pre-fix | gaps: warning text at old :469-472 NOT revised (was a contract element); latch semantics under-pinned (3 latch mutations survive — minor); golden VALUES not compared in the test (shape only); latent: rescue computed over the one-filled array — a state missing EROSION_STATUS disables the rigid rescue or fabricates a death at the fill boundary (adjudicated MINOR: unreachable from real anim_to_vtk output, which emits the array in every state; fix is cheap — compute ever_alive before the fill); the fill warning says "source OFF array missing" when the source is EROSION_STATUS |
| T5 atomic rework | **PARTIAL** — whole-family re.escape cleanup works (1/2/3-digit + brackets), mid-WRITE failure preserves the family, no temp litter | **2 CONFIRMED majors remain**: (a) regex is case-SENSITIVE on case-insensitive Windows (BOX.d3plot02 survives); (b) the old family is deleted BEFORE new files move in — a failure in that window (locked file: LS-PrePost holding the previous d3plot open is routine) destroys the pre-existing good family, no restore path. M9 acceptance still not fully met. Cleanup-exception masking not fixed; the M9-named hollow test left vacuous; the new injection test fails before the try block (hollow) |
| T2 absmax | **FIXED** — all-states accumulate, zug warning reports 2015; pinned | %.4g formatting caps the pin's precision (minor) |
| Item-4 minors | 5 of 9 real (f64-72, R2-eps/stress-lower value pins — the ×3.7 mutants die, part-names BOM/cp1252, OFF+EROSION warning, default title) | surrogate test contains NO assert (verified by read); M8-named vacuous test untouched (a sibling was fixed instead); f64-72 test blind to the fix; UTF-16 part-names still silent; progress is cosmetic (0% of wall-clock — the VTK reading phase stays silent) |
| lspp_check last-state | **NOT FIXED — claimed but inoperative**: `state last;` is a silent LS-PrePost no-op; default fringe PNG is byte-identical to state 1 (t=0) | this is the exact all-zero-fringe trap the roadmap warned about; the commit message claims it fixed (over-claim, round 6) |
| Records | plan.md itself mostly truthful (best round yet); the commit message over-claims lspp_check | |

## M11 scoreboard

| Item | Status | Notes |
|---|---|---|
| Shell strain (membrane path) | **FIXED** — TENSILE values exact to f32 (2e-07); shear convention validated against the Vortex golden's own strain invariants (I1/I2 rel ~5e-06; halving or doubling shear degrades to 1e-2 — convention correct); ISTRN set; LS-PrePost strain fringe renders | |
| Shell strain (2-layer path) | **CONFIRMED MAJOR (S1)** — on the REAL zug file (carries membrane+lower+upper) the membrane branch preempts: both real layers are DISCARDED, membrane duplicated into both slots, while strain_layers=2 is still declared to MAXINT reconciliation; a single layer alone ((upper) or (lower) without partner/membrane) is dropped ENTIRELY with no warning — the P1b class recurring; the three strain names were added to _RECOGNIZED_CELL so the M9 unmapped-net is explicitly OFF for exactly these arrays; the 2-layer branch executes in 0 of 108 tests; slot-swap and delete-branch mutants both survive | precedence must invert (2-layer wins over membrane, like stress); add *_only flags; pin zug's real gradient against the golden |
| Solid strain | **NOT SHIPPED** — element_solid_strain appears nowhere; 3DELEM_Strain not even recognized (reported as unmapped) | roadmap item 7 is half-done |
| Shell thickness | **FIXED** — TENSILE necking 1.700→1.01681 exact | internal energy NOT shipped (item 8 half-done; the discarded half carries the largest live numbers: W3 6.05e+09) |
| Acceleration | **FIXED** — W3 values verified | |
| Silent zero-fill (S2) | **CONFIRMED MAJOR ×3** — strain, thickness, AND acceleration all zero-fill missing states with no warning (the N1 class, re-introduced for every new array; thickness 0.0 is physically alarming in a fringe); no local-frame warning for strain (the M1 class) | wire all new arrays into the N1 fill machinery + extend the M1 warning |
| CI | **DONE-with-nits** — YAML valid, markers auto-skip proven without local venv, conftest sys.executable fix real | matrix lacks 3.14 (the dev interpreter); lint excludes tools/ and ruff currently fails there; lasso installed unpinned before the pin re-applies |
| JSON report | **DONE** — schema sane, umlauts survive, tests fail pre-fix | warning CODES not done (still free prose — pipelines must substring-match English); no machine-readable failure mode |
| Golden manifest | **DEAD CODE** — no manifest generated or committed, no test, no CI step, nothing references the tool (verified by grep); files added to the tree are invisible to verify | |
| lasso pitfalls | **1 of 3** — one real pin; the solid-stress test never reaches the silent-zero path and its comment records the OPPOSITE of the actual behavior; mdlopt=1 has no test | |

## Roadmap ledger (the "rest of the roadmap is done" claim)

**Done (6):** 1 erosion (latched), 2 unmapped warnings, 3 part titles,
4 SPH skip, 9 acceleration, 17 CI.
**Partial (7):** 5 atomic (window + case), 6 CLI (lspp_check no-op,
progress cosmetic), 7 strain (shell-membrane only; solid missing; S1),
8 thickness (energy missing), 18 golden mgmt (manifest dead; no TENSILE
golden), 19 report (JSON yes, codes no), 20 pitfalls (1/3).
**Not started (7):** 10 local→global rotation, 11 beam forces, 12 global
energies, 13 flat cells, 14 streaming, 15 binary/.vtu, 16 property-based
tests. Open questions: all still open except the pyradioss frame answer.
Leftover-minors ledger: C2 cross-block policy untouched (6th round);
padding asymmetry untouched; membrane+upper silent drop now exists for
BOTH stress and strain.

## Contract for the next fix round (M12)

1. **S1 strain path**: 2-layer branch wins over membrane; *_only flags
   (mirror the M8 stress fix); pin zug element_shell_strain against the
   Vortex golden VALUES; kill the slot-swap and delete-branch mutants.
2. **T5 finish**: never a window with zero families — move new files in
   under temp names, then swap (or restore-on-failure); case-insensitive
   family regex on Windows; locked-file test + a true in-window injection
   test (current one fails before the try block); fix the two hollow
   atomic tests; cleanup-exception must not mask the original error.
3. **S2 fill warnings**: strain/thickness/acceleration into the N1
   machinery; local-frame warning extended to strain; is_alive fill
   warning names the actual source array; rescue-over-onefilled fix
   (compute ever_alive pre-fill).
4. **lspp_check last-state for real**: resolve the state count first
   (lasso or parsing) and emit a concrete `state N;`; test asserts the
   rendered state differs from t=0 on a multi-state model.
5. **Golden manifest enforcement**: commit the manifest + a suite test
   that verifies it; add the TENSILE golden (item 18).
6. **Test hardening**: value pins for M11 arrays (1.01681, 7.94411e+06,
   zug strain gradient); rewrite the assert-less surrogate test; real
   f64-72 pin; latch pins (golden-values compare, part ids, family list);
   proper lasso mdlopt pin + correct the solid-stress pin's direction.
7. Leftovers opportunistically: solid strain + internal energy (finish
   items 7/8), warning text at the old :469-472 (2nd miss), UTF-16
   part-names, progress during the read phase, %.4g absmax, C2 policy.
8. Records: correct the 795e03a commit-message claim in plan.md
   (lspp_check state default NOT delivered in M10).

Acceptance: suite green incl. new tests failing on ebf2588; zug strain
read-back equals the Vortex golden within f32; injected in-window failure
and locked-file runs leave the pre-existing family intact byte-for-byte;
lspp_check default PNG differs from the state-1 PNG on zug.
